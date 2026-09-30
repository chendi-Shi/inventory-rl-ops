"""Versioned model bundle for the multi-SKU decision service."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from inventory_rl.agent import DQNAgent
from inventory_rl.m5 import M5Series
from inventory_rl.portfolio import (
    ORDER_CHOICES,
    PortfolioConfig,
    allocate,
    base_stock_scores,
    residual_scores,
)

SCHEMA_VERSION = 3


def save_portfolio_model(agent: DQNAgent, data: M5Series, config: PortfolioConfig,
                         directory: Path, promotion_eligible: bool, *,
                         baseline_recent: bool, baseline_cover: float,
                         hybrid_alpha: float | None = None) -> None:
    if not isinstance(baseline_recent, bool) or not np.isfinite(baseline_cover) or baseline_cover < 0:
        raise ValueError("invalid baseline selection")
    if hybrid_alpha is not None and (not np.isfinite(hybrid_alpha) or hybrid_alpha < 0):
        raise ValueError("invalid residual alpha")
    directory.mkdir(parents=True, exist_ok=True)
    model = directory / "portfolio_model.npz"
    np.savez_compressed(model, **agent.online.params)
    mean_train = data.sales[:, max(0, data.train_end - 365):data.train_end].mean(axis=1)
    manifest = {
        "schema_version": SCHEMA_VERSION if hybrid_alpha is not None else 2,
        "algorithm": "shared-network factored Double DQN with budget allocator",
        "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "source_sha256": data.source_sha256,
        "store_id": data.store_id,
        "item_ids": list(data.item_ids),
        "mean_train": mean_train.tolist(),
        "lead_time": (1 + np.arange(data.n_sku) % 3).tolist(),
        "economics": asdict(config),
        "promotion_eligible": promotion_eligible,
        "active_policy": ("hybrid_rl" if hybrid_alpha is not None else "rl")
        if promotion_eligible else "base_stock",
        "baseline_selection": {"recent_sales": baseline_recent, "cover": baseline_cover},
    }
    if hybrid_alpha is not None:
        manifest["hybrid_alpha"] = hybrid_alpha
    (directory / "portfolio_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def load_portfolio_model(directory: Path) -> tuple[DQNAgent, dict]:
    manifest = json.loads((directory / "portfolio_manifest.json").read_text(encoding="utf-8"))
    version = manifest.get("schema_version")
    if version not in (2, SCHEMA_VERSION):
        raise ValueError("unsupported portfolio model schema")
    model = directory / "portfolio_model.npz"
    if hashlib.sha256(model.read_bytes()).hexdigest() != manifest["model_sha256"]:
        raise ValueError("portfolio model checksum mismatch")
    count = len(manifest["item_ids"])
    if count < 2 or len(set(manifest["item_ids"])) != count:
        raise ValueError("invalid item IDs")
    if len(manifest["mean_train"]) != count or len(manifest["lead_time"]) != count:
        raise ValueError("invalid portfolio metadata dimensions")
    baseline = manifest["baseline_selection"]
    if not isinstance(baseline["recent_sales"], bool) or not isinstance(baseline["cover"], (int, float)):
        raise TypeError("invalid baseline metadata")
    if not np.isfinite(baseline["cover"]) or baseline["cover"] < 0:
        raise ValueError("invalid baseline coverage")
    if version == SCHEMA_VERSION:
        alpha = manifest["hybrid_alpha"]
        if not isinstance(alpha, (int, float)):
            raise TypeError("invalid residual alpha type")
        if not np.isfinite(alpha) or alpha < 0:
            raise ValueError("invalid residual alpha")
    expected_policy = ("hybrid_rl" if version == SCHEMA_VERSION else "rl") if (
        manifest["promotion_eligible"]
    ) else "base_stock"
    if manifest["active_policy"] != expected_policy:
        raise ValueError("active policy conflicts with promotion decision")
    PortfolioConfig(**manifest["economics"])
    agent = DQNAgent(8, len(ORDER_CHOICES))
    with np.load(model, allow_pickle=False) as weights:
        if set(weights.files) != set(agent.online.params):
            raise ValueError("unexpected portfolio parameter names")
        for key, expected in agent.online.params.items():
            if weights[key].shape != expected.shape or not np.isfinite(weights[key]).all():
                raise ValueError(f"invalid portfolio parameter: {key}")
            expected[...] = weights[key]
    agent.target.copy_from(agent.online)
    return agent, manifest


def recommend_portfolio(agent: DQNAgent, manifest: dict, *, day: int,
                        stock: np.ndarray, pipeline: np.ndarray,
                        last_sales: np.ndarray, force_rl: bool = False) -> dict:
    n_sku = len(manifest["item_ids"])
    if day < 0 or stock.shape != (n_sku,) or pipeline.shape != (3, n_sku) or (
        last_sales.shape != (n_sku, 7)
    ):
        raise ValueError("invalid portfolio request dimensions")
    if not np.isfinite(stock).all() or not np.isfinite(pipeline).all() or not np.isfinite(last_sales).all():
        raise ValueError("portfolio inputs must be finite")
    if (stock < 0).any() or (pipeline < 0).any() or (last_sales < 0).any():
        raise ValueError("portfolio inputs must be nonnegative")
    config = PortfolioConfig(**manifest["economics"])
    budget = config.budget_per_sku * n_sku
    capacity = config.capacity_per_sku * n_sku
    lead = np.asarray(manifest["lead_time"])
    mean_train = np.asarray(manifest["mean_train"])
    is_hybrid = manifest["schema_version"] == SCHEMA_VERSION
    if force_rl:
        policy_type = ("hybrid_rl" if is_hybrid else "rl") if (
            manifest["promotion_eligible"]
        ) else ("hybrid_unpromoted" if is_hybrid else "rl_unpromoted")
    else:
        policy_type = manifest["active_policy"]
    if policy_type == "base_stock":
        baseline = manifest["baseline_selection"]
        scores = base_stock_scores(stock, pipeline.sum(axis=0), last_sales, mean_train,
                                   lead, cover=baseline["cover"], recent=baseline["recent_sales"])
    else:
        state = np.column_stack((
            np.minimum(stock / 40.0, 5),
            np.minimum(pipeline.sum(axis=0) / 40.0, 5),
            np.minimum(last_sales[:, -1] / 40.0, 5),
            np.minimum(last_sales.mean(axis=1) / 40.0, 5),
            np.minimum(mean_train / 40.0, 5),
            lead / 3.0,
            np.full(n_sku, np.sin(2 * np.pi * day / 7)),
            np.full(n_sku, np.cos(2 * np.pi * day / 7)),
        ))
        q_values = agent.online.predict(state)
        if is_hybrid:
            baseline = manifest["baseline_selection"]
            rule = base_stock_scores(stock, pipeline.sum(axis=0), last_sales, mean_train,
                                     lead, cover=baseline["cover"],
                                     recent=baseline["recent_sales"])
            scores = residual_scores(rule, q_values, manifest["hybrid_alpha"])
        else:
            scores = q_values
    orders = allocate(scores, stock=stock, pipeline=pipeline.sum(axis=0),
                      budget=budget, capacity=capacity, unit_cost=config.unit_cost)
    return {"store_id": manifest["store_id"],
            "policy_type": policy_type,
            "orders": {item: int(qty) for item, qty in zip(manifest["item_ids"], orders)},
            "spend": float(orders.sum() * config.unit_cost),
            "model_sha256": manifest["model_sha256"]}
