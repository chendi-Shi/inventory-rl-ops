"""Integrity-checked context actor bundle and guarded recommendations."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from inventory_rl.m5 import M5Series
from inventory_rl.portfolio import PortfolioConfig, allocate, context_policy_scores

CONTEXT_SCHEMA_VERSION = 1


def save_context_model(theta: np.ndarray, data: M5Series, config: PortfolioConfig,
                       directory: Path, promotion_eligible: bool, *,
                       baseline_cover: float, baseline_recent: bool,
                       baseline_beta: float) -> None:
    if theta.shape != (9,) or not np.isfinite(theta).all():
        raise ValueError("invalid context actor coefficients")
    if not isinstance(baseline_recent, bool) or any(
        not np.isfinite(x) or x < 0 for x in (baseline_cover, baseline_beta)
    ):
        raise ValueError("invalid context baseline")
    directory.mkdir(parents=True, exist_ok=True)
    model = directory / "context_model.npz"
    np.savez_compressed(model, theta=theta)
    mean_train = data.sales[:, max(0, data.train_end - 365):data.train_end].mean(axis=1)
    manifest = {
        "schema_version": CONTEXT_SCHEMA_VERSION,
        "algorithm": "context-aware whole-portfolio CEM policy search",
        "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "source_sha256": data.source_sha256,
        "store_id": data.store_id,
        "item_ids": list(data.item_ids),
        "mean_train": mean_train.tolist(),
        "lead_time": (1 + np.arange(data.n_sku) % 3).tolist(),
        "economics": asdict(config),
        "baseline_selection": {"cover": baseline_cover, "recent_sales": baseline_recent,
                               "beta": baseline_beta},
        "promotion_eligible": bool(promotion_eligible),
        "active_policy": "context_actor_rl" if promotion_eligible else "strong_rule",
    }
    (directory / "context_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def load_context_model(directory: Path) -> tuple[np.ndarray, dict]:
    manifest = json.loads((directory / "context_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != CONTEXT_SCHEMA_VERSION:
        raise ValueError("unsupported context schema")
    model = directory / "context_model.npz"
    if hashlib.sha256(model.read_bytes()).hexdigest() != manifest["model_sha256"]:
        raise ValueError("context actor checksum mismatch")
    count = len(manifest["item_ids"])
    if count < 2 or len(set(manifest["item_ids"])) != count or (
        len(manifest["mean_train"]) != count or len(manifest["lead_time"]) != count
    ):
        raise ValueError("invalid context portfolio dimensions")
    baseline = manifest["baseline_selection"]
    if not isinstance(baseline["recent_sales"], bool) or any(
        not np.isfinite(x) or x < 0 for x in (baseline["cover"], baseline["beta"])
    ):
        raise ValueError("invalid context baseline")
    expected = "context_actor_rl" if manifest["promotion_eligible"] else "strong_rule"
    if manifest["active_policy"] != expected:
        raise ValueError("context release decision conflicts with policy")
    PortfolioConfig(**manifest["economics"])
    with np.load(model, allow_pickle=False) as weights:
        if set(weights.files) != {"theta"} or weights["theta"].shape != (9,) or (
            not np.isfinite(weights["theta"]).all()
        ):
            raise ValueError("invalid context actor weights")
        theta = weights["theta"].copy()
    return theta, manifest


def recommend_context(theta: np.ndarray, manifest: dict, *, day: int,
                      stock: np.ndarray, pipeline: np.ndarray,
                      last_sales: np.ndarray, force_actor: bool = False) -> dict:
    count = len(manifest["item_ids"])
    if day < 0 or stock.shape != (count,) or pipeline.shape != (3, count) or (
        last_sales.shape != (count, 7)
    ):
        raise ValueError("invalid context request dimensions")
    if not all(np.isfinite(x).all() and (x >= 0).all()
               for x in (stock, pipeline, last_sales)):
        raise ValueError("context inputs must be finite and nonnegative")
    config = PortfolioConfig(**manifest["economics"])
    baseline = manifest["baseline_selection"]
    policy_type = ("context_actor_rl" if manifest["promotion_eligible"] else
                   "context_actor_unpromoted") if force_actor else manifest["active_policy"]
    active_theta = theta if policy_type != "strong_rule" else np.zeros(9)
    scores = context_policy_scores(
        stock, pipeline.sum(axis=0), last_sales,
        np.asarray(manifest["mean_train"]), np.asarray(manifest["lead_time"]),
        day, config.budget_per_sku * count, config.unit_cost, active_theta,
        cover=baseline["cover"], recent=baseline["recent_sales"], beta=baseline["beta"]
    )
    orders = allocate(scores, stock=stock, pipeline=pipeline.sum(axis=0),
                      budget=config.budget_per_sku * count,
                      capacity=config.capacity_per_sku * count,
                      unit_cost=config.unit_cost)
    return {"store_id": manifest["store_id"], "policy_type": policy_type,
            "orders": {item: int(qty) for item, qty in zip(manifest["item_ids"], orders)},
            "spend": float(orders.sum() * config.unit_cost),
            "model_sha256": manifest["model_sha256"]}
