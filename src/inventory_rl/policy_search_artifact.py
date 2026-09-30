"""Versioned actor bundle and guarded portfolio recommendations."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from inventory_rl.m5 import M5Series
from inventory_rl.portfolio import PortfolioConfig, allocate, policy_search_scores

ACTOR_SCHEMA_VERSION = 1


def save_actor_model(theta: np.ndarray, data: M5Series, config: PortfolioConfig,
                     directory: Path, promotion_eligible: bool, *,
                     baseline_cover: float, baseline_recent: bool) -> None:
    if theta.shape != (5,) or not np.isfinite(theta).all():
        raise ValueError("invalid actor coefficients")
    if not np.isfinite(baseline_cover) or baseline_cover < 0 or (
        not isinstance(baseline_recent, bool)
    ):
        raise ValueError("invalid baseline configuration")
    directory.mkdir(parents=True, exist_ok=True)
    model = directory / "actor_model.npz"
    np.savez_compressed(model, theta=theta)
    mean_train = data.sales[:, max(0, data.train_end - 365):data.train_end].mean(axis=1)
    manifest = {
        "schema_version": ACTOR_SCHEMA_VERSION,
        "algorithm": "cross-entropy episodic portfolio policy search",
        "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "source_sha256": data.source_sha256,
        "store_id": data.store_id,
        "item_ids": list(data.item_ids),
        "mean_train": mean_train.tolist(),
        "lead_time": (1 + np.arange(data.n_sku) % 3).tolist(),
        "economics": asdict(config),
        "baseline_selection": {"cover": baseline_cover,
                               "recent_sales": baseline_recent},
        "promotion_eligible": bool(promotion_eligible),
        "active_policy": "policy_search_rl" if promotion_eligible else "base_stock",
    }
    (directory / "actor_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def load_actor_model(directory: Path) -> tuple[np.ndarray, dict]:
    manifest = json.loads((directory / "actor_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != ACTOR_SCHEMA_VERSION:
        raise ValueError("unsupported actor schema")
    model = directory / "actor_model.npz"
    if hashlib.sha256(model.read_bytes()).hexdigest() != manifest["model_sha256"]:
        raise ValueError("actor checksum mismatch")
    count = len(manifest["item_ids"])
    if count < 2 or len(set(manifest["item_ids"])) != count or (
        len(manifest["mean_train"]) != count or len(manifest["lead_time"]) != count
    ):
        raise ValueError("invalid actor portfolio dimensions")
    baseline = manifest["baseline_selection"]
    if not isinstance(baseline["recent_sales"], bool) or (
        not np.isfinite(baseline["cover"]) or baseline["cover"] < 0
    ):
        raise ValueError("invalid actor baseline")
    expected = "policy_search_rl" if manifest["promotion_eligible"] else "base_stock"
    if manifest["active_policy"] != expected:
        raise ValueError("actor release decision conflicts with policy")
    PortfolioConfig(**manifest["economics"])
    with np.load(model, allow_pickle=False) as weights:
        if set(weights.files) != {"theta"} or weights["theta"].shape != (5,) or (
            not np.isfinite(weights["theta"]).all()
        ):
            raise ValueError("invalid actor weights")
        theta = weights["theta"].copy()
    return theta, manifest


def recommend_actor(theta: np.ndarray, manifest: dict, *, day: int,
                    stock: np.ndarray, pipeline: np.ndarray,
                    last_sales: np.ndarray, force_actor: bool = False) -> dict:
    count = len(manifest["item_ids"])
    if day < 0 or stock.shape != (count,) or pipeline.shape != (3, count) or (
        last_sales.shape != (count, 7)
    ):
        raise ValueError("invalid actor request dimensions")
    if not all(np.isfinite(x).all() and (x >= 0).all()
               for x in (stock, pipeline, last_sales)):
        raise ValueError("actor inputs must be finite and nonnegative")
    config = PortfolioConfig(**manifest["economics"])
    baseline = manifest["baseline_selection"]
    policy_type = ("policy_search_rl" if manifest["promotion_eligible"] else
                   "policy_search_unpromoted") if force_actor else manifest["active_policy"]
    active_theta = theta if policy_type != "base_stock" else np.zeros(5)
    scores = policy_search_scores(
        stock, pipeline.sum(axis=0), last_sales,
        np.asarray(manifest["mean_train"]), np.asarray(manifest["lead_time"]),
        day, active_theta, cover=baseline["cover"], recent=baseline["recent_sales"]
    )
    orders = allocate(scores, stock=stock, pipeline=pipeline.sum(axis=0),
                      budget=config.budget_per_sku * count,
                      capacity=config.capacity_per_sku * count,
                      unit_cost=config.unit_cost)
    return {"store_id": manifest["store_id"], "policy_type": policy_type,
            "orders": {item: int(qty) for item, qty in zip(manifest["item_ids"], orders)},
            "spend": float(orders.sum() * config.unit_cost),
            "model_sha256": manifest["model_sha256"]}
