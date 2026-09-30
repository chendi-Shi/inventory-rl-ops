"""Checksummed context actor weights, validated metadata, and guarded orders."""

import hashlib
import json
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np

from inventory_rl.m5 import M5Series
from inventory_rl.portfolio import PortfolioConfig, allocate, context_policy_scores

CONTEXT_SCHEMA_VERSION = 1


def _sku_order_sha256(item_ids: list[str]) -> str:
    """Fingerprint the positional request contract without relying on JSON whitespace."""
    encoded = json.dumps(item_ids, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _nonnegative_number(value: object) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return bool(np.isfinite(value) and value >= 0)
    except (OverflowError, TypeError):
        return False


def _validate_manifest(manifest: dict) -> None:
    item_ids = manifest["item_ids"]
    if (not isinstance(item_ids, list) or len(item_ids) < 2 or
            any(not isinstance(item, str) or not item for item in item_ids) or
            len(set(item_ids)) != len(item_ids)):
        raise ValueError("invalid context item IDs")
    count = len(item_ids)
    means = manifest["mean_train"]
    leads = manifest["lead_time"]
    if (not isinstance(means, list) or len(means) != count or
            not all(_nonnegative_number(value) for value in means)):
        raise ValueError("invalid context training means")
    if (not isinstance(leads, list) or len(leads) != count or
            any(type(value) is not int or value not in (1, 2, 3) for value in leads)):
        raise ValueError("invalid context lead times")
    if (not isinstance(manifest["store_id"], str) or not manifest["store_id"] or
            not isinstance(manifest["source_sha256"], str) or
            not manifest["source_sha256"]):
        raise ValueError("invalid context provenance")
    economics = manifest["economics"]
    expected_fields = {field.name for field in fields(PortfolioConfig)}
    if not isinstance(economics, dict) or set(economics) != expected_fields:
        raise ValueError("invalid context economics")
    if (any(not _nonnegative_number(value) for value in economics.values()) or
            type(economics["capacity_per_sku"]) is not int):
        raise ValueError("invalid context economics")
    PortfolioConfig(**economics)
    baseline = manifest["baseline_selection"]
    if (not isinstance(baseline, dict) or set(baseline) !=
            {"cover", "recent_sales", "beta"} or
            type(baseline["recent_sales"]) is not bool or
            not _nonnegative_number(baseline["cover"]) or
            not _nonnegative_number(baseline["beta"])):
        raise ValueError("invalid context baseline")
    if type(manifest["promotion_eligible"]) is not bool:
        raise ValueError("invalid context promotion decision")
    expected = "context_actor_rl" if manifest["promotion_eligible"] else "strong_rule"
    if manifest["active_policy"] != expected:
        raise ValueError("context release decision conflicts with policy")


def _quantity_array(value: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric quantities") from exc
    if array.shape != shape:
        raise ValueError(f"{name} has invalid dimensions")
    if (not np.isfinite(array).all() or (array < 0).any() or
            (array >= 2**53).any() or (array != np.floor(array)).any()):
        raise ValueError(f"{name} must contain finite nonnegative integer quantities")
    return array


def save_context_model(theta: np.ndarray, data: M5Series, config: PortfolioConfig,
                       directory: Path, promotion_eligible: bool, *,
                       baseline_cover: float, baseline_recent: bool,
                       baseline_beta: float) -> None:
    if theta.shape != (9,) or not np.isfinite(theta).all():
        raise ValueError("invalid context actor coefficients")
    if type(promotion_eligible) is not bool:
        raise ValueError("promotion_eligible must be a boolean")
    if (type(baseline_recent) is not bool or
            not _nonnegative_number(baseline_cover) or
            not _nonnegative_number(baseline_beta)):
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
        "promotion_eligible": promotion_eligible,
        "active_policy": "context_actor_rl" if promotion_eligible else "strong_rule",
    }
    _validate_manifest(manifest)
    (directory / "context_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def load_context_model(directory: Path) -> tuple[np.ndarray, dict]:
    manifest_bytes = (directory / "context_manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    if not isinstance(manifest, dict):
        raise TypeError("invalid context manifest")
    if manifest.get("schema_version") != CONTEXT_SCHEMA_VERSION:
        raise ValueError("unsupported context schema")
    model = directory / "context_model.npz"
    if hashlib.sha256(model.read_bytes()).hexdigest() != manifest["model_sha256"]:
        raise ValueError("context actor checksum mismatch")
    _validate_manifest(manifest)
    with np.load(model, allow_pickle=False) as weights:
        if set(weights.files) != {"theta"} or weights["theta"].shape != (9,) or (
            not np.isfinite(weights["theta"]).all()
        ):
            raise ValueError("invalid context actor weights")
        theta = weights["theta"].copy()
    manifest["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    manifest["sku_order_sha256"] = _sku_order_sha256(manifest["item_ids"])
    return theta, manifest


def recommend_context(theta: np.ndarray, manifest: dict, *, day: int,
                      stock: np.ndarray, pipeline: np.ndarray,
                      last_sales: np.ndarray, force_actor: bool = False,
                      item_ids: list[str] | None = None,
                      sku_order_sha256: str | None = None) -> dict:
    count = len(manifest["item_ids"])
    if type(day) is not int or day < 0:
        raise ValueError("day must be a nonnegative integer")
    if item_ids is None and sku_order_sha256 is None:
        raise ValueError("item_ids or sku_order_sha256 is required")
    if item_ids is not None and item_ids != manifest["item_ids"]:
        raise ValueError("item_ids must match the model SKU order exactly")
    expected_order = manifest.get("sku_order_sha256") or _sku_order_sha256(manifest["item_ids"])
    if sku_order_sha256 is not None and sku_order_sha256 != expected_order:
        raise ValueError("sku_order_sha256 does not match the model SKU order")
    stock = _quantity_array(stock, (count,), "stock")
    pipeline = _quantity_array(pipeline, (3, count), "pipeline")
    last_sales = _quantity_array(last_sales, (count, 7), "last_sales")
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
            "day": day,
            "orders": {item: int(qty) for item, qty in zip(manifest["item_ids"], orders)},
            "spend": float(orders.sum() * config.unit_cost),
            "model_sha256": manifest["model_sha256"],
            "manifest_sha256": manifest.get("manifest_sha256"),
            "source_sha256": manifest["source_sha256"],
            "sku_order_sha256": expected_order,
            "sku_order_verified": item_ids is not None or sku_order_sha256 is not None}
