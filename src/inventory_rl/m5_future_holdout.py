"""Frozen WI_1 and TX_3 policy check on M5 days 1914–1941.

The procedure is fixed in docs/NEW_TIME_HOLDOUT_PLAN.md. This module never
trains a policy or uses the future period to choose a model, rule, or SKU.
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from inventory_rl.context_artifact import load_context_model
from inventory_rl.m5 import M5Series, load_m5
from inventory_rl.m5_experiment import block_bootstrap_ci
from inventory_rl.policy_search_artifact import load_actor_model
from inventory_rl.portfolio import (
    PortfolioConfig,
    PortfolioEnv,
    allocate,
    base_stock_scores,
    context_policy_scores,
    policy_search_scores,
)

VALIDATION_SHA256 = "f368e66ed1dbecb48b2cc8fc589bf68b3deddbbb36bf5c88b4d6d0a09b9b6724"
EVALUATION_MD5 = "b806dfc9f30a745102b708c09951f6aa"
WI1_MODEL_SHA256 = "71ac12c3f3ddf294f7070cf2843d0b89b7202a33f395eca026455df923a18117"
TX3_MODEL_SHA256 = "f7abb861d378a6eecc557a3a371ad2de726a32fee1b0ef093a57bde7ed9755e5"


@dataclass(frozen=True)
class FrozenSpec:
    store_id: str
    family: str
    model_dir: Path
    report_path: Path
    model_sha256: str


def _digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_close(actual: float, expected: float, label: str) -> None:
    if not np.isclose(actual, expected, atol=1e-7, rtol=0):
        raise ValueError(f"{label} does not match published warm-up result")


def _load_frozen(spec: FrozenSpec) -> tuple[np.ndarray, dict, dict]:
    if spec.family == "policy_search":
        theta, manifest = load_actor_model(spec.model_dir)
    elif spec.family == "context":
        theta, manifest = load_context_model(spec.model_dir)
    else:
        raise ValueError("unknown frozen policy family")
    if manifest["model_sha256"] != spec.model_sha256:
        raise ValueError(f"{spec.store_id} model SHA-256 differs from predeclared bundle")
    report = json.loads(spec.report_path.read_text(encoding="utf-8"))
    return theta, manifest, report


def _verify_frozen(spec: FrozenSpec, old: M5Series, new: M5Series,
                   theta: np.ndarray, manifest: dict, report: dict,
                   *, warmup_start: int, tail_start: int, end: int,
                   old_sha256: str) -> tuple[PortfolioConfig, dict]:
    meta = report["dataset"]
    if (old.sales.shape[1] != tail_start or new.sales.shape[1] != end or
            old.store_id != spec.store_id or new.store_id != spec.store_id or
            old.source_sha256 != old_sha256 or
            manifest["source_sha256"] != old_sha256 or
            meta["source_sha256"] != old_sha256 or
            meta["store_id"] != spec.store_id or
            meta["train_days"] != [1, old.train_end] or
            meta["validation_days"] != [old.train_end + 1, old.validation_end] or
            meta["test_days"] != [warmup_start + 1, tail_start]):
        raise ValueError(f"{spec.store_id} source or published split mismatch")
    if (old.item_ids != new.item_ids or
            list(old.item_ids) != manifest["item_ids"] or
            list(old.item_ids) != meta["selected_item_ids"] or
            meta["sku_count"] != old.n_sku):
        raise ValueError(f"{spec.store_id} frozen SKU order mismatch")
    if not np.array_equal(old.sales, new.sales[:, :tail_start]):
        raise ValueError(f"{spec.store_id} d1–d{tail_start} sales prefix mismatch")
    config = PortfolioConfig(**manifest["economics"])
    baseline = manifest["baseline_selection"]
    baseline_keys = ("cover", "recent_sales", "beta") if spec.family == "context" else (
        "cover", "recent_sales"
    )
    published_baseline = {key: report["baseline_selection"][key] for key in baseline_keys}
    expected_mean = old.sales[:, max(0, old.train_end - 365):old.train_end].mean(axis=1)
    expected_lead = 1 + np.arange(old.n_sku) % 3
    if (manifest["economics"] != report["economics"] or
            baseline != published_baseline or
            manifest["active_policy"] != report["test"]["active_policy"] or
            manifest["promotion_eligible"] != report["test"]["promotion_eligible"] or
            not np.array_equal(theta, np.asarray(report["model_selection"]["selected_theta"])) or
            not np.array_equal(manifest["mean_train"], expected_mean) or
            not np.array_equal(manifest["lead_time"], expected_lead)):
        raise ValueError(f"{spec.store_id} frozen bundle and published report disagree")
    for name in ("actor", "strong_rule" if spec.family == "context" else "base_stock"):
        if len(report["test"][name]["daily_profit"]) != tail_start - warmup_start:
            raise ValueError(f"{spec.store_id} published warm-up length mismatch")
    return config, baseline


def _replay_continuously(data: M5Series, config: PortfolioConfig, theta: np.ndarray,
                         baseline: dict, *, family: str, actor: bool,
                         warmup_start: int, tail_start: int, end: int) -> dict:
    env = PortfolioEnv(data, config)
    env.reset(warmup_start, end)
    warmup_profit = []
    daily_profit = []
    daily_spend = []
    daily_demand = []
    daily_sold = []
    while env.day < end:
        position = env.pipeline.sum(axis=0)
        if actor and family == "policy_search":
            scores = policy_search_scores(
                env.stock, position, env.last_sales, env.mean_train, env.lead,
                env.day, theta, cover=baseline["cover"], recent=baseline["recent_sales"]
            )
        elif actor and family == "context":
            scores = context_policy_scores(
                env.stock, position, env.last_sales, env.mean_train, env.lead,
                env.day, env.budget, config.unit_cost, theta,
                cover=baseline["cover"], recent=baseline["recent_sales"],
                beta=baseline["beta"]
            )
        else:
            scores = base_stock_scores(
                env.stock, position, env.last_sales, env.mean_train, env.lead,
                cover=baseline["cover"], recent=baseline["recent_sales"],
                beta=baseline.get("beta", 1.0)
            )
        orders = allocate(scores, stock=env.stock, pipeline=position,
                          budget=env.budget, capacity=env.capacity,
                          unit_cost=config.unit_cost)
        day = env.day
        _, reward, _, info = env.step(orders)
        profit = float(reward.sum())
        if day < tail_start:
            warmup_profit.append(profit)
        else:
            daily_profit.append(profit)
            daily_spend.append(info["spend"])
            daily_demand.append(info["demand"])
            daily_sold.append(info["sold"])
    return {
        "warmup_daily_profit": warmup_profit,
        "profit": float(sum(daily_profit)),
        "daily_profit": daily_profit,
        "daily_spend": daily_spend,
        "daily_demand": daily_demand,
        "daily_sold": daily_sold,
        "fill_rate": float(sum(daily_sold) / max(sum(daily_demand), 1)),
        "mean_daily_spend": float(sum(daily_spend) / (end - tail_start)),
        "p10_daily_profit": float(np.quantile(daily_profit, 0.1)),
        "days": end - tail_start,
        "sku_count": data.n_sku,
    }


def _evaluate_store(spec: FrozenSpec, old: M5Series, new: M5Series,
                    theta: np.ndarray, manifest: dict, report: dict,
                    *, warmup_start: int, tail_start: int, end: int,
                    old_sha256: str) -> dict:
    config, baseline = _verify_frozen(
        spec, old, new, theta, manifest, report,
        warmup_start=warmup_start, tail_start=tail_start,
        end=end, old_sha256=old_sha256
    )
    actor = _replay_continuously(new, config, theta, baseline,
                                 family=spec.family, actor=True,
                                 warmup_start=warmup_start,
                                 tail_start=tail_start, end=end)
    rule = _replay_continuously(new, config, theta, baseline,
                                family=spec.family, actor=False,
                                warmup_start=warmup_start,
                                tail_start=tail_start, end=end)
    rule_name = "strong_rule" if spec.family == "context" else "base_stock"
    for label, replayed in (("actor", actor), (rule_name, rule)):
        published = report["test"][label]
        for offset, (actual, expected) in enumerate(
            zip(replayed["warmup_daily_profit"], published["daily_profit"]),
            start=warmup_start + 1,
        ):
            _check_close(actual, expected, f"{spec.store_id} {label} day {offset} profit")
        _check_close(sum(replayed["warmup_daily_profit"]), published["profit"],
                     f"{spec.store_id} {label} warm-up total")
        del replayed["warmup_daily_profit"]
    difference = float(actor["profit"] - rule["profit"])
    paired_daily = [
        {"m5_day": tail_start + index + 1,
         "actor_profit": actor_profit,
         "rule_profit": rule_profit,
         "difference": actor_profit - rule_profit}
        for index, (actor_profit, rule_profit) in enumerate(
            zip(actor["daily_profit"], rule["daily_profit"]))
    ]
    return {
        "store_id": spec.store_id,
        "policy_family": spec.family,
        "model_sha256": manifest["model_sha256"],
        "manifest_source_sha256": manifest["source_sha256"],
        "sku_count": new.n_sku,
        "sku_order": list(new.item_ids),
        "baseline_selection": baseline,
        "economics": asdict(config),
        "warmup_days_reconciled": [warmup_start + 1, tail_start],
        "evaluation_days": [tail_start + 1, end],
        "actor": actor,
        "rule": rule,
        "paired_profit_difference": difference,
        "paired_profit_uplift_pct": (
            100 * difference / rule["profit"] if rule["profit"] else None
        ),
        "paired_profit_difference_ci95_7day": block_bootstrap_ci(
            actor["daily_profit"], rule["daily_profit"]
        ),
        "paired_daily_profit": paired_daily,
    }


def _run_frozen_pair(validation_path: Path, evaluation_path: Path, output: Path,
                     specs: tuple[FrozenSpec, ...], *, expected_old_sha256: str,
                     expected_new_md5: str, train_end: int, validation_end: int,
                     warmup_start: int, tail_start: int, end: int) -> dict:
    """Internal protocol engine; configurable days and hashes permit synthetic fixtures."""
    old_sha = _digest(validation_path, "sha256")
    if old_sha != expected_old_sha256:
        raise ValueError("old validation CSV SHA-256 mismatch")
    new_md5 = _digest(evaluation_path, "md5")
    if new_md5 != expected_new_md5:
        raise ValueError("official evaluation CSV MD5 mismatch")
    if (not specs or warmup_start != validation_end or
            not 28 < train_end < validation_end < tail_start < end or
            end - tail_start < 7):
        raise ValueError("invalid frozen future-period protocol")
    new_sha = _digest(evaluation_path, "sha256")
    results = []
    for spec in specs:
        theta, manifest, published = _load_frozen(spec)
        old = load_m5(validation_path, store_id=spec.store_id,
                      sku_count=len(manifest["item_ids"]),
                      train_end=train_end, validation_end=validation_end)
        new = load_m5(evaluation_path, store_id=spec.store_id,
                      sku_count=len(manifest["item_ids"]),
                      train_end=train_end, validation_end=validation_end)
        results.append(_evaluate_store(
            spec, old, new, theta, manifest, published,
            warmup_start=warmup_start, tail_start=tail_start,
            end=end, old_sha256=old_sha
        ))
    report = {
        "analysis_status": "predeclared frozen-policy later-calendar-period simulation",
        "protocol": "docs/NEW_TIME_HOLDOUT_PLAN.md",
        "data": {"validation_sha256": old_sha,
                 "evaluation_md5": new_md5,
                 "evaluation_sha256": new_sha,
                 "warmup_days": [warmup_start + 1, tail_start],
                 "scored_days": [tail_start + 1, end]},
        "policy_selection": "frozen bundles, SKU order, economics and validation-selected rules",
        "stores": results,
        "caveat": "Sales are a censored demand proxy; all profits and economics are simulated.",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def run_future_holdout(validation_path: Path, evaluation_path: Path, output: Path) -> dict:
    """Run the committed WI_1/TX_3 comparison once, without changing either bundle."""
    specs = (
        FrozenSpec("WI_1", "policy_search", Path("models/wi1-policy-search"),
                   Path("docs/m5-wi1-policy-search-report.json"), WI1_MODEL_SHA256),
        FrozenSpec("TX_3", "context", Path("models/tx3-context"),
                   Path("docs/m5-tx3-context-report.json"), TX3_MODEL_SHA256),
    )
    return _run_frozen_pair(
        validation_path, evaluation_path, output, specs,
        expected_old_sha256=VALIDATION_SHA256,
        expected_new_md5=EVALUATION_MD5,
        train_end=1700, validation_end=1800,
        warmup_start=1800, tail_start=1913, end=1941
    )
