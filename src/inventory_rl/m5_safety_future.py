"""Frozen TX_3 safety-stock comparator on the later M5 calendar period.

The protocol is fixed in docs/SAFETY_FUTURE_COMPARISON_PLAN.md. No candidate
selection, retraining, or promotion decision is performed here.
"""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from inventory_rl.m5 import M5Series, load_m5
from inventory_rl.m5_experiment import block_bootstrap_ci
from inventory_rl.m5_future_holdout import (
    EVALUATION_MD5,
    TX3_MODEL_SHA256,
    VALIDATION_SHA256,
    FrozenSpec,
    _check_close,
    _digest,
    _evaluate_store,
    _load_frozen,
)
from inventory_rl.m5_safety_stock import safety_stock_scores, training_safety_premium
from inventory_rl.portfolio import PortfolioConfig, PortfolioEnv, allocate

SAFETY_QUANTILE = 0.70
SAFETY_COVER = 2
SAFETY_RECENT = True
SAFETY_BETA = 1.0


def _verify_retrospective(retro: dict, published: dict, old: M5Series,
                          manifest: dict, config: PortfolioConfig,
                          premium: np.ndarray, *, warmup_start: int,
                          tail_start: int, old_sha256: str) -> None:
    meta = retro["dataset"]
    selection = retro["safety_stock_selection"]
    if (meta["source_sha256"] != old_sha256 or
            meta["store_id"] != "TX_3" or meta["sku_count"] != old.n_sku or
            meta["train_days"] != [1, old.train_end] or
            meta["validation_days"] != [old.train_end + 1, old.validation_end] or
            meta["retrospective_days"] != [warmup_start + 1, tail_start] or
            retro["economics"] != asdict(config) or
            retro["frozen_actor_model_sha256"] != manifest["model_sha256"] or
            retro["frozen_rule_selection"] != manifest["baseline_selection"]):
        raise ValueError("safety retrospective source or frozen policy mismatch")
    if (selection["quantile"] != SAFETY_QUANTILE or
            type(selection["cover"]) is not int or selection["cover"] != SAFETY_COVER or
            selection["recent_sales"] is not SAFETY_RECENT or
            selection["beta"] != SAFETY_BETA):
        raise ValueError("safety rule differs from predeclared validation selection")
    published_premium = np.asarray(retro["safety_premium_units"])
    if not np.array_equal(published_premium, premium):
        raise ValueError("safety premium differs from training-only published values")
    for retro_name, original_name in (("context_actor_rl", "actor"),
                                      ("strong_rule", "strong_rule")):
        retrospective = retro["retrospective"][retro_name]
        original = published["test"][original_name]
        if len(retrospective["daily_profit"]) != tail_start - warmup_start:
            raise ValueError("safety retrospective warm-up length mismatch")
        for actual, expected in zip(retrospective["daily_profit"],
                                    original["daily_profit"]):
            _check_close(actual, expected, f"retrospective {retro_name} daily profit")


def _replay_safety_continuously(data: M5Series, config: PortfolioConfig,
                                premium: np.ndarray, *, warmup_start: int,
                                tail_start: int, end: int) -> dict:
    env = PortfolioEnv(data, config)
    env.reset(warmup_start, end)
    warmup_daily_profit = []
    daily_profit = []
    daily_spend = []
    daily_demand = []
    daily_sold = []
    while env.day < end:
        scores = safety_stock_scores(env, premium, cover=SAFETY_COVER,
                                     recent=SAFETY_RECENT, beta=SAFETY_BETA)
        orders = allocate(scores, stock=env.stock, pipeline=env.pipeline.sum(axis=0),
                          budget=env.budget, capacity=env.capacity,
                          unit_cost=config.unit_cost)
        day = env.day
        _, reward, _, info = env.step(orders)
        profit = float(reward.sum())
        if day < tail_start:
            warmup_daily_profit.append(profit)
        else:
            daily_profit.append(profit)
            daily_spend.append(info["spend"])
            daily_demand.append(info["demand"])
            daily_sold.append(info["sold"])
    return {
        "warmup_daily_profit": warmup_daily_profit,
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


def _run_safety_future(validation_path: Path, evaluation_path: Path, output: Path,
                       spec: FrozenSpec, retrospective_path: Path, *,
                       expected_old_sha256: str, expected_new_md5: str,
                       train_end: int, validation_end: int,
                       warmup_start: int, tail_start: int, end: int) -> dict:
    """Internal fixture-capable runner; public entry point fixes official hashes and days."""
    old_sha = _digest(validation_path, "sha256")
    if old_sha != expected_old_sha256:
        raise ValueError("old validation CSV SHA-256 mismatch")
    new_md5 = _digest(evaluation_path, "md5")
    if new_md5 != expected_new_md5:
        raise ValueError("official evaluation CSV MD5 mismatch")
    if (spec.store_id != "TX_3" or spec.family != "context" or
            warmup_start != validation_end or
            not 28 < train_end < validation_end < tail_start < end or
            end - tail_start < 7):
        raise ValueError("invalid frozen safety future-period protocol")
    new_sha = _digest(evaluation_path, "sha256")
    theta, manifest, published = _load_frozen(spec)
    old = load_m5(validation_path, store_id="TX_3", sku_count=len(manifest["item_ids"]),
                  train_end=train_end, validation_end=validation_end)
    new = load_m5(evaluation_path, store_id="TX_3", sku_count=len(manifest["item_ids"]),
                  train_end=train_end, validation_end=validation_end)
    frozen = _evaluate_store(spec, old, new, theta, manifest, published,
                             warmup_start=warmup_start, tail_start=tail_start,
                             end=end, old_sha256=old_sha)
    config = PortfolioConfig(**manifest["economics"])
    lead = 1 + np.arange(old.n_sku) % 3
    premium = training_safety_premium(old, lead, cover=SAFETY_COVER,
                                      quantile=SAFETY_QUANTILE)
    retrospective = json.loads(retrospective_path.read_text(encoding="utf-8"))
    _verify_retrospective(retrospective, published, old, manifest, config, premium,
                          warmup_start=warmup_start, tail_start=tail_start,
                          old_sha256=old_sha)
    safety = _replay_safety_continuously(new, config, premium,
                                         warmup_start=warmup_start,
                                         tail_start=tail_start, end=end)
    published_safety = retrospective["retrospective"]["safety_stock"]
    if len(published_safety["daily_profit"]) != tail_start - warmup_start:
        raise ValueError("safety retrospective warm-up length mismatch")
    for day, (actual, expected) in enumerate(zip(safety["warmup_daily_profit"],
                                                 published_safety["daily_profit"]),
                                             start=warmup_start + 1):
        _check_close(actual, expected, f"safety day {day} profit")
    _check_close(sum(safety["warmup_daily_profit"]), published_safety["profit"],
                 "safety warm-up total")
    del safety["warmup_daily_profit"]
    actor = frozen["actor"]
    rule = frozen["rule"]

    def paired_against(other: dict, name: str) -> dict:
        delta = safety["profit"] - other["profit"]
        return {"comparator": name, "simulated_profit_difference": delta,
                "uplift_pct_of_comparator_profit": (
                    100 * delta / other["profit"] if other["profit"] else None
                ),
                "paired_profit_difference_ci95_7day": block_bootstrap_ci(
                    safety["daily_profit"], other["daily_profit"]
                )}

    paired_daily = [
        {"m5_day": tail_start + index + 1,
         "safety_profit": safety_profit,
         "actor_profit": actor_profit,
         "rule_profit": rule_profit,
         "safety_minus_actor": safety_profit - actor_profit,
         "safety_minus_rule": safety_profit - rule_profit}
        for index, (safety_profit, actor_profit, rule_profit) in enumerate(
            zip(safety["daily_profit"], actor["daily_profit"], rule["daily_profit"]))
    ]
    report = {
        "analysis_status": "predeclared frozen safety-rule later-calendar-period simulation",
        "protocol": "docs/SAFETY_FUTURE_COMPARISON_PLAN.md",
        "data": {"validation_sha256": old_sha, "evaluation_md5": new_md5,
                 "evaluation_sha256": new_sha,
                 "warmup_days": [warmup_start + 1, tail_start],
                 "scored_days": [tail_start + 1, end]},
        "store_id": "TX_3",
        "model_sha256": manifest["model_sha256"],
        "sku_order": list(new.item_ids),
        "economics": asdict(config),
        "safety_stock_selection": {"quantile": SAFETY_QUANTILE,
                                   "cover": SAFETY_COVER,
                                   "recent_sales": SAFETY_RECENT,
                                   "beta": SAFETY_BETA},
        "safety_premium_units": premium.tolist(),
        "safety_stock": safety,
        "context_actor_rl": actor,
        "strong_rule": rule,
        "comparisons": {"safety_minus_actor": paired_against(actor, "context_actor_rl"),
                        "safety_minus_rule": paired_against(rule, "strong_rule")},
        "paired_daily_profit": paired_daily,
        "release_decision": "none; previously published policy bundles remain unchanged",
        "caveat": "Observed sales are a censored demand proxy and all economics are simulated.",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def run_safety_future(validation_path: Path, evaluation_path: Path, output: Path) -> dict:
    """Score the locked safety comparator against the frozen TX_3 actor and rule."""
    spec = FrozenSpec("TX_3", "context", Path("models/tx3-context"),
                      Path("docs/m5-tx3-context-report.json"), TX3_MODEL_SHA256)
    return _run_safety_future(
        validation_path, evaluation_path, output, spec,
        Path("docs/m5-safety-stock-retro-report.json"),
        expected_old_sha256=VALIDATION_SHA256,
        expected_new_md5=EVALUATION_MD5,
        train_end=1700, validation_end=1800,
        warmup_start=1800, tail_start=1913, end=1941
    )
