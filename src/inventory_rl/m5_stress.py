"""Post-hoc economics stress test for frozen, already evaluated M5 policies."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from inventory_rl.context_artifact import load_context_model
from inventory_rl.m5 import M5Series, load_m5
from inventory_rl.m5_context_actor import replay_context
from inventory_rl.m5_experiment import block_bootstrap_ci, replay
from inventory_rl.m5_policy_search import replay_actor
from inventory_rl.policy_search_artifact import load_actor_model
from inventory_rl.portfolio import PortfolioConfig

# The policies, baseline parameters and SKU selection remain frozen in every scenario.
SCENARIOS = (
    ("original", {}),
    ("sale_price_20pct_lower", {"unit_price": 8.0}),
    ("sale_price_20pct_higher", {"unit_price": 12.0}),
    ("no_lost_sale_penalty", {"lost_sale_penalty": 0.0}),
    ("lost_sale_penalty_doubled", {"lost_sale_penalty": 4.0}),
    ("holding_cost_doubled", {"holding_cost": 0.30}),
    ("purchase_budget_25pct_lower", {"budget_per_sku": 24.0}),
    ("purchase_budget_25pct_higher", {"budget_per_sku": 40.0}),
    ("unit_procurement_cost_25pct_higher", {"unit_cost": 5.0}),
)


def evaluate_frozen_policy(data: M5Series, theta: np.ndarray, manifest: dict,
                           kind: str, config: PortfolioConfig) -> dict:
    """Replay an unchanged coefficient vector and unchanged validation-selected rule."""
    if data.store_id != manifest["store_id"] or list(data.item_ids) != manifest["item_ids"]:
        raise ValueError("dataset does not match frozen policy item order")
    if data.source_sha256 != manifest["source_sha256"]:
        raise ValueError("dataset checksum does not match frozen policy")
    start, end = data.validation_end, data.sales.shape[1]
    baseline = manifest["baseline_selection"]
    if kind == "context":
        actor = replay_context(data, config, start, end, theta,
                               cover=baseline["cover"], recent=baseline["recent_sales"],
                               beta=baseline["beta"])
        beta = baseline["beta"]
    elif kind == "policy_search":
        actor = replay_actor(data, config, start, end, theta,
                             cover=baseline["cover"], recent=baseline["recent_sales"])
        beta = 1.0
    else:
        raise ValueError("unknown frozen policy kind")
    rule = replay(data, config, start, end, cover=baseline["cover"],
                  recent=baseline["recent_sales"], baseline_beta=beta)
    difference = actor["profit"] - rule["profit"]
    return {"actor_profit": actor["profit"], "rule_profit": rule["profit"],
            "profit_difference": difference,
            "paired_profit_difference_ci95_7day": block_bootstrap_ci(
                actor["daily_profit"], rule["daily_profit"]),
            "uplift_pct_of_abs_rule_profit": 100 * difference / max(abs(rule["profit"]), 1),
            "actor_fill_rate": actor["fill_rate"], "rule_fill_rate": rule["fill_rate"],
            "actor_mean_daily_spend": actor["mean_daily_spend"],
            "rule_mean_daily_spend": rule["mean_daily_spend"]}


def run_stress(path: Path, output: Path, *,
               context_bundle: Path = Path("models/tx3-context"),
               policy_search_bundle: Path = Path("models/wi1-policy-search")) -> dict:
    """Evaluate frozen TX_3 and WI_1 bundles under alternative simulated economics.

    This is a retrospective diagnostic. It must not be presented as a new holdout,
    used to change the published release decision, or used to select coefficients.
    """
    bundles = (("TX_3", "context", *load_context_model(context_bundle)),
               ("WI_1", "policy_search", *load_actor_model(policy_search_bundle)))
    results = []
    for store, kind, theta, manifest in bundles:
        data = load_m5(path, store_id=store, sku_count=len(manifest["item_ids"]))
        original = PortfolioConfig(**manifest["economics"])
        rows = []
        for name, overrides in SCENARIOS:
            config = replace(original, **overrides)
            metrics = evaluate_frozen_policy(data, theta, manifest, kind, config)
            rows.append({"scenario": name, "economics": asdict(config), **metrics})
        results.append({"store_id": store, "policy_kind": kind,
                        "model_sha256": manifest["model_sha256"],
                        "source_sha256": data.source_sha256,
                        "test_days": [data.validation_end + 1, data.sales.shape[1]],
                        "scenarios": rows})
    report = {
        "analysis_status": "post-hoc diagnostic on previously inspected test periods",
        "policy_selection": "frozen model weights, SKU order and validation-selected rule",
        "interpretation": "Scenario outcomes are descriptive, not independent confirmations or "
                          "confidence intervals for future stores or real business profit.",
        "results": results,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
