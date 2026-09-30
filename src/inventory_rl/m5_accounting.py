"""Post-hoc accounting audit of the published WI_1 and TX_3 M5 replays.

The audit loads the frozen, integrity-checked model bundles, reproduces every
test-day action, and reconciles revenue and costs to the published daily profit.
It does not train, tune, or select a policy.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from inventory_rl.context_artifact import load_context_model
from inventory_rl.m5 import M5Series, load_m5
from inventory_rl.policy_search_artifact import load_actor_model
from inventory_rl.portfolio import (
    PortfolioConfig,
    PortfolioEnv,
    allocate,
    base_stock_scores,
    context_policy_scores,
    policy_search_scores,
)


def _require_close(actual: float, expected: float, label: str) -> None:
    if not np.isclose(actual, expected, rtol=0, atol=1e-7):
        raise ValueError(f"{label} does not reconcile: {actual} vs {expected}")


def replay_accounting(data: M5Series, config: PortfolioConfig, start: int, end: int,
                      *, family: str, actor: bool, theta: np.ndarray,
                      baseline: dict) -> dict:
    """Reproduce a frozen policy and record mutually reconcilable unit accounts."""
    if family not in ("policy_search", "context"):
        raise ValueError("unknown policy family")
    expected_shape = (5,) if family == "policy_search" else (9,)
    if theta.shape != expected_shape or not np.isfinite(theta).all():
        raise ValueError("invalid frozen coefficients")
    env = PortfolioEnv(data, config)
    env.reset(start, end)
    units = {"demand_units": 0, "sold_units": 0, "lost_units": 0,
             "ordered_units": 0, "held_inventory_unit_days": 0}
    daily_profit = []
    while env.day < end:
        position = env.pipeline.sum(axis=0)
        if actor and family == "policy_search":
            scores = policy_search_scores(
                env.stock, position, env.last_sales, env.mean_train, env.lead,
                env.day, theta, cover=baseline["cover"], recent=baseline["recent_sales"]
            )
        elif actor:
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
        demand = data.sales[:, env.day]
        sold = np.minimum(env.stock, demand)
        lost = demand - sold
        held = env.stock - sold  # Holding is charged before today's arrivals.
        day_profit = (float(sold.sum()) * config.unit_price -
                      float(orders.sum()) * config.unit_cost -
                      float(held.sum()) * config.holding_cost -
                      float(lost.sum()) * config.lost_sale_penalty)
        units["demand_units"] += int(demand.sum())
        units["sold_units"] += int(sold.sum())
        units["lost_units"] += int(lost.sum())
        units["ordered_units"] += int(orders.sum())
        units["held_inventory_unit_days"] += int(held.sum())
        _, reward, _, info = env.step(orders)
        _require_close(day_profit, float(reward.sum()), "simulator daily profit")
        _require_close(day_profit, info["profit"], "simulator daily info")
        daily_profit.append(day_profit)
    amounts = {
        "sales_revenue": units["sold_units"] * config.unit_price,
        "procurement_cost": units["ordered_units"] * config.unit_cost,
        "holding_cost": units["held_inventory_unit_days"] * config.holding_cost,
        "lost_sale_penalty": units["lost_units"] * config.lost_sale_penalty,
    }
    raw_profit = (amounts["sales_revenue"] - amounts["procurement_cost"] -
                  amounts["holding_cost"] - amounts["lost_sale_penalty"])
    _require_close(raw_profit, sum(daily_profit), "accounted total profit")
    amounts = {key: round(float(value), 2) for key, value in amounts.items()}
    profit = round(float(raw_profit), 2)
    terminal = {
        "on_hand_units": int(env.stock.sum()),
        "pipeline_units": int(env.pipeline.sum()),
        "position_units": int(env.stock.sum() + env.pipeline.sum()),
    }
    return {
        "units": units,
        "amounts": amounts,
        "profit": profit,
        "daily_profit": daily_profit,
        "fill_rate": units["sold_units"] / max(units["demand_units"], 1),
        "mean_daily_spend": amounts["procurement_cost"] / (end - start),
        "terminal": terminal,
    }


def audit_report(data_path: Path, report_path: Path, model_directory: Path,
                 *, family: str) -> dict:
    """Verify source/model provenance, replay accounts, and published results."""
    if family not in ("policy_search", "context"):
        raise ValueError("unknown policy family")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    meta = report["dataset"]
    store = meta["store_id"]
    train_end = meta["train_days"][1]
    validation_end = meta["validation_days"][1]
    data = load_m5(data_path, store_id=store, sku_count=meta["sku_count"],
                   train_end=train_end, validation_end=validation_end)
    if (data.source_sha256 != meta["source_sha256"] or
            list(data.item_ids) != meta["selected_item_ids"] or
            meta["train_days"] != [1, train_end] or
            meta["validation_days"] != [train_end + 1, validation_end] or
            meta["test_days"] != [validation_end + 1, data.sales.shape[1]]):
        raise ValueError("published M5 source, SKU order, or split mismatch")
    if family == "policy_search":
        theta, manifest = load_actor_model(model_directory)
        actor_name, rule_name = "actor", "base_stock"
        baseline_keys = ("cover", "recent_sales")
    else:
        theta, manifest = load_context_model(model_directory)
        actor_name, rule_name = "actor", "strong_rule"
        baseline_keys = ("cover", "recent_sales", "beta")
    config = PortfolioConfig(**report["economics"])
    baseline = report["baseline_selection"]
    frozen_baseline = {key: baseline[key] for key in baseline_keys}
    if (manifest["store_id"] != store or
            manifest["source_sha256"] != data.source_sha256 or
            manifest["item_ids"] != list(data.item_ids) or
            manifest["economics"] != asdict(config) or
            manifest["baseline_selection"] != frozen_baseline or
            not np.array_equal(theta, np.asarray(report["model_selection"]["selected_theta"]))):
        raise ValueError("published model bundle and report disagree")
    start, end = validation_end, data.sales.shape[1]
    actor = replay_accounting(data, config, start, end, family=family, actor=True,
                              theta=theta, baseline=frozen_baseline)
    rule = replay_accounting(data, config, start, end, family=family, actor=False,
                             theta=theta, baseline=frozen_baseline)
    for name, replayed in ((actor_name, actor), (rule_name, rule)):
        published = report["test"][name]
        _require_close(replayed["profit"], published["profit"], f"{name} total profit")
        _require_close(replayed["fill_rate"], published["fill_rate"], f"{name} fill rate")
        _require_close(replayed["mean_daily_spend"], published["mean_daily_spend"],
                       f"{name} mean daily spend")
        if len(replayed["daily_profit"]) != len(published["daily_profit"]):
            raise ValueError(f"{name} daily profit length mismatch")
        for day, (actual, expected) in enumerate(zip(replayed["daily_profit"],
                                                     published["daily_profit"]),
                                                 start=start + 1):
            _require_close(actual, expected, f"{name} day {day} profit")
    difference = round(actor["profit"] - rule["profit"], 2)
    _require_close(difference, report["test"]["paired_profit_uplift"],
                   "published paired uplift")
    amount_delta = {key: round(actor["amounts"][key] - rule["amounts"][key], 2)
                    for key in actor["amounts"]}
    profit_effect = {
        "sales_revenue": amount_delta["sales_revenue"],
        "procurement_cost": -amount_delta["procurement_cost"],
        "holding_cost": -amount_delta["holding_cost"],
        "lost_sale_penalty": -amount_delta["lost_sale_penalty"],
    }
    _require_close(sum(profit_effect.values()), difference,
                   "actor-minus-rule profit bridge")
    unit_delta = {key: actor["units"][key] - rule["units"][key]
                  for key in actor["units"]}
    terminal_delta = {key: actor["terminal"][key] - rule["terminal"][key]
                      for key in actor["terminal"]}
    if terminal_delta["position_units"] != (unit_delta["ordered_units"] -
                                             unit_delta["sold_units"]):
        raise ValueError("terminal inventory conservation failed")
    for policy in (actor, rule):
        del policy["daily_profit"]  # Full daily paths remain in the original reports.
    return {
        "analysis_status": "post-hoc accounting diagnostic; no new independent test",
        "store_id": store,
        "family": family,
        "source_sha256": data.source_sha256,
        "model_sha256": manifest["model_sha256"],
        "test_days": meta["test_days"],
        "economics": asdict(config),
        "actor": actor,
        "rule": rule,
        "actor_minus_rule": {
            "units": unit_delta,
            "amounts": amount_delta,
            "profit_effect": profit_effect,
            "terminal": terminal_delta,
            "profit": difference,
        },
        "published_profit_difference": round(
            report["test"]["paired_profit_uplift"], 2),
        "reconciliation_error": round(
            difference - report["test"]["paired_profit_uplift"], 2),
    }


def run_published_audit(data_path: Path) -> dict:
    """Audit both frozen store results from the repository root."""
    stores = [
        audit_report(data_path, Path("docs/m5-wi1-policy-search-report.json"),
                     Path("models/wi1-policy-search"), family="policy_search"),
        audit_report(data_path, Path("docs/m5-tx3-context-report.json"),
                     Path("models/tx3-context"), family="context"),
    ]
    return {
        "analysis_status": "post-hoc accounting diagnostic; shared M5 calendar days",
        "stores": stores,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path,
                        help="Official M5 sales_train_validation.csv")
    parser.add_argument("--output", type=Path,
                        help="Write JSON audit here; otherwise print to stdout")
    args = parser.parse_args()
    audit = run_published_audit(args.data)
    rendered = json.dumps(audit, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
