"""Independent reconciliation checks for frozen-policy accounting."""

import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from inventory_rl.context_artifact import save_context_model
from inventory_rl.m5 import load_m5
from inventory_rl.m5_accounting import audit_report, replay_accounting
from inventory_rl.m5_context_actor import replay_context
from inventory_rl.m5_experiment import replay
from inventory_rl.m5_policy_search import replay_actor
from inventory_rl.policy_search_artifact import save_actor_model
from inventory_rl.portfolio import PortfolioConfig


def _write_fixture(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "item_id", "store_id"] +
                        [f"d_{day}" for day in range(1, 161)])
        for item in range(4):
            sales = [2 + (day + item * 3) % (5 + item) for day in range(1, 161)]
            writer.writerow([f"item_{item}_S", f"item_{item}", "S"] + sales)


@pytest.mark.parametrize("family", ["policy_search", "context"])
def test_published_accounting_reconciles_each_day_and_rejects_tampering(family: str) -> None:
    data_path = Path("artifacts/_test_accounting.csv")
    _write_fixture(data_path)
    data = load_m5(data_path, store_id="S", sku_count=4,
                   train_end=120, validation_end=140)
    config = PortfolioConfig(budget_per_sku=24.0, capacity_per_sku=45)
    start, end = data.validation_end, data.sales.shape[1]
    if family == "policy_search":
        baseline = {"cover": 2.0, "recent_sales": True}
        theta = np.array([0.2, -0.1, 0.3, 0.1, -0.2])
        actor = replay_actor(data, config, start, end, theta,
                             cover=baseline["cover"], recent=baseline["recent_sales"])
        rule = replay(data, config, start, end, cover=baseline["cover"],
                      recent=baseline["recent_sales"])
        rule_name = "base_stock"
        directory = Path("artifacts/_test_accounting_actor")
        save_actor_model(theta, data, config, directory, True,
                         baseline_cover=baseline["cover"],
                         baseline_recent=baseline["recent_sales"])
    else:
        baseline = {"cover": 2.0, "recent_sales": True, "beta": 0.5}
        theta = np.array([0.2, -0.1, 0.3, 0.1, -0.2, 0.15, -0.1, 0.2, 0.05])
        actor = replay_context(data, config, start, end, theta, cover=baseline["cover"],
                               recent=baseline["recent_sales"], beta=baseline["beta"])
        rule = replay(data, config, start, end, cover=baseline["cover"],
                      recent=baseline["recent_sales"], baseline_beta=baseline["beta"])
        rule_name = "strong_rule"
        directory = Path("artifacts/_test_accounting_context")
        save_context_model(theta, data, config, directory, True,
                           baseline_cover=baseline["cover"],
                           baseline_recent=baseline["recent_sales"],
                           baseline_beta=baseline["beta"])
    report = {
        "dataset": {"source_sha256": data.source_sha256, "store_id": "S",
                    "sku_count": 4, "train_days": [1, 120],
                    "validation_days": [121, 140], "test_days": [141, 160],
                    "selected_item_ids": list(data.item_ids)},
        "economics": asdict(config),
        "baseline_selection": baseline,
        "model_selection": {"selected_theta": theta.tolist()},
        "test": {"actor": actor, rule_name: rule,
                 "paired_profit_uplift": actor["profit"] - rule["profit"]},
    }
    report_path = Path(f"artifacts/_test_accounting_{family}_report.json")
    report_path.write_text(json.dumps(report), encoding="utf-8")
    result = audit_report(data_path, report_path, directory, family=family)
    assert result["reconciliation_error"] == pytest.approx(0)
    assert result["actor_minus_rule"]["profit"] == pytest.approx(
        actor["profit"] - rule["profit"])
    assert sum(result["actor_minus_rule"]["profit_effect"].values()) == pytest.approx(
        actor["profit"] - rule["profit"])
    units = result["actor_minus_rule"]["units"]
    assert result["actor_minus_rule"]["terminal"]["position_units"] == (
        units["ordered_units"] - units["sold_units"])
    for policy in (result["actor"], result["rule"]):
        amounts = policy["amounts"]
        assert policy["profit"] == pytest.approx(
            amounts["sales_revenue"] - amounts["procurement_cost"] -
            amounts["holding_cost"] - amounts["lost_sale_penalty"])
        assert policy["units"]["demand_units"] == (
            policy["units"]["sold_units"] + policy["units"]["lost_units"])
    report["test"]["actor"]["daily_profit"][0] += 1
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="actor day 141 profit"):
        audit_report(data_path, report_path, directory, family=family)


def test_accounting_uses_same_actions_as_both_original_replays() -> None:
    data_path = Path("artifacts/_test_accounting.csv")
    _write_fixture(data_path)
    data = load_m5(data_path, store_id="S", sku_count=4,
                   train_end=120, validation_end=140)
    config = PortfolioConfig(budget_per_sku=24.0, capacity_per_sku=45)
    for family, theta, beta in (
        ("policy_search", np.array([0.2, -0.1, 0.3, 0.1, -0.2]), None),
        ("context", np.array([0.2, -0.1, 0.3, 0.1, -0.2, 0.15, -0.1, 0.2, 0.05]), 0.5),
    ):
        baseline = {"cover": 2.0, "recent_sales": True}
        if beta is not None:
            baseline["beta"] = beta
        accounted = replay_accounting(data, config, 140, 160, family=family,
                                      actor=True, theta=theta, baseline=baseline)
        if family == "policy_search":
            original = replay_actor(data, config, 140, 160, theta, cover=2.0, recent=True)
        else:
            original = replay_context(data, config, 140, 160, theta,
                                      cover=2.0, recent=True, beta=0.5)
        assert accounted["daily_profit"] == pytest.approx(original["daily_profit"])
