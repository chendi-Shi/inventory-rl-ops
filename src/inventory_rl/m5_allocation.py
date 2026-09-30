"""Predeclared test of forecast-rate normalization in constrained allocation."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from inventory_rl.m5 import load_m5
from inventory_rl.m5_experiment import block_bootstrap_ci, replay
from inventory_rl.portfolio import PortfolioConfig


def run_allocation(path: Path, output: Path, *, store_id: str = "TX_2",
                   sku_count: int = 64, train_end: int = 1700,
                   validation_end: int = 1800) -> dict:
    data = load_m5(path, store_id=store_id, sku_count=sku_count,
                   train_end=train_end, validation_end=validation_end)
    config = PortfolioConfig()
    candidates = []
    for beta in (0.0, 0.5, 1.0):
        for recent in (False, True):
            for cover in (0.5, 1.0, 1.5, 2.0, 3.0):
                validation = replay(data, config, data.train_end, data.validation_end,
                                    cover=cover, recent=recent, baseline_beta=beta)
                candidates.append({"beta": beta, "recent_sales": recent,
                                   "cover": cover, "validation_profit": validation["profit"]})

    def rank(candidate: dict) -> tuple[float, float, float, int]:
        return (candidate["validation_profit"], candidate["beta"],
                -candidate["cover"], -int(candidate["recent_sales"]))

    selected = max(candidates, key=rank)
    old_rule = max((c for c in candidates if c["beta"] == 1.0), key=rank)
    start, end = data.validation_end, data.sales.shape[1]

    def test(candidate: dict) -> dict:
        return replay(data, config, start, end, cover=candidate["cover"],
                      recent=candidate["recent_sales"], baseline_beta=candidate["beta"])

    new_test = test(selected)
    old_test = test(old_rule)
    ci = block_bootstrap_ci(new_test["daily_profit"], old_test["daily_profit"])
    new_p10 = float(np.quantile(new_test["daily_profit"], 0.1))
    old_p10 = float(np.quantile(old_test["daily_profit"], 0.1))
    eligible = (selected["beta"] < 1 and ci[0] > 0 and
                new_test["fill_rate"] >= old_test["fill_rate"] - 0.02 and
                new_p10 >= old_p10)
    report = {
        "dataset": {"name": "M5 sales_train_validation", "source_sha256": data.source_sha256,
                    "store_id": store_id, "sku_count": data.n_sku,
                    "train_days": [1, data.train_end],
                    "validation_days": [data.train_end + 1, data.validation_end],
                    "test_days": [data.validation_end + 1, end],
                    "selected_item_ids": list(data.item_ids)},
        "economics": asdict(config),
        "model_selection": {"candidate_count": len(candidates), "candidates": candidates,
                            "expanded_rule": selected, "old_rule": old_rule},
        "test": {"expanded_rule": new_test, "old_rule": old_test,
                 "paired_profit_uplift": new_test["profit"] - old_test["profit"],
                 "paired_uplift_block_bootstrap_ci95": ci,
                 "p10_daily_profit_expanded_rule": new_p10,
                 "p10_daily_profit_old_rule": old_p10,
                 "adopt_expanded_rule": bool(eligible),
                 "active_rule": "expanded" if eligible else "old"},
        "caveat": "Observed sales are a censored demand proxy; economics and lead times are simulated.",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
