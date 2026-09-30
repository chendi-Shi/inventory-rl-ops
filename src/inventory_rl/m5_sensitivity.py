"""Explicitly post-hoc stronger-rule sensitivity on the inspected WI_1 test."""

import json
from pathlib import Path

import numpy as np

from inventory_rl.m5 import load_m5
from inventory_rl.m5_experiment import block_bootstrap_ci, replay
from inventory_rl.m5_policy_search import replay_actor
from inventory_rl.policy_search_artifact import load_actor_model
from inventory_rl.portfolio import PortfolioConfig


def run_sensitivity(path: Path, actor_dir: Path, output: Path, *,
                    sku_count: int = 64, train_end: int = 1700,
                    validation_end: int = 1800) -> dict:
    theta, manifest = load_actor_model(actor_dir)
    data = load_m5(path, store_id=manifest["store_id"], sku_count=sku_count,
                   train_end=train_end, validation_end=validation_end)
    if data.source_sha256 != manifest["source_sha256"]:
        raise ValueError("sensitivity data differs from actor training source")
    config = PortfolioConfig(**manifest["economics"])
    candidates = []
    for beta in (0.0, 0.5, 1.0):
        for recent in (False, True):
            for cover in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0):
                validation = replay(data, config, data.train_end, data.validation_end,
                                    cover=cover, recent=recent, baseline_beta=beta)
                candidates.append({"beta": beta, "recent_sales": recent,
                                   "cover": cover, "validation_profit": validation["profit"]})
    selected = max(candidates, key=lambda c: (c["validation_profit"], c["beta"],
                                              -c["cover"], -int(c["recent_sales"])))
    start, end = data.validation_end, data.sales.shape[1]
    baseline = replay(data, config, start, end, cover=selected["cover"],
                      recent=selected["recent_sales"], baseline_beta=selected["beta"])
    original = manifest["baseline_selection"]
    actor = replay_actor(data, config, start, end, theta,
                         cover=original["cover"], recent=original["recent_sales"])
    ci = block_bootstrap_ci(actor["daily_profit"], baseline["daily_profit"])
    report = {
        "analysis_status": "post-hoc sensitivity after WI_1 test was inspected",
        "source_sha256": data.source_sha256,
        "actor_model_sha256": manifest["model_sha256"],
        "candidate_count": len(candidates), "validation_candidates": candidates,
        "selected_stronger_rule": selected,
        "test": {"actor": actor, "stronger_rule": baseline,
                 "paired_profit_uplift": actor["profit"] - baseline["profit"],
                 "paired_uplift_block_bootstrap_ci95": ci,
                 "p10_daily_profit_actor": float(np.quantile(actor["daily_profit"], 0.1)),
                 "p10_daily_profit_rule": float(np.quantile(baseline["daily_profit"], 0.1))},
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
