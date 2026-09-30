"""Frozen-actor transfer check on previously untouched M5 stores."""

import json
from pathlib import Path

import numpy as np

from inventory_rl.m5 import load_m5
from inventory_rl.m5_experiment import block_bootstrap_ci, replay
from inventory_rl.m5_policy_search import replay_actor
from inventory_rl.policy_search_artifact import load_actor_model
from inventory_rl.portfolio import PortfolioConfig


def run_transfer(path: Path, actor_dir: Path, output: Path, *,
                 stores: tuple[str, ...] = ("WI_2", "WI_3"),
                 sku_count: int = 64, train_end: int = 1700,
                 validation_end: int = 1800) -> dict:
    if not stores or len(set(stores)) != len(stores):
        raise ValueError("stores must be nonempty and unique")
    theta, actor_manifest = load_actor_model(actor_dir)
    config = PortfolioConfig(**actor_manifest["economics"])
    results = {}
    for store_id in stores:
        data = load_m5(path, store_id=store_id, sku_count=sku_count,
                       train_end=train_end, validation_end=validation_end)
        if data.source_sha256 != actor_manifest["source_sha256"]:
            raise ValueError("transfer dataset differs from actor training source")
        candidates = []
        for recent in (False, True):
            for cover in (0.5, 1.0, 1.5, 2.0, 3.0):
                validation = replay(data, config, data.train_end, data.validation_end,
                                    cover=cover, recent=recent)
                candidates.append((validation["profit"], recent, cover, validation))
        _, recent, cover, baseline_validation = max(candidates)
        start, end = data.validation_end, data.sales.shape[1]
        actor = replay_actor(data, config, start, end, theta, cover=cover, recent=recent)
        baseline = replay(data, config, start, end, cover=cover, recent=recent)
        ci = block_bootstrap_ci(actor["daily_profit"], baseline["daily_profit"])
        p10_actor = float(np.quantile(actor["daily_profit"], 0.1))
        p10_base = float(np.quantile(baseline["daily_profit"], 0.1))
        passes = (ci[0] > 0 and actor["fill_rate"] >= baseline["fill_rate"] - 0.02 and
                  p10_actor >= p10_base)
        results[store_id] = {
            "dataset": {"store_id": store_id, "sku_count": data.n_sku,
                        "source_sha256": data.source_sha256,
                        "selected_item_ids": list(data.item_ids),
                        "train_days": [1, data.train_end],
                        "validation_days": [data.train_end + 1, data.validation_end],
                        "test_days": [data.validation_end + 1, end]},
            "baseline_selection": {"recent_sales": recent, "cover": cover,
                                   "validation": baseline_validation},
            "test": {"actor": actor, "base_stock": baseline,
                     "paired_profit_uplift": actor["profit"] - baseline["profit"],
                     "paired_uplift_block_bootstrap_ci95": ci,
                     "p10_daily_profit_actor": p10_actor,
                     "p10_daily_profit_base_stock": p10_base,
                     "passes_transfer_gate": bool(passes)},
        }
    report = {"frozen_actor": {"source_store": actor_manifest["store_id"],
                               "model_sha256": actor_manifest["model_sha256"],
                               "theta": theta.tolist()},
              "stores": results,
              "broad_transfer_supported": all(r["test"]["passes_transfer_gate"]
                                              for r in results.values()),
              "caveat": "Sales are a censored demand proxy; economics are simulated."}
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
