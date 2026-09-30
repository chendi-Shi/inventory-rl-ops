"""Predeclared residual-RL evaluation on a fresh M5 store."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from inventory_rl.m5 import load_m5
from inventory_rl.m5_experiment import block_bootstrap_ci, replay, train_one
from inventory_rl.portfolio import PortfolioConfig
from inventory_rl.portfolio_artifact import save_portfolio_model

DEFAULT_ALPHAS = (0.0, 0.25, 0.5, 1.0, 2.0)


def run_hybrid(path: Path, output: Path, *, store_id: str = "CA_4", sku_count: int = 64,
               seeds: tuple[int, ...] = (11, 22, 33), episodes: int = 60,
               alphas: tuple[float, ...] = DEFAULT_ALPHAS,
               train_end: int = 1700, validation_end: int = 1800) -> dict:
    if not seeds:
        raise ValueError("at least one seed is required")
    if not alphas or 0.0 not in alphas or len(alphas) != len(set(alphas)) or (
        not np.isfinite(alphas).all()
    ) or min(alphas) < 0:
        raise ValueError("alphas must be unique, nonnegative, finite and include zero")
    data = load_m5(path, store_id=store_id, sku_count=sku_count,
                   train_end=train_end, validation_end=validation_end)
    config = PortfolioConfig()
    baseline_candidates = []
    for recent in (False, True):
        for cover in (0.5, 1.0, 1.5, 2.0, 3.0):
            validation = replay(data, config, data.train_end, data.validation_end,
                                cover=cover, recent=recent)
            baseline_candidates.append((validation["profit"], recent, cover, validation))
    _, baseline_recent, baseline_cover, baseline_validation = max(baseline_candidates)

    candidates = []
    for seed in seeds:
        agent, training = train_one(data, config, seed, episodes)
        for alpha in alphas:
            validation = replay(data, config, data.train_end, data.validation_end, agent,
                                cover=baseline_cover, recent=baseline_recent,
                                hybrid_alpha=alpha)
            candidates.append({"seed": seed, "alpha": alpha, "agent": agent,
                               "training": training, "validation": validation})
    selected = max(candidates, key=lambda candidate: (
        candidate["validation"]["profit"], -candidate["alpha"], -candidate["seed"]
    ))
    # Store, split, alpha grid and selection rule were fixed before this test period was read.
    start, end = data.validation_end, data.sales.shape[1]
    test_candidate = replay(data, config, start, end, selected["agent"],
                            cover=baseline_cover, recent=baseline_recent,
                            hybrid_alpha=selected["alpha"])
    test_baseline = replay(data, config, start, end, cover=baseline_cover,
                           recent=baseline_recent)
    test_random = replay(data, config, start, end, random_seed=2026)
    ci = block_bootstrap_ci(test_candidate["daily_profit"], test_baseline["daily_profit"])
    p10_candidate = float(np.quantile(test_candidate["daily_profit"], 0.1))
    p10_base = float(np.quantile(test_baseline["daily_profit"], 0.1))
    eligible = (selected["alpha"] > 0 and ci[0] > 0 and
                test_candidate["fill_rate"] >= test_baseline["fill_rate"] - 0.02 and
                p10_candidate >= p10_base)
    report = {
        "dataset": {"name": "M5 sales_train_validation", "source_sha256": data.source_sha256,
                    "store_id": store_id, "sku_count": data.n_sku,
                    "train_days": [1, data.train_end],
                    "validation_days": [data.train_end + 1, data.validation_end],
                    "test_days": [data.validation_end + 1, end],
                    "selected_item_ids": list(data.item_ids)},
        "economics": asdict(config),
        "model_selection": {
            "policy_family": "base-stock plus scaled Double DQN residual",
            "seeds": list(seeds), "alphas": list(alphas), "candidate_count": len(candidates),
            "selected_seed": selected["seed"], "selected_alpha": selected["alpha"],
            "selected_training": selected["training"],
            "selected_validation": selected["validation"],
            "validation_candidates": [
                {"seed": candidate["seed"], "alpha": candidate["alpha"],
                 "profit": candidate["validation"]["profit"]} for candidate in candidates
            ],
        },
        "baseline_selection": {"recent_sales": baseline_recent, "cover": baseline_cover,
                               "validation": baseline_validation},
        "test": {"candidate": test_candidate, "base_stock": test_baseline,
                 "random": test_random,
                 "paired_profit_uplift": test_candidate["profit"] - test_baseline["profit"],
                 "paired_uplift_block_bootstrap_ci95": ci,
                 "p10_daily_profit_candidate": p10_candidate,
                 "p10_daily_profit_base_stock": p10_base,
                 "promotion_eligible": bool(eligible),
                 "active_policy": "hybrid_rl" if eligible else "base_stock"},
        "caveat": "Observed sales are a censored demand proxy; economics and lead times are simulated.",
    }
    output.mkdir(parents=True, exist_ok=True)
    save_portfolio_model(selected["agent"], data, config, output, bool(eligible),
                         baseline_recent=baseline_recent, baseline_cover=baseline_cover,
                         hybrid_alpha=selected["alpha"])
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
