"""Predeclared heuristic-guided exploration ablation on an untouched M5 store."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from inventory_rl.m5 import load_m5
from inventory_rl.m5_experiment import block_bootstrap_ci, replay, train_one
from inventory_rl.m5_hybrid import DEFAULT_ALPHAS
from inventory_rl.portfolio import PortfolioConfig
from inventory_rl.portfolio_artifact import save_portfolio_model


def run_guided(path: Path, output: Path, *, store_id: str = "TX_1", sku_count: int = 64,
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
    for guided in (False, True):
        for seed in seeds:
            agent, training = train_one(data, config, seed, episodes,
                                        guidance_start=0.5 if guided else 0.0)
            for alpha in alphas:
                validation = replay(data, config, data.train_end, data.validation_end, agent,
                                    cover=baseline_cover, recent=baseline_recent,
                                    hybrid_alpha=alpha)
                candidates.append({"guided": guided, "seed": seed, "alpha": alpha,
                                   "agent": agent, "training": training,
                                   "validation": validation})

    def rank(candidate: dict) -> tuple[float, float, int, int]:
        return (candidate["validation"]["profit"], -candidate["alpha"],
                -int(candidate["guided"]), -candidate["seed"])

    selected = max(candidates, key=rank)
    by_family = {name: max((c for c in candidates if c["guided"] == guided), key=rank)
                 for name, guided in (("unguided", False), ("guided", True))}
    start, end = data.validation_end, data.sales.shape[1]

    def test_candidate(candidate: dict) -> dict:
        result = replay(data, config, start, end, candidate["agent"],
                        cover=baseline_cover, recent=baseline_recent,
                        hybrid_alpha=candidate["alpha"])
        ci = block_bootstrap_ci(result["daily_profit"], baseline_test["daily_profit"])
        return {"guided": candidate["guided"], "seed": candidate["seed"],
                "alpha": candidate["alpha"], "result": result,
                "paired_profit_uplift": result["profit"] - baseline_test["profit"],
                "paired_uplift_block_bootstrap_ci95": ci,
                "p10_daily_profit": float(np.quantile(result["daily_profit"], 0.1))}

    baseline_test = replay(data, config, start, end, cover=baseline_cover,
                           recent=baseline_recent)
    family_tests = {name: test_candidate(candidate) for name, candidate in by_family.items()}
    selected_test = family_tests["guided" if selected["guided"] else "unguided"]
    p10_base = float(np.quantile(baseline_test["daily_profit"], 0.1))
    eligible = (selected["alpha"] > 0 and
                selected_test["paired_uplift_block_bootstrap_ci95"][0] > 0 and
                selected_test["result"]["fill_rate"] >= baseline_test["fill_rate"] - 0.02 and
                selected_test["p10_daily_profit"] >= p10_base)
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
            "guidance_start": 0.5, "guidance_rule": {"cover": 2.0, "recent": False},
            "seeds": list(seeds), "alphas": list(alphas),
            "candidate_count": len(candidates),
            "selected_trainer": "guided" if selected["guided"] else "unguided",
            "selected_seed": selected["seed"], "selected_alpha": selected["alpha"],
            "selected_training": selected["training"],
            "validation_candidates": [
                {"trainer": "guided" if c["guided"] else "unguided",
                 "seed": c["seed"], "alpha": c["alpha"],
                 "profit": c["validation"]["profit"]} for c in candidates
            ],
        },
        "baseline_selection": {"recent_sales": baseline_recent, "cover": baseline_cover,
                               "validation": baseline_validation},
        "test": {"selected": selected_test, "by_trainer": family_tests,
                 "base_stock": baseline_test,
                 "random": replay(data, config, start, end, random_seed=2026),
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
