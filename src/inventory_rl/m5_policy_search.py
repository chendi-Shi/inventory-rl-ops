"""Episodic policy search against constrained portfolio return."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from inventory_rl.m5 import M5Series, load_m5
from inventory_rl.m5_experiment import block_bootstrap_ci, replay
from inventory_rl.policy_search_artifact import save_actor_model
from inventory_rl.portfolio import (
    PortfolioConfig,
    PortfolioEnv,
    allocate,
    policy_search_scores,
)

N_FEATURES = 5


def actor_scores(env: PortfolioEnv, theta: np.ndarray, *, cover: float,
                 recent: bool) -> np.ndarray:
    return policy_search_scores(env.stock, env.pipeline.sum(axis=0), env.last_sales,
                                env.mean_train, env.lead, env.day, theta,
                                cover=cover, recent=recent)


def replay_actor(data: M5Series, config: PortfolioConfig, start: int, end: int,
                 theta: np.ndarray, *, cover: float, recent: bool) -> dict:
    env = PortfolioEnv(data, config)
    env.reset(start, end)
    daily = []
    demand = sold = spend = 0.0
    while env.day < end:
        scores = actor_scores(env, theta, cover=cover, recent=recent)
        orders = allocate(scores, stock=env.stock, pipeline=env.pipeline.sum(axis=0),
                          budget=env.budget, capacity=env.capacity,
                          unit_cost=config.unit_cost)
        _, reward, _, info = env.step(orders)
        daily.append(float(reward.sum()))
        demand += info["demand"]
        sold += info["sold"]
        spend += info["spend"]
    return {"profit": float(sum(daily)), "daily_profit": daily,
            "fill_rate": float(sold / max(demand, 1)),
            "mean_daily_spend": float(spend / (end - start)),
            "days": end - start, "sku_count": data.n_sku}


def run_policy_search(path: Path, output: Path, *, store_id: str = "WI_1",
                      sku_count: int = 64, train_end: int = 1700,
                      validation_end: int = 1800, seeds: tuple[int, ...] = (11, 22),
                      iterations: int = 10, population: int = 8,
                      window: int = 84) -> dict:
    if not seeds or iterations < 1 or population < 2 or window < 7:
        raise ValueError("invalid policy-search settings")
    data = load_m5(path, store_id=store_id, sku_count=sku_count,
                   train_end=train_end, validation_end=validation_end)
    if data.train_end - window < 365:
        raise ValueError("insufficient training history")
    config = PortfolioConfig()
    baseline_candidates = []
    for recent in (False, True):
        for cover in (0.5, 1.0, 1.5, 2.0, 3.0):
            validation = replay(data, config, data.train_end, data.validation_end,
                                cover=cover, recent=recent)
            baseline_candidates.append((validation["profit"], recent, cover, validation))
    _, recent, cover, baseline_validation = max(baseline_candidates)

    checkpoints = []
    histories = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        windows = rng.choice(np.arange(365, data.train_end - window + 1),
                             size=2, replace=False)
        mean = np.zeros(N_FEATURES)
        sigma = 0.5
        history = {"seed": seed, "train_windows_zero_based": windows.tolist(),
                   "iterations": []}
        for iteration in range(iterations + 1):
            validation = replay_actor(data, config, data.train_end, data.validation_end,
                                      mean, cover=cover, recent=recent)
            checkpoints.append({"seed": seed, "iteration": iteration,
                                "theta": mean.copy(),
                                "validation_profit": validation["profit"]})
            if iteration == iterations:
                break
            population_theta = mean + rng.normal(0, sigma, size=(population, N_FEATURES))
            train_returns = []
            for theta in population_theta:
                train_returns.append(sum(replay_actor(data, config, int(start),
                                                      int(start) + window, theta,
                                                      cover=cover, recent=recent)["profit"]
                                         for start in windows))
            elite_indices = np.argsort(train_returns)[-2:]
            mean = population_theta[elite_indices].mean(axis=0)
            history["iterations"].append({
                "iteration": iteration + 1, "sigma": sigma,
                "train_returns": [float(x) for x in train_returns],
                "elite_indices": elite_indices.tolist(),
                "mean_theta": mean.tolist(),
            })
            sigma = max(0.05, sigma * 0.85)
        histories.append(history)
    selected = max(checkpoints, key=lambda c: (c["validation_profit"],
                                                -np.linalg.norm(c["theta"]), -c["seed"]))
    start, end = data.validation_end, data.sales.shape[1]
    actor_test = replay_actor(data, config, start, end, selected["theta"],
                              cover=cover, recent=recent)
    baseline_test = replay(data, config, start, end, cover=cover, recent=recent)
    ci = block_bootstrap_ci(actor_test["daily_profit"], baseline_test["daily_profit"])
    p10_actor = float(np.quantile(actor_test["daily_profit"], 0.1))
    p10_base = float(np.quantile(baseline_test["daily_profit"], 0.1))
    eligible = (np.linalg.norm(selected["theta"]) > 0 and ci[0] > 0 and
                actor_test["fill_rate"] >= baseline_test["fill_rate"] - 0.02 and
                p10_actor >= p10_base)
    report = {
        "dataset": {"name": "M5 sales_train_validation", "source_sha256": data.source_sha256,
                    "store_id": store_id, "sku_count": data.n_sku,
                    "train_days": [1, data.train_end],
                    "validation_days": [data.train_end + 1, data.validation_end],
                    "test_days": [data.validation_end + 1, end],
                    "selected_item_ids": list(data.item_ids)},
        "economics": asdict(config),
        "baseline_selection": {"recent_sales": recent, "cover": cover,
                               "validation": baseline_validation},
        "model_selection": {
            "algorithm": "cross-entropy episodic portfolio policy search",
            "seeds": list(seeds), "iterations": iterations, "population": population,
            "window": window, "selected_seed": selected["seed"],
            "selected_iteration": selected["iteration"],
            "selected_theta": selected["theta"].tolist(),
            "selected_validation_profit": selected["validation_profit"],
            "validation_checkpoints": [
                {"seed": c["seed"], "iteration": c["iteration"],
                 "theta": c["theta"].tolist(), "profit": c["validation_profit"]}
                for c in checkpoints
            ],
            "training_histories": histories,
        },
        "test": {"actor": actor_test, "base_stock": baseline_test,
                 "paired_profit_uplift": actor_test["profit"] - baseline_test["profit"],
                 "paired_uplift_block_bootstrap_ci95": ci,
                 "p10_daily_profit_actor": p10_actor,
                 "p10_daily_profit_base_stock": p10_base,
                 "promotion_eligible": bool(eligible),
                 "active_policy": "policy_search_rl" if eligible else "base_stock"},
        "caveat": "Observed sales are a censored demand proxy; economics and lead times are simulated.",
    }
    output.mkdir(parents=True, exist_ok=True)
    save_actor_model(selected["theta"], data, config, output, bool(eligible),
                     baseline_cover=cover, baseline_recent=recent)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
