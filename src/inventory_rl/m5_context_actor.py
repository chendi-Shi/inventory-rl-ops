"""Context-aware marginal-score actor for binding shared purchasing budgets."""

import json
from pathlib import Path

import numpy as np

from inventory_rl.context_artifact import save_context_model
from inventory_rl.m5 import M5Series, load_m5
from inventory_rl.m5_experiment import block_bootstrap_ci, replay
from inventory_rl.portfolio import PortfolioConfig, PortfolioEnv, allocate, context_policy_scores

N_CONTEXT_FEATURES = 9
COVER_GRID = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0)
BETA_GRID = (0.0, 0.5, 1.0)


def context_scores(env: PortfolioEnv, theta: np.ndarray, *, cover: float,
                   recent: bool, beta: float) -> np.ndarray:
    """Add a bounded SKU priority to the validation-selected rule scores."""
    return context_policy_scores(
        env.stock, env.pipeline.sum(axis=0), env.last_sales,
        env.mean_train, env.lead, env.day, env.budget,
        env.config.unit_cost, theta, cover=cover, recent=recent, beta=beta
    )


def replay_context(data: M5Series, config: PortfolioConfig, start: int, end: int,
                   theta: np.ndarray, *, cover: float, recent: bool,
                   beta: float) -> dict:
    env = PortfolioEnv(data, config)
    env.reset(start, end)
    daily = []
    demand = sold = spend = 0.0
    while env.day < end:
        scores = context_scores(env, theta, cover=cover, recent=recent, beta=beta)
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


def select_strong_rule(data: M5Series, config: PortfolioConfig) -> tuple[dict, list[dict]]:
    candidates = []
    for beta in BETA_GRID:
        for recent in (False, True):
            for cover in COVER_GRID:
                validation = replay(data, config, data.train_end, data.validation_end,
                                    cover=cover, recent=recent, baseline_beta=beta)
                candidates.append({"beta": beta, "recent_sales": recent,
                                   "cover": cover, "validation_profit": validation["profit"]})
    selected = max(candidates, key=lambda c: (c["validation_profit"], c["beta"],
                                              -c["cover"], -int(c["recent_sales"])))
    return selected, candidates


def fit_context(data: M5Series, config: PortfolioConfig, baseline: dict, *,
                seeds: tuple[int, ...] = (11, 22), iterations: int = 10,
                population: int = 8, window: int = 84) -> dict:
    if not seeds or iterations < 1 or population < 2 or window < 7:
        raise ValueError("invalid context search settings")
    if data.train_end - window < 365:
        raise ValueError("insufficient training history")
    cover = baseline["cover"]
    recent = baseline["recent_sales"]
    beta = baseline["beta"]
    checkpoints = []
    histories = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        windows = rng.choice(np.arange(365, data.train_end - window + 1),
                             size=2, replace=False)
        mean = np.zeros(N_CONTEXT_FEATURES)
        sigma = 0.5
        history = {"seed": seed, "train_windows_zero_based": windows.tolist(),
                   "iterations": []}
        for iteration in range(iterations + 1):
            validation = replay_context(data, config, data.train_end, data.validation_end,
                                        mean, cover=cover, recent=recent, beta=beta)
            checkpoints.append({"seed": seed, "iteration": iteration,
                                "theta": mean.copy(),
                                "validation_profit": validation["profit"]})
            if iteration == iterations:
                break
            population_theta = mean + rng.normal(0, sigma,
                                                 size=(population, N_CONTEXT_FEATURES))
            train_returns = []
            for theta in population_theta:
                train_returns.append(sum(replay_context(data, config, int(start),
                                                        int(start) + window, theta,
                                                        cover=cover, recent=recent,
                                                        beta=beta)["profit"]
                                         for start in windows))
            elite_indices = np.argsort(train_returns)[-2:]
            mean = population_theta[elite_indices].mean(axis=0)
            history["iterations"].append({
                "iteration": iteration + 1, "sigma": sigma,
                "train_returns": [float(x) for x in train_returns],
                "elite_indices": elite_indices.tolist(), "mean_theta": mean.tolist(),
            })
            sigma = max(0.05, sigma * 0.85)
        histories.append(history)
    selected = max(checkpoints, key=lambda c: (c["validation_profit"],
                                                -np.linalg.norm(c["theta"]), -c["seed"]))
    return {"selected": selected, "checkpoints": checkpoints,
            "training_histories": histories}


def run_context_development(path: Path, output: Path, *, store_id: str = "WI_2",
                            sku_count: int = 64, train_end: int = 1700,
                            validation_end: int = 1800, **fit_kwargs) -> dict:
    """Train and report validation only; this function never evaluates test days."""
    data = load_m5(path, store_id=store_id, sku_count=sku_count,
                   train_end=train_end, validation_end=validation_end)
    config = PortfolioConfig()
    baseline, baseline_candidates = select_strong_rule(data, config)
    fitted = fit_context(data, config, baseline, **fit_kwargs)
    selected = fitted["selected"]
    report = {
        "analysis_status": "development train and validation only",
        "store_id": store_id, "source_sha256": data.source_sha256,
        "selected_item_ids": list(data.item_ids),
        "baseline": baseline, "baseline_candidate_count": len(baseline_candidates),
        "selected_actor": {"seed": selected["seed"],
                           "iteration": selected["iteration"],
                           "theta": selected["theta"].tolist(),
                           "validation_profit": selected["validation_profit"]},
        "validation_checkpoints": [
            {"seed": c["seed"], "iteration": c["iteration"],
             "theta": c["theta"].tolist(), "profit": c["validation_profit"]}
            for c in fitted["checkpoints"]
        ],
        "training_histories": fitted["training_histories"],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def run_context_final(path: Path, output: Path, *, store_id: str = "TX_3",
                      sku_count: int = 64, train_end: int = 1700,
                      validation_end: int = 1800,
                      seeds: tuple[int, ...] = (11, 22), iterations: int = 10,
                      population: int = 8, window: int = 84) -> dict:
    """Apply the frozen training and selection protocol once on a fresh store."""
    data = load_m5(path, store_id=store_id, sku_count=sku_count,
                   train_end=train_end, validation_end=validation_end)
    config = PortfolioConfig()
    baseline, baseline_candidates = select_strong_rule(data, config)
    fitted = fit_context(data, config, baseline, seeds=seeds,
                         iterations=iterations, population=population, window=window)
    selected = fitted["selected"]
    start, end = data.validation_end, data.sales.shape[1]
    actor = replay_context(data, config, start, end, selected["theta"],
                           cover=baseline["cover"], recent=baseline["recent_sales"],
                           beta=baseline["beta"])
    rule = replay(data, config, start, end, cover=baseline["cover"],
                  recent=baseline["recent_sales"], baseline_beta=baseline["beta"])
    ci = block_bootstrap_ci(actor["daily_profit"], rule["daily_profit"])
    p10_actor = float(np.quantile(actor["daily_profit"], 0.1))
    p10_rule = float(np.quantile(rule["daily_profit"], 0.1))
    eligible = (np.linalg.norm(selected["theta"]) > 0 and ci[0] > 0 and
                actor["fill_rate"] >= rule["fill_rate"] - 0.02 and
                p10_actor >= p10_rule)
    report = {
        "dataset": {"name": "M5 sales_train_validation", "source_sha256": data.source_sha256,
                    "store_id": store_id, "sku_count": data.n_sku,
                    "train_days": [1, data.train_end],
                    "validation_days": [data.train_end + 1, data.validation_end],
                    "test_days": [data.validation_end + 1, end],
                    "selected_item_ids": list(data.item_ids)},
        "economics": vars(config),
        "baseline_selection": baseline,
        "baseline_validation_candidates": baseline_candidates,
        "model_selection": {
            "algorithm": "context-aware marginal-score CEM actor",
            "features": N_CONTEXT_FEATURES, "seeds": list(seeds),
            "iterations": iterations, "population": population, "window": window,
            "selected_seed": selected["seed"],
            "selected_iteration": selected["iteration"],
            "selected_theta": selected["theta"].tolist(),
            "selected_validation_profit": selected["validation_profit"],
            "validation_checkpoints": [
                {"seed": c["seed"], "iteration": c["iteration"],
                 "theta": c["theta"].tolist(), "profit": c["validation_profit"]}
                for c in fitted["checkpoints"]
            ],
            "training_histories": fitted["training_histories"],
        },
        "test": {"actor": actor, "strong_rule": rule,
                 "paired_profit_uplift": actor["profit"] - rule["profit"],
                 "paired_uplift_block_bootstrap_ci95": ci,
                 "p10_daily_profit_actor": p10_actor,
                 "p10_daily_profit_rule": p10_rule,
                 "promotion_eligible": bool(eligible),
                 "active_policy": "context_actor_rl" if eligible else "strong_rule"},
        "caveat": "Observed sales are a censored demand proxy; economics and lead times are simulated.",
    }
    output.mkdir(parents=True, exist_ok=True)
    save_context_model(selected["theta"], data, config, output, bool(eligible),
                       baseline_cover=baseline["cover"],
                       baseline_recent=baseline["recent_sales"],
                       baseline_beta=baseline["beta"])
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
