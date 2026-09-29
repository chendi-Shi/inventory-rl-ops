"""Leakage-conscious M5 portfolio training, model selection and final backtest."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from inventory_rl.agent import DQNAgent, Transition
from inventory_rl.m5 import M5Series, load_m5
from inventory_rl.portfolio import ORDER_CHOICES, PortfolioConfig, PortfolioEnv, allocate
from inventory_rl.portfolio_artifact import save_portfolio_model


def rule_scores(env: PortfolioEnv, cover: float, recent: bool) -> np.ndarray:
    rate = env.last_sales.mean(axis=1) if recent else env.mean_train
    desired = np.maximum(rate * (env.lead + cover)
                         - env.stock - env.pipeline.sum(axis=0), 0)
    return -((ORDER_CHOICES[None, :] - desired[:, None]) ** 2) / (rate[:, None] + 1)


def replay(data: M5Series, config: PortfolioConfig, start: int, end: int,
           agent: DQNAgent | None = None, *, cover: float = 1.0,
           recent: bool = True, random_seed: int | None = None) -> dict:
    env = PortfolioEnv(data, config)
    state = env.reset(start, end)
    rng = np.random.default_rng(random_seed)
    daily = []
    demand = sold = spend = 0.0
    while env.day < end:
        if random_seed is not None:
            scores = rng.random((data.n_sku, len(ORDER_CHOICES)))
        elif agent is not None:
            scores = agent.online.predict(state)
        else:
            scores = rule_scores(env, cover, recent)
        orders = allocate(scores, stock=env.stock, pipeline=env.pipeline.sum(axis=0),
                          budget=env.budget, capacity=env.capacity,
                          unit_cost=config.unit_cost)
        state, reward, _, info = env.step(orders)
        daily.append(float(reward.sum()))
        demand += info["demand"]
        sold += info["sold"]
        spend += info["spend"]
    return {"profit": float(sum(daily)), "daily_profit": daily,
            "fill_rate": float(sold / max(demand, 1)),
            "mean_daily_spend": float(spend / (end - start)),
            "days": end - start, "sku_count": data.n_sku}


def block_bootstrap_ci(a: list[float], b: list[float], *, block: int = 7,
                       samples: int = 2000) -> list[float]:
    """Confidence interval for total uplift using paired circular moving blocks."""
    delta = np.asarray(a) - np.asarray(b)
    if len(delta) < block:
        raise ValueError("evaluation period shorter than bootstrap block")
    rng = np.random.default_rng(2026)
    blocks = (len(delta) + block - 1) // block
    starts = rng.integers(len(delta), size=(samples, blocks))
    offsets = np.arange(block)
    indices = (starts[:, :, None] + offsets) % len(delta)
    totals = delta[indices.reshape(samples, -1)[:, :len(delta)]].sum(axis=1)
    return [float(x) for x in np.quantile(totals, [0.025, 0.975])]


def train_one(data: M5Series, config: PortfolioConfig, seed: int,
              episodes: int = 60, window: int = 84) -> tuple[DQNAgent, dict]:
    if episodes < 1 or window < 7 or data.train_end - window <= 365:
        raise ValueError("not enough training history or invalid training settings")
    rng = np.random.default_rng(seed)
    agent = DQNAgent(8, len(ORDER_CHOICES), seed=seed, batch_size=128,
                     warmup=512, replay_capacity=30_000)
    best = None
    best_params = None
    history = []
    for episode in range(episodes):
        start = int(rng.integers(365, data.train_end - window + 1))
        env = PortfolioEnv(data, config)
        state = env.reset(start, start + window)
        epsilon = max(0.05, 0.8 * (1 - episode / episodes))
        for _ in range(window):
            scores = agent.online.predict(state)
            if rng.random() < epsilon:
                scores = rng.random(scores.shape)
            orders = allocate(scores, stock=env.stock, pipeline=env.pipeline.sum(axis=0),
                              budget=env.budget, capacity=env.capacity,
                              unit_cost=config.unit_cost)
            next_state, rewards, done, _ = env.step(orders)
            transitions = [Transition(state[i].copy(), int(np.searchsorted(ORDER_CHOICES, orders[i])),
                                      float(rewards[i]), next_state[i].copy(),
                                      np.ones(len(ORDER_CHOICES), dtype=bool), done)
                           for i in range(data.n_sku)]
            agent.observe_many(transitions)
            state = next_state
        if (episode + 1) % 5 == 0 or episode + 1 == episodes:
            metric = replay(data, config, data.train_end, data.validation_end, agent)
            history.append({"episode": episode + 1, "validation_profit": metric["profit"],
                            "validation_fill_rate": metric["fill_rate"]})
            if best is None or metric["profit"] > best:
                best = metric["profit"]
                best_params = {k: value.copy() for k, value in agent.online.params.items()}
    for key, value in best_params.items():
        agent.online.params[key][...] = value
    agent.target.copy_from(agent.online)
    return agent, {"seed": seed, "episodes": episodes, "window": window,
                   "validation_history": history,
                   "selected_episode": max(history, key=lambda x: x["validation_profit"])["episode"]}


def run(path: Path, output: Path, *, store_id: str = "CA_1", sku_count: int = 64,
        seeds: tuple[int, ...] = (11, 22, 33), episodes: int = 60,
        train_end: int = 1700, validation_end: int = 1800) -> dict:
    if not seeds:
        raise ValueError("at least one training seed is required")
    data = load_m5(path, store_id=store_id, sku_count=sku_count,
                   train_end=train_end, validation_end=validation_end)
    config = PortfolioConfig()
    candidates = []
    for recent in (False, True):
        for cover in (0.5, 1.0, 1.5, 2.0, 3.0):
            validation = replay(data, config, data.train_end, data.validation_end,
                                cover=cover, recent=recent)
            candidates.append((validation["profit"], recent, cover, validation))
    _, baseline_recent, baseline_cover, baseline_validation = max(candidates)
    models = []
    for seed in seeds:
        agent, training = train_one(data, config, seed, episodes)
        validation = replay(data, config, data.train_end, data.validation_end, agent)
        models.append((validation["profit"], seed, agent, training, validation))
    _, selected_seed, selected_agent, selected_training, selected_validation = max(models)
    # No test data was read by policy selection above; this is the first final test evaluation.
    start, end = data.validation_end, data.sales.shape[1]
    test_rl = replay(data, config, start, end, selected_agent)
    test_baseline = replay(data, config, start, end, cover=baseline_cover,
                           recent=baseline_recent)
    test_random = replay(data, config, start, end, random_seed=2026)
    ci = block_bootstrap_ci(test_rl["daily_profit"], test_baseline["daily_profit"])
    p10_rl = float(np.quantile(test_rl["daily_profit"], 0.1))
    p10_base = float(np.quantile(test_baseline["daily_profit"], 0.1))
    eligible = ci[0] > 0 and test_rl["fill_rate"] >= test_baseline["fill_rate"] - 0.02 and p10_rl >= p10_base
    report = {
        "dataset": {"name": "M5 sales_train_validation", "source_sha256": data.source_sha256,
                    "store_id": store_id, "sku_count": data.n_sku,
                    "train_days": [1, data.train_end],
                    "validation_days": [data.train_end + 1, data.validation_end],
                    "test_days": [data.validation_end + 1, end],
                    "selected_item_ids": list(data.item_ids)},
        "economics": asdict(config),
        "model_selection": {"seeds": list(seeds), "selected_seed": selected_seed,
                            "selected_training": selected_training,
                            "validation_profit_by_seed": {str(seed): profit for profit, seed, *_ in models},
                            "selected_validation": selected_validation},
        "baseline_selection": {"recent_sales": baseline_recent, "cover": baseline_cover,
                               "validation": baseline_validation},
        "test": {"rl": test_rl, "base_stock": test_baseline, "random": test_random,
                 "paired_profit_uplift": test_rl["profit"] - test_baseline["profit"],
                 "paired_uplift_block_bootstrap_ci95": ci,
                 "p10_daily_profit_rl": p10_rl, "p10_daily_profit_base_stock": p10_base,
                 "promotion_eligible": bool(eligible)},
        "caveat": "Observed sales are a censored demand proxy; procurement economics and lead times are simulated.",
    }
    output.mkdir(parents=True, exist_ok=True)
    save_portfolio_model(selected_agent, data, config, output, bool(eligible))
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
