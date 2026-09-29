"""Paired-seed evaluation and conservative model-promotion checks."""

from collections.abc import Callable

import numpy as np

from inventory_rl.env import InventoryConfig, InventoryEnv
from inventory_rl.policies import base_stock_action, random_action


def rollout(config: InventoryConfig, seed: int,
            policy: Callable[[InventoryEnv], int]) -> dict[str, float]:
    env = InventoryEnv(config, seed)
    totals = {key: 0.0 for key in
              ("profit", "revenue", "procurement", "holding", "lost_sale_penalty",
               "demand", "sold", "lost")}
    while not env.done:
        _, reward, _, info = env.step(policy(env))
        totals["profit"] += reward
        for key in totals.keys() - {"profit"}:
            totals[key] += info[key]
    totals["fill_rate"] = totals["sold"] / max(totals["demand"], 1)
    return totals


def compare(agent, config: InventoryConfig, seeds: list[int]) -> dict:
    if not seeds:
        raise ValueError("provide at least one evaluation seed")
    results = {}
    for name in ("dqn", "base_stock", "random"):
        episodes = []
        for seed in seeds:
            rng = np.random.default_rng(seed + 10_000)
            if name == "dqn":
                policy = lambda env: agent.act(env.observation(), env.valid_actions())
            elif name == "base_stock":
                policy = base_stock_action
            else:
                policy = lambda env, rng=rng: random_action(env, rng)
            episodes.append(rollout(config, seed, policy))
        results[name] = {
            "mean_profit": float(np.mean([x["profit"] for x in episodes])),
            "profit_std": float(np.std([x["profit"] for x in episodes], ddof=0)),
            "p10_profit": float(np.quantile([x["profit"] for x in episodes], 0.1)),
            "mean_fill_rate": float(np.mean([x["fill_rate"] for x in episodes])),
            "episodes": episodes,
        }
    uplift = np.asarray([x["profit"] - y["profit"] for x, y in zip(
        results["dqn"]["episodes"], results["base_stock"]["episodes"]
    )])
    results["paired_mean_uplift"] = float(uplift.mean())
    results["paired_uplift_ci95"] = [float(x) for x in np.quantile(
        np.random.default_rng(2026).choice(uplift, size=(2000, len(uplift)), replace=True).mean(axis=1),
        [0.025, 0.975],
    )]
    results["promotion"] = {
        "eligible": bool(results["paired_uplift_ci95"][0] > 0 and
                         results["dqn"]["mean_fill_rate"] >=
                         results["base_stock"]["mean_fill_rate"] - 0.02 and
                         results["dqn"]["p10_profit"] >= results["base_stock"]["p10_profit"]),
        "rules": "lower 95% bootstrap CI of paired profit uplift > 0; fill-rate drop <= 2pp; p10 profit >= baseline",
    }
    results["seeds"] = seeds
    return results
