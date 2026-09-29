"""Operational baselines under the same action constraints as the agent."""

import numpy as np

from inventory_rl.env import InventoryEnv


def base_stock_action(env: InventoryEnv) -> int:
    """Order toward lead-time demand plus a one-period safety buffer."""
    config = env.config
    cover = np.asarray(config.mean_demand) * config.demand_multiplier * (
        np.asarray(config.lead_time) + 1.0
    )
    position = env.stock + env.pipeline.sum(axis=0)
    desired = np.maximum(cover - position, 0)
    candidates = env.actions.astype(float)
    score = ((candidates - desired) ** 2 / (cover + 1)).sum(axis=1)
    score[~env.valid_actions()] = np.inf
    return int(score.argmin())


def random_action(env: InventoryEnv, rng: np.random.Generator) -> int:
    return int(rng.choice(np.flatnonzero(env.valid_actions())))
