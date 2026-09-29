import numpy as np
import pytest

from inventory_rl.env import InventoryConfig, InventoryEnv
from inventory_rl.policies import base_stock_action


def test_seed_reproduces_full_trajectory():
    trajectories = []
    for _ in range(2):
        env = InventoryEnv(seed=27)
        rows = []
        while not env.done:
            _, reward, _, info = env.step(base_stock_action(env))
            rows.append((reward, info["demand"], info["sold"], tuple(info["order"])))
        trajectories.append(rows)
    assert trajectories[0] == trajectories[1]


def test_conservation_and_hard_constraints():
    env = InventoryEnv(seed=19)
    for _ in range(env.config.horizon):
        mask = env.valid_actions()
        assert mask.any()
        action = int(np.flatnonzero(mask)[-1])
        previous_stock = env.stock.copy()
        previous_pipeline = env.pipeline.copy()
        _, reward, _, info = env.step(action)
        assert info["demand"] == info["sold"] + info["lost"]
        assert info["spend"] <= env.config.daily_budget
        assert env.stock.sum() + env.pipeline.sum() <= env.config.storage_capacity
        assert reward == pytest.approx(
            info["revenue"] - info["procurement"] - info["holding"]
            - info["lost_sale_penalty"]
        )
        assert np.all(env.stock >= 0) and np.all(env.pipeline >= 0)
        assert info["sold"] <= int(previous_stock.sum())
        assert int(env.stock.sum() + env.pipeline.sum()) <= (
            int(previous_stock.sum() + previous_pipeline.sum()) + sum(info["order"])
        )


def test_invalid_action_rejected_without_mutation():
    env = InventoryEnv(InventoryConfig(daily_budget=1), seed=1)
    before = env.observation().copy()
    invalid = int(np.flatnonzero(~env.valid_actions())[0])
    with pytest.raises(ValueError):
        env.step(invalid)
    np.testing.assert_array_equal(env.observation(), before)
