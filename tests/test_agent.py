from pathlib import Path

import numpy as np
import pytest

from inventory_rl.agent import DQNAgent, QNetwork, Transition
from inventory_rl.artifact import load_model, save_model
from inventory_rl.env import InventoryConfig, InventoryEnv
from inventory_rl.evaluation import compare


def test_network_update_reduces_supervised_error():
    rng = np.random.default_rng(3)
    net = QNetwork(2, 2, 16, rng)
    states = np.array([[1.0, 0.0], [0.0, 1.0]] * 16)
    actions = np.array([0, 1] * 16)
    targets = np.array([0.8, -0.5] * 16)
    initial = np.mean((net.predict(states)[np.arange(32), actions] - targets) ** 2)
    for _ in range(100):
        net.update(states, actions, targets, 0.003)
    final = np.mean((net.predict(states)[np.arange(32), actions] - targets) ** 2)
    assert final < initial / 10


def test_agent_respects_action_mask_and_learns():
    agent = DQNAgent(2, 3, seed=9, batch_size=4, warmup=4, target_interval=2)
    mask = np.array([False, True, False])
    for _ in range(8):
        assert agent.act(np.array([0.2, 0.3]), mask, epsilon=1) == 1
        loss = agent.learn(Transition(
            np.array([0.2, 0.3]), 1, 2.0, np.array([0.3, 0.4]), mask, False
        ))
    assert loss is not None and np.isfinite(loss)


def test_artifact_roundtrip_and_checksum():
    tmp_path = Path("artifacts/_test_roundtrip")
    config = InventoryConfig()
    env = InventoryEnv(config)
    agent = DQNAgent(len(env.observation()), len(env.actions), seed=5)
    state, mask = env.observation(), env.valid_actions()
    save_model(agent, config, tmp_path)
    restored, restored_config, _ = load_model(tmp_path)
    assert restored_config == config
    np.testing.assert_allclose(restored.online.predict(state[None]),
                               agent.online.predict(state[None]))
    assert restored.act(state, mask) == agent.act(state, mask)
    with (tmp_path / "model.npz").open("ab") as file:
        file.write(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        load_model(tmp_path)


def test_paired_evaluation_is_reproducible():
    env = InventoryEnv()
    agent = DQNAgent(len(env.observation()), len(env.actions), seed=2)
    first = compare(agent, env.config, [100, 101, 102])
    second = compare(agent, env.config, [100, 101, 102])
    assert first == second
