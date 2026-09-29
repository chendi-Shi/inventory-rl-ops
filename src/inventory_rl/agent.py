"""Small NumPy Double DQN with replay, target network and Huber loss."""

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    next_mask: np.ndarray
    done: bool


class QNetwork:
    def __init__(self, n_state: int, n_action: int, hidden: int, rng: np.random.Generator):
        self.params = {
            "w1": rng.normal(0, np.sqrt(2 / n_state), (n_state, hidden)),
            "b1": np.zeros(hidden),
            "w2": rng.normal(0, np.sqrt(2 / hidden), (hidden, n_action)),
            "b2": np.zeros(n_action),
        }
        self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.updates = 0

    def forward(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        z = states @ self.params["w1"] + self.params["b1"]
        h = np.maximum(z, 0)
        q = h @ self.params["w2"] + self.params["b2"]
        return q, z, h

    def predict(self, states: np.ndarray) -> np.ndarray:
        return self.forward(states)[0]

    def copy_from(self, other: "QNetwork") -> None:
        for key in self.params:
            self.params[key][...] = other.params[key]

    def update(self, states: np.ndarray, actions: np.ndarray,
               targets: np.ndarray, learning_rate: float) -> float:
        q, z, h = self.forward(states)
        error = q[np.arange(len(actions)), actions] - targets
        abs_error = np.abs(error)
        loss = np.where(abs_error <= 1, 0.5 * error**2, abs_error - 0.5).mean()
        dq = np.zeros_like(q)
        dq[np.arange(len(actions)), actions] = np.clip(error, -1, 1) / len(actions)
        grads = {
            "w2": h.T @ dq,
            "b2": dq.sum(axis=0),
        }
        dh = (dq @ self.params["w2"].T) * (z > 0)
        grads["w1"] = states.T @ dh
        grads["b1"] = dh.sum(axis=0)
        self.updates += 1
        for key, grad in grads.items():
            self.m[key] = 0.9 * self.m[key] + 0.1 * grad
            self.v[key] = 0.999 * self.v[key] + 0.001 * grad**2
            m_hat = self.m[key] / (1 - 0.9**self.updates)
            v_hat = self.v[key] / (1 - 0.999**self.updates)
            self.params[key] -= learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)
        return float(loss)


class DQNAgent:
    def __init__(self, n_state: int, n_action: int, seed: int = 0,
                 hidden: int = 64, gamma: float = 0.97, learning_rate: float = 0.0007,
                 replay_capacity: int = 20_000, batch_size: int = 64,
                 warmup: int = 256, target_interval: int = 200):
        self.rng = np.random.default_rng(seed)
        self.online = QNetwork(n_state, n_action, hidden, self.rng)
        self.target = QNetwork(n_state, n_action, hidden, self.rng)
        self.target.copy_from(self.online)
        self.replay: deque[Transition] = deque(maxlen=replay_capacity)
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.warmup = warmup
        self.target_interval = target_interval
        self.steps = 0

    def act(self, state: np.ndarray, mask: np.ndarray, epsilon: float = 0) -> int:
        valid = np.flatnonzero(mask)
        if len(valid) == 0:
            raise ValueError("no valid actions")
        if self.rng.random() < epsilon:
            return int(self.rng.choice(valid))
        values = self.online.predict(state[None, :])[0].copy()
        values[~mask] = -np.inf
        return int(values.argmax())

    def learn(self, transition: Transition) -> float | None:
        self.replay.append(transition)
        return self.optimize()

    def observe_many(self, transitions: list[Transition]) -> float | None:
        """Add a portfolio's item transitions and perform one shared-network update."""
        self.replay.extend(transitions)
        return self.optimize()

    def optimize(self) -> float | None:
        self.steps += 1
        if len(self.replay) < self.warmup:
            return None
        batch = [self.replay[i] for i in self.rng.integers(len(self.replay), size=self.batch_size)]
        states = np.stack([t.state for t in batch])
        next_states = np.stack([t.next_state for t in batch])
        masks = np.stack([t.next_mask for t in batch])
        next_q = self.online.predict(next_states)
        next_q[~masks] = -np.inf
        next_actions = next_q.argmax(axis=1)
        bootstrap = self.target.predict(next_states)[np.arange(len(batch)), next_actions]
        rewards = np.asarray([t.reward for t in batch]) / 100.0
        dones = np.asarray([t.done for t in batch])
        targets = rewards + self.gamma * bootstrap * (~dones)
        loss = self.online.update(
            states, np.asarray([t.action for t in batch]), targets, self.learning_rate
        )
        if self.steps % self.target_interval == 0:
            self.target.copy_from(self.online)
        return loss
