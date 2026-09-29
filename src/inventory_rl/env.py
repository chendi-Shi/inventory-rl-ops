"""Seeded, auditable inventory simulator. All costs are synthetic currency units."""

from dataclasses import asdict, dataclass
from itertools import product

import numpy as np


@dataclass(frozen=True)
class InventoryConfig:
    horizon: int = 60
    initial_stock: tuple[int, int, int] = (12, 10, 8)
    mean_demand: tuple[float, float, float] = (4.0, 3.0, 2.5)
    lead_time: tuple[int, int, int] = (1, 2, 2)
    unit_price: tuple[float, float, float] = (9.0, 12.0, 16.0)
    unit_cost: tuple[float, float, float] = (4.0, 5.0, 7.0)
    holding_cost: tuple[float, float, float] = (0.30, 0.40, 0.55)
    lost_sale_penalty: tuple[float, float, float] = (2.0, 2.5, 3.0)
    daily_budget: float = 70.0
    storage_capacity: int = 80
    order_sizes: tuple[int, int, int] = (0, 4, 8)
    demand_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if self.horizon < 1 or self.daily_budget <= 0 or self.storage_capacity < 1:
            raise ValueError("horizon, budget and capacity must be positive")
        vectors = (self.initial_stock, self.mean_demand, self.lead_time,
                   self.unit_price, self.unit_cost, self.holding_cost,
                   self.lost_sale_penalty)
        if any(len(v) != 3 for v in vectors):
            raise ValueError("exactly three SKUs are required")
        if any(x < 0 for v in vectors for x in v) or any(x < 1 for x in self.lead_time):
            raise ValueError("inventory parameters must be nonnegative; lead time >= 1")
        if sum(self.initial_stock) > self.storage_capacity:
            raise ValueError("initial stock exceeds capacity")
        if self.demand_multiplier <= 0:
            raise ValueError("demand multiplier must be positive")
        if 0 not in self.order_sizes or any(x < 0 for x in self.order_sizes):
            raise ValueError("order sizes must contain zero and be nonnegative")

    def to_dict(self) -> dict:
        return asdict(self)


class InventoryEnv:
    """Lost sales, fixed lead times, and hard purchasing and storage constraints.

    At decision time, today's arrivals are already received. Orders arrive after
    `lead_time` future demand periods; the pipeline is fully observed.
    """

    def __init__(self, config: InventoryConfig | None = None, seed: int = 0):
        self.config = config or InventoryConfig()
        self.actions = np.asarray(list(product(self.config.order_sizes, repeat=3)), dtype=np.int64)
        self.reset(seed)

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.day = 0
        self.stock = np.asarray(self.config.initial_stock, dtype=np.int64).copy()
        self.pipeline = np.zeros((max(self.config.lead_time), 3), dtype=np.int64)
        self.last_demand = np.asarray(self.config.mean_demand, dtype=np.float64)
        self.done = False
        return self.observation()

    def observation(self) -> np.ndarray:
        phase = 2 * np.pi * self.day / 14
        return np.concatenate((
            self.stock / 30.0,
            self.pipeline.flatten() / 30.0,
            self.last_demand / 15.0,
            [np.sin(phase), np.cos(phase), self.day / self.config.horizon],
        )).astype(np.float64)

    def valid_actions(self) -> np.ndarray:
        spend = self.actions @ np.asarray(self.config.unit_cost)
        occupied = int(self.stock.sum() + self.pipeline.sum())
        return (spend <= self.config.daily_budget + 1e-9) & (
            occupied + self.actions.sum(axis=1) <= self.config.storage_capacity
        )

    def step(self, action_id: int) -> tuple[np.ndarray, float, bool, dict]:
        if self.done:
            raise RuntimeError("reset before stepping a completed episode")
        if not 0 <= action_id < len(self.actions) or not self.valid_actions()[action_id]:
            raise ValueError("invalid action: purchasing budget or storage constraint exceeded")
        order = self.actions[action_id].copy()
        cost = float(order @ np.asarray(self.config.unit_cost))
        # Slot zero is due after today's demand; lead_time=1 arrives next decision.
        for sku, lead in enumerate(self.config.lead_time):
            self.pipeline[lead - 1, sku] += order[sku]
        phase = 2 * np.pi * self.day / 14
        weekly = 1.0 + 0.25 * np.sin(phase + np.arange(3) * 0.7)
        rates = np.asarray(self.config.mean_demand) * weekly * self.config.demand_multiplier
        demand = self.rng.poisson(rates)
        sold = np.minimum(self.stock, demand)
        lost = demand - sold
        self.stock -= sold
        revenue = float(sold @ np.asarray(self.config.unit_price))
        holding = float(self.stock @ np.asarray(self.config.holding_cost))
        penalty = float(lost @ np.asarray(self.config.lost_sale_penalty))
        reward = revenue - cost - holding - penalty
        arrivals = self.pipeline[0].copy()
        self.pipeline[:-1] = self.pipeline[1:]
        self.pipeline[-1] = 0
        self.stock += arrivals
        self.last_demand = demand.astype(np.float64)
        self.day += 1
        self.done = self.day >= self.config.horizon
        info = {"revenue": revenue, "procurement": cost, "holding": holding,
                "lost_sale_penalty": penalty, "demand": int(demand.sum()),
                "sold": int(sold.sum()), "lost": int(lost.sum()),
                "spend": cost, "order": order.tolist()}
        return self.observation(), reward, self.done, info
