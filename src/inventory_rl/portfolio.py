"""Multi-SKU demand-replay simulator with shared purchasing and storage limits."""

from dataclasses import dataclass

import numpy as np

from inventory_rl.m5 import M5Series

ORDER_CHOICES = np.array([0, 4, 8, 16], dtype=np.int32)


def base_stock_scores(stock: np.ndarray, pipeline: np.ndarray, last_sales: np.ndarray,
                      mean_train: np.ndarray, lead: np.ndarray, *, cover: float,
                      recent: bool, beta: float = 1.0) -> np.ndarray:
    """Score feasible packs for the validation-tuned replenishment rule."""
    if not np.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and nonnegative")
    rate = last_sales.mean(axis=1) if recent else mean_train
    desired = np.maximum(rate * (lead + cover) - stock - pipeline, 0)
    return -((ORDER_CHOICES[None, :] - desired[:, None]) ** 2) / (rate[:, None] + 1) ** beta


def policy_search_scores(stock: np.ndarray, pipeline: np.ndarray,
                         last_sales: np.ndarray, mean_train: np.ndarray,
                         lead: np.ndarray, day: int, theta: np.ndarray, *,
                         cover: float, recent: bool) -> np.ndarray:
    """State-dependent cover adjustment learned from whole-portfolio return."""
    if theta.shape != (5,) or not np.isfinite(theta).all():
        raise ValueError("invalid policy-search coefficients")
    if not np.isfinite(cover) or cover < 0:
        raise ValueError("invalid baseline coverage")
    if not theta.any():
        return base_stock_scores(stock, pipeline, last_sales, mean_train, lead,
                                 cover=cover, recent=recent)
    denom = mean_train + 1.0
    trend = np.clip((last_sales.mean(axis=1) - mean_train) / denom, -2.0, 2.0)
    momentum = np.clip((last_sales[:, -3:].mean(axis=1) -
                        last_sales[:, :4].mean(axis=1)) / denom, -2.0, 2.0)
    volatility = np.clip(last_sales.std(axis=1) / denom, 0.0, 3.0)
    lead_feature = (lead - 2.0).astype(np.float64)
    phase = np.full(len(stock), np.sin(2 * np.pi * day / 7))
    features = np.column_stack((trend, momentum, volatility, lead_feature, phase))
    rate = last_sales.mean(axis=1) if recent else mean_train
    residual = 2.0 * np.tanh(features @ theta)
    desired = np.maximum(rate * (lead + cover + residual) - stock - pipeline, 0)
    return -((ORDER_CHOICES[None, :] - desired[:, None]) ** 2) / (rate[:, None] + 1)


def context_policy_scores(stock: np.ndarray, pipeline: np.ndarray,
                          last_sales: np.ndarray, mean_train: np.ndarray,
                          lead: np.ndarray, day: int, budget: float,
                          unit_cost: float, theta: np.ndarray, *,
                          cover: float, recent: bool, beta: float) -> np.ndarray:
    """Portfolio-aware SKU priority atop a strong replenishment rule."""
    if theta.shape != (9,) or not np.isfinite(theta).all():
        raise ValueError("invalid context actor coefficients")
    rule = base_stock_scores(stock, pipeline, last_sales, mean_train, lead,
                             cover=cover, recent=recent, beta=beta)
    if not theta.any():
        return rule
    sales = last_sales
    mean = mean_train
    rate = sales.mean(axis=1) if recent else mean
    denom = mean + 1.0
    trend = np.clip((sales.mean(axis=1) - mean) / denom, -2, 2)
    momentum = np.clip((sales[:, -3:].mean(axis=1) -
                        sales[:, :4].mean(axis=1)) / denom, -2, 2)
    volatility = np.clip(sales.std(axis=1) / denom, 0, 3)
    gap = np.clip((rate * (lead + cover) - stock - pipeline) /
                  (rate + 1), -4, 4) / 4
    pipeline_cover = np.clip(pipeline / (rate + 1), 0, 6) / 6
    velocity = np.clip(np.log1p(mean) / np.log(41), 0, 2)
    lead_feature = (lead - 2.0).astype(np.float64)
    budget_units = budget / unit_cost
    pressure = np.clip(rate.sum() / max(budget_units, 1) - 1, -2, 2)
    global_pressure = np.full(len(stock), pressure)
    phase = np.full(len(stock), np.sin(2 * np.pi * day / 7))
    features = np.column_stack((trend, momentum, volatility, gap, pipeline_cover,
                                velocity, lead_feature, global_pressure, phase))
    priority = np.tanh(features @ theta)
    scale = float(np.median(np.abs(np.diff(rule, axis=1))))
    return rule + scale * priority[:, None] * (ORDER_CHOICES[None, :] / 4.0)


def residual_scores(rule: np.ndarray, q_values: np.ndarray, alpha: float) -> np.ndarray:
    """Add a scaled Q-value residual without changing the exact alpha-zero rule."""
    if rule.shape != q_values.shape or rule.ndim != 2 or rule.shape[1] != len(ORDER_CHOICES):
        raise ValueError("invalid residual score dimensions")
    if not np.isfinite(rule).all() or not np.isfinite(q_values).all():
        raise ValueError("residual scores must be finite")
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and nonnegative")
    if alpha == 0:
        return rule.copy()
    rule_marginal = np.median(np.abs(np.diff(rule, axis=1)))
    q_marginal = np.median(np.abs(np.diff(q_values, axis=1)))
    scale = rule_marginal / max(q_marginal, 1e-6)
    return rule + alpha * scale * (q_values - q_values[:, :1])


@dataclass(frozen=True)
class PortfolioConfig:
    budget_per_sku: float = 32.0
    capacity_per_sku: int = 50
    unit_price: float = 10.0
    unit_cost: float = 4.0
    holding_cost: float = 0.15
    lost_sale_penalty: float = 2.0

    def __post_init__(self) -> None:
        if min(self.budget_per_sku, self.capacity_per_sku, self.unit_price,
               self.unit_cost) <= 0:
            raise ValueError("budget, capacity and prices must be positive")
        if min(self.holding_cost, self.lost_sale_penalty) < 0:
            raise ValueError("costs must be nonnegative")


def allocate(scores: np.ndarray, *, stock: np.ndarray, pipeline: np.ndarray,
             budget: float, capacity: int, unit_cost: float) -> np.ndarray:
    """Deterministic marginal-value allocator; always returns a feasible order.

    It greedily upgrades an item's current action by one pack level. Scores are
    policy-dependent estimates and need not be calibrated across time; within
    a decision they must be comparable across SKUs.
    """
    n_sku = len(stock)
    if scores.shape != (n_sku, len(ORDER_CHOICES)):
        raise ValueError("scores have wrong shape")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    if stock.shape != pipeline.shape or (stock < 0).any() or (pipeline < 0).any():
        raise ValueError("invalid stock or pipeline")
    if stock.sum() + pipeline.sum() > capacity:
        raise ValueError("current position exceeds capacity")
    selected = np.zeros(n_sku, dtype=np.int32)
    remaining_budget = budget
    remaining_capacity = capacity - int(stock.sum() + pipeline.sum())
    sku_indices = np.arange(n_sku)
    while True:
        next_level = np.minimum(selected + 1, len(ORDER_CHOICES) - 1)
        units = ORDER_CHOICES[next_level] - ORDER_CHOICES[selected]
        costs = units * unit_cost
        gains = scores[sku_indices, next_level] - scores[sku_indices, selected]
        valid = ((selected < len(ORDER_CHOICES) - 1) &
                 (costs <= remaining_budget + 1e-9) &
                 (units <= remaining_capacity) & (gains > 0))
        if not valid.any():
            break
        ratios = np.full(n_sku, -np.inf)
        ratios[valid] = gains[valid] / units[valid]
        best_ratio = ratios.max()
        finalists = valid & (ratios == best_ratio)
        best_gain = gains[finalists].max()
        sku = int(np.flatnonzero(finalists & (gains == best_gain))[0])
        selected[sku] += 1
        remaining_budget -= float(costs[sku])
        remaining_capacity -= int(units[sku])
    return ORDER_CHOICES[selected]


def allocate_exact(scores: np.ndarray, *, stock: np.ndarray, pipeline: np.ndarray,
                   budget: float, capacity: int, unit_cost: float) -> np.ndarray:
    """Maximize the sum of per-SKU scores subject to shared order limits.

    Every action is a multiple of four units, and all SKUs have the same unit
    cost. The budget and storage limits therefore collapse to one integer
    capacity in four-unit quanta. Dynamic programming solves the resulting
    multiple-choice knapsack exactly. Equal scores prefer fewer ordered units,
    then earlier SKUs; the result does not depend on dictionary or hash order.
    """
    n_sku = len(stock)
    if scores.shape != (n_sku, len(ORDER_CHOICES)) or not np.isfinite(scores).all():
        raise ValueError("scores must be a finite n_sku-by-4 array")
    if (stock.shape != (n_sku,) or pipeline.shape != (n_sku,) or
            not np.issubdtype(stock.dtype, np.integer) or
            not np.issubdtype(pipeline.dtype, np.integer) or
            (stock < 0).any() or (pipeline < 0).any()):
        raise ValueError("stock and pipeline must be nonnegative integer vectors")
    if (not np.isfinite(budget) or budget < 0 or
            not np.isfinite(unit_cost) or unit_cost <= 0 or
            isinstance(capacity, (bool, np.bool_)) or not np.isfinite(capacity) or
            capacity < 0 or int(capacity) != capacity):
        raise ValueError("invalid budget, capacity or unit cost")
    occupied = int(stock.sum(dtype=np.int64) + pipeline.sum(dtype=np.int64))
    if occupied > capacity:
        raise ValueError("current position exceeds capacity")

    maximum_order_units = int(ORDER_CHOICES[-1]) * n_sku
    budget_units = int(np.floor(min((budget + 1e-9) / unit_cost,
                                    maximum_order_units)))
    available_units = min(budget_units, int(capacity) - occupied,
                          maximum_order_units)
    max_quanta = available_units // 4
    weights = ORDER_CHOICES // 4
    delta = scores.astype(np.float64) - scores[:, :1]
    if not np.isfinite(delta).all():
        raise ValueError("score differences must be finite")

    best = np.full(max_quanta + 1, -np.inf, dtype=np.float64)
    best[0] = 0.0
    decisions = np.full((n_sku, max_quanta + 1), -1, dtype=np.int8)
    for sku in range(n_sku):
        following = np.full_like(best, -np.inf)
        for action, weight in enumerate(weights):
            if weight > max_quanta:
                continue
            candidates = best[:max_quanta + 1 - weight] + delta[sku, action]
            destinations = following[weight:]
            improve = candidates > destinations
            destinations[improve] = candidates[improve]
            decisions[sku, weight:][improve] = action
        best = following

    used_quanta = int(np.argmax(best))
    orders = np.zeros(n_sku, dtype=np.int32)
    for sku in range(n_sku - 1, -1, -1):
        action = int(decisions[sku, used_quanta])
        if action < 0:
            raise RuntimeError("exact allocation reconstruction failed")
        orders[sku] = ORDER_CHOICES[action]
        used_quanta -= int(weights[action])
    if used_quanta != 0:
        raise RuntimeError("exact allocation reconstruction used the wrong capacity")
    return orders


class PortfolioEnv:
    """Replay observed M5 sales as an exogenous demand proxy for selected SKUs."""

    def __init__(self, data: M5Series, config: PortfolioConfig | None = None):
        self.data = data
        self.config = config or PortfolioConfig()
        self.lead = 1 + np.arange(data.n_sku) % 3
        self.mean_train = data.sales[:, max(0, data.train_end - 365):data.train_end].mean(axis=1)
        self.budget = self.config.budget_per_sku * data.n_sku
        self.capacity = self.config.capacity_per_sku * data.n_sku

    def reset(self, start: int, end: int) -> np.ndarray:
        if not 28 <= start < end <= self.data.sales.shape[1]:
            raise ValueError("invalid replay window")
        self.day = start
        self.end = end
        recent = self.data.sales[:, start - 28:start].mean(axis=1)
        self.stock = np.minimum(np.ceil(recent * (self.lead + 1)), 40).astype(np.int32)
        self.pipeline = np.zeros((3, self.data.n_sku), dtype=np.int32)
        self.last_sales = self.data.sales[:, start - 7:start].astype(np.float64).copy()
        if self.stock.sum() > self.capacity:
            raise ValueError("initial position exceeds capacity")
        return self.observation()

    def observation(self) -> np.ndarray:
        day_sin = np.sin(2 * np.pi * self.day / 7)
        day_cos = np.cos(2 * np.pi * self.day / 7)
        return np.column_stack((
            np.minimum(self.stock / 40.0, 5),
            np.minimum(self.pipeline.sum(axis=0) / 40.0, 5),
            np.minimum(self.last_sales[:, -1] / 40.0, 5),
            np.minimum(self.last_sales.mean(axis=1) / 40.0, 5),
            np.minimum(self.mean_train / 40.0, 5),
            self.lead / 3.0,
            np.full(self.data.n_sku, day_sin),
            np.full(self.data.n_sku, day_cos),
        )).astype(np.float64)

    def step(self, orders: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool, dict]:
        if self.day >= self.end:
            raise RuntimeError("reset before stepping a completed replay")
        if orders.shape != (self.data.n_sku,) or not np.isin(orders, ORDER_CHOICES).all():
            raise ValueError("invalid order shape or pack size")
        if orders.sum() * self.config.unit_cost > self.budget + 1e-9:
            raise ValueError("purchasing budget exceeded")
        if self.stock.sum() + self.pipeline.sum() + orders.sum() > self.capacity:
            raise ValueError("storage capacity exceeded")
        demand = self.data.sales[:, self.day]
        sold = np.minimum(self.stock, demand)
        lost = demand - sold
        self.stock -= sold
        for sku, lead in enumerate(self.lead):
            self.pipeline[lead - 1, sku] += orders[sku]
        reward = (sold * self.config.unit_price - orders * self.config.unit_cost
                  - self.stock * self.config.holding_cost
                  - lost * self.config.lost_sale_penalty).astype(np.float64)
        arrivals = self.pipeline[0].copy()
        self.pipeline[:-1] = self.pipeline[1:]
        self.pipeline[-1] = 0
        self.stock += arrivals
        self.last_sales[:, :-1] = self.last_sales[:, 1:]
        self.last_sales[:, -1] = sold
        self.day += 1
        info = {"demand": int(demand.sum()), "sold": int(sold.sum()),
                "lost": int(lost.sum()), "spend": float(orders.sum() * self.config.unit_cost),
                "stock": int(self.stock.sum()), "profit": float(reward.sum())}
        return self.observation(), reward, self.day >= self.end, info
