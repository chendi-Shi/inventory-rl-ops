"""The exact shared-limit allocator is checked against enumeration."""

from itertools import product

import numpy as np
import pytest

from inventory_rl.portfolio import ORDER_CHOICES, allocate, allocate_exact


def _best_enumerated_score(scores: np.ndarray, *, budget: float,
                           capacity: int, unit_cost: float) -> float:
    best = -np.inf
    for levels in product(range(len(ORDER_CHOICES)), repeat=len(scores)):
        units = int(ORDER_CHOICES[list(levels)].sum())
        if units <= capacity and units * unit_cost <= budget + 1e-9:
            best = max(best, float(scores[np.arange(len(scores)), levels].sum()))
    return best


def test_exact_matches_exhaustive_enumeration_and_shared_limits():
    rng = np.random.default_rng(2027)
    for n_sku in (2, 3, 4):
        for _ in range(30):
            scores = rng.integers(-20, 21, size=(n_sku, 4)).astype(float)
            stock = rng.integers(0, 5, size=n_sku, dtype=np.int32)
            pipeline = rng.integers(0, 4, size=n_sku, dtype=np.int32)
            free_capacity = int(rng.integers(0, 16 * n_sku + 1))
            capacity = int(stock.sum() + pipeline.sum()) + free_capacity
            unit_cost = 2.5
            budget = float(rng.integers(0, 16 * n_sku + 1)) * unit_cost
            orders = allocate_exact(scores, stock=stock, pipeline=pipeline,
                                    budget=budget, capacity=capacity,
                                    unit_cost=unit_cost)
            assert np.isin(orders, ORDER_CHOICES).all()
            assert orders.sum() <= free_capacity
            assert orders.sum() * unit_cost <= budget + 1e-9
            levels = np.searchsorted(ORDER_CHOICES, orders)
            actual_score = float(scores[np.arange(n_sku), levels].sum())
            expected_score = _best_enumerated_score(
                scores, budget=budget, capacity=free_capacity, unit_cost=unit_cost
            )
            assert actual_score == pytest.approx(expected_score)


def test_indivisible_pack_can_make_greedy_suboptimal():
    # Both score rows are concave quadratics of the order quantity.
    scores = np.array([
        [-256.0, -144.0, -64.0, 0.0],
        [-42.25, -6.25, -2.25, -90.25],
    ])
    stock = np.zeros(2, dtype=np.int32)
    pipeline = stock.copy()
    kwargs = {"stock": stock, "pipeline": pipeline, "budget": 64.0,
              "capacity": 16, "unit_cost": 4.0}
    np.testing.assert_array_equal(allocate(scores, **kwargs), [8, 8])
    np.testing.assert_array_equal(allocate_exact(scores, **kwargs), [16, 0])


def test_exact_ties_prefer_less_spend_then_earlier_sku():
    stock = np.zeros(2, dtype=np.int32)
    kwargs = {"stock": stock, "pipeline": stock.copy(), "budget": 64.0,
              "capacity": 16, "unit_cost": 4.0}
    np.testing.assert_array_equal(
        allocate_exact(np.zeros((2, 4)), **kwargs), [0, 0]
    )
    scores = np.array([[0.0, 10.0, -1.0, -2.0],
                       [0.0, 10.0, -1.0, -2.0]])
    np.testing.assert_array_equal(
        allocate_exact(scores, **{**kwargs, "capacity": 4}), [4, 0]
    )


def test_exact_rejects_invalid_inputs_and_caps_unused_budget():
    scores = np.zeros((2, 4))
    stock = np.zeros(2, dtype=np.int32)
    kwargs = {"stock": stock, "pipeline": stock.copy(), "budget": 64.0,
              "capacity": 16, "unit_cost": 4.0}
    with pytest.raises(ValueError, match="scores"):
        allocate_exact(np.full((2, 4), np.nan), **kwargs)
    with pytest.raises(ValueError, match="stock and pipeline"):
        allocate_exact(scores, **{**kwargs, "stock": np.array([-1, 0])})
    with pytest.raises(ValueError, match="current position"):
        allocate_exact(scores, **{**kwargs, "stock": np.array([9, 8])})
    with pytest.raises(ValueError, match="invalid budget"):
        allocate_exact(scores, **{**kwargs, "unit_cost": 0.0})
    with pytest.raises(ValueError, match="invalid budget"):
        allocate_exact(scores, **{**kwargs, "budget": -1.0})
    big = allocate_exact(np.tile([0.0, 1.0, 2.0, 3.0], (64, 1)),
                         stock=np.zeros(64, dtype=np.int32),
                         pipeline=np.zeros(64, dtype=np.int32),
                         budget=1e9, capacity=1_000_000, unit_cost=4.0)
    np.testing.assert_array_equal(big, np.full(64, 16))
