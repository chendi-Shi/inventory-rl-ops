"""Safety-stock diagnostics must respect training and validation boundaries."""

from pathlib import Path

import numpy as np
import pytest

from inventory_rl.m5 import M5Series
from inventory_rl.m5_safety_stock import (
    replay_safety_stock,
    run_safety_stock_diagnostic,
    safety_stock_scores,
    select_safety_stock,
    training_safety_premium,
)
from inventory_rl.portfolio import PortfolioConfig, PortfolioEnv, base_stock_scores


def series(*, alter_validation: bool = False, alter_test: bool = False) -> M5Series:
    days = np.arange(90)
    sales = np.vstack(((days % 7 + 1) * 2,
                       (days % 5 + 2) * 3,
                       (days % 11 + 1)))
    if alter_validation:
        sales[:, 60:70] += 12
    if alter_test:
        sales[:, 70:] += 100
    return M5Series("FIXTURE", ("sku-a", "sku-b", "sku-c"),
                    sales.astype(np.int32), 60, 70, "fixture-sha")


def test_training_quantiles_never_read_validation_or_test() -> None:
    original = series()
    changed = series(alter_validation=True, alter_test=True)
    lead = np.array([1, 2, 3])
    for quantile in (0.7, 0.85, 0.95):
        expected = training_safety_premium(original, lead, cover=2, quantile=quantile)
        observed = training_safety_premium(changed, lead, cover=2, quantile=quantile)
        np.testing.assert_array_equal(observed, expected)
    low = training_safety_premium(original, lead, cover=2, quantile=0.7)
    high = training_safety_premium(original, lead, cover=2, quantile=0.95)
    assert np.all(high >= low)


def test_constant_demand_reduces_to_original_mean_rule() -> None:
    sales = np.vstack((np.full(90, 2), np.full(90, 4), np.full(90, 6))).astype(np.int32)
    data = M5Series("FIXTURE", ("a", "b", "c"), sales, 60, 70, "constant")
    env = PortfolioEnv(data, PortfolioConfig())
    env.reset(60, 70)
    premium = training_safety_premium(data, env.lead, cover=2, quantile=0.95)
    np.testing.assert_array_equal(premium, np.zeros(3))
    actual = safety_stock_scores(env, premium, cover=2, recent=False, beta=1)
    expected = base_stock_scores(env.stock, env.pipeline.sum(axis=0), env.last_sales,
                                 env.mean_train, env.lead, cover=2, recent=False, beta=1)
    np.testing.assert_array_equal(actual, expected)
    result = replay_safety_stock(data, env.config, 70, 90, premium,
                                 cover=2, recent=True, beta=1)
    assert result["days"] == 20
    assert result["mean_daily_spend"] <= env.budget


def test_selection_does_not_inspect_retrospective_period() -> None:
    config = PortfolioConfig()
    selected, candidates = select_safety_stock(series(), config)
    changed_selected, changed_candidates = select_safety_stock(series(alter_test=True), config)
    assert len(candidates) == 48
    assert candidates == changed_candidates
    assert selected == changed_selected


def test_diagnostic_rejects_evaluation_file_before_loading_it() -> None:
    with pytest.raises(ValueError, match="sales_train_validation.csv"):
        run_safety_stock_diagnostic(Path("sales_train_evaluation.csv"),
                                    Path("unused-model"), Path("unused-report.json"))
