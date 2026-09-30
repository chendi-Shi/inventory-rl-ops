from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")

from inventory_rl.agent import DQNAgent
from inventory_rl.api import (
    PortfolioRequest,
    portfolio_health,
    portfolio_model_bundle,
    recommend_many,
)
from inventory_rl.m5 import M5Series
from inventory_rl.portfolio import PortfolioConfig
from inventory_rl.portfolio_artifact import save_portfolio_model


def test_unpromoted_api_uses_baseline_and_labels_demo_override(monkeypatch):
    directory = Path("artifacts/_test_api_bundle")
    data = M5Series("CA_1", ("item_a", "item_b"), np.ones((2, 40), dtype=np.int32),
                    30, 35, "fixture-sha256")
    save_portfolio_model(DQNAgent(8, 4), data, PortfolioConfig(), directory, False,
                         baseline_recent=False, baseline_cover=2.0)
    monkeypatch.setenv("PORTFOLIO_MODEL_DIR", str(directory))
    monkeypatch.delenv("ALLOW_UNPROMOTED", raising=False)
    portfolio_model_bundle.cache_clear()
    try:
        assert portfolio_health()["active_policy"] == "base_stock"
        request = PortfolioRequest(day=36, stock=[0, 0], pipeline=[[0, 0]] * 3,
                                   last_sales=[[1] * 7, [1] * 7])
        recommendation = recommend_many(request)
        assert recommendation["policy_type"] == "base_stock"
        assert recommendation["spend"] <= 2 * PortfolioConfig().budget_per_sku
        monkeypatch.setenv("ALLOW_UNPROMOTED", "1")
        assert recommend_many(request)["policy_type"] == "rl_unpromoted"
    finally:
        portfolio_model_bundle.cache_clear()
