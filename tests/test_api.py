from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")

from inventory_rl.agent import DQNAgent
from inventory_rl.api import (
    PortfolioRequest,
    context_health,
    context_model_bundle,
    policy_search_bundle,
    policy_search_health,
    portfolio_health,
    portfolio_model_bundle,
    recommend_context_actor,
    recommend_many,
    recommend_policy_search,
)
from inventory_rl.context_artifact import load_context_model, save_context_model
from inventory_rl.m5 import M5Series
from inventory_rl.policy_search_artifact import load_actor_model, save_actor_model
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
        hybrid_directory = Path("artifacts/_test_api_hybrid_bundle")
        save_portfolio_model(DQNAgent(8, 4), data, PortfolioConfig(), hybrid_directory, False,
                             baseline_recent=False, baseline_cover=2.0, hybrid_alpha=0.25)
        monkeypatch.setenv("PORTFOLIO_MODEL_DIR", str(hybrid_directory))
        monkeypatch.delenv("ALLOW_UNPROMOTED")
        portfolio_model_bundle.cache_clear()
        assert portfolio_health()["active_policy"] == "base_stock"
        assert recommend_many(request)["policy_type"] == "base_stock"
        monkeypatch.setenv("ALLOW_UNPROMOTED", "1")
        assert recommend_many(request)["policy_type"] == "hybrid_unpromoted"
    finally:
        portfolio_model_bundle.cache_clear()


def test_promoted_policy_search_bundle_serves_feasible_actor(monkeypatch):
    directory = Path("artifacts/_test_actor_api_bundle")
    data = M5Series("WI_1", ("item_a", "item_b"), np.ones((2, 40), dtype=np.int32),
                    30, 35, "fixture-sha256")
    theta = np.array([0.5, 0.2, 0.1, 0.0, -0.1])
    save_actor_model(theta, data, PortfolioConfig(), directory, True,
                     baseline_cover=2.0, baseline_recent=False)
    loaded, manifest = load_actor_model(directory)
    np.testing.assert_array_equal(loaded, theta)
    assert manifest["active_policy"] == "policy_search_rl"
    monkeypatch.setenv("POLICY_SEARCH_MODEL_DIR", str(directory))
    policy_search_bundle.cache_clear()
    try:
        assert policy_search_health()["active_policy"] == "policy_search_rl"
        request = PortfolioRequest(day=36, stock=[0, 0], pipeline=[[0, 0]] * 3,
                                   last_sales=[[1] * 7, [1] * 7])
        recommendation = recommend_policy_search(request)
        assert recommendation["policy_type"] == "policy_search_rl"
        assert recommendation["spend"] <= 2 * PortfolioConfig().budget_per_sku
        assert set(recommendation["orders"]) == {"item_a", "item_b"}
    finally:
        policy_search_bundle.cache_clear()


def test_promoted_context_bundle_serves_feasible_actor(monkeypatch):
    directory = Path("artifacts/_test_context_api_bundle")
    data = M5Series("TX_3", ("item_a", "item_b"), np.ones((2, 40), dtype=np.int32),
                    30, 35, "fixture-sha256")
    theta = np.linspace(-0.2, 0.2, 9)
    save_context_model(theta, data, PortfolioConfig(), directory, True,
                       baseline_cover=2.0, baseline_recent=False, baseline_beta=1.0)
    loaded, manifest = load_context_model(directory)
    np.testing.assert_array_equal(loaded, theta)
    assert manifest["active_policy"] == "context_actor_rl"
    monkeypatch.setenv("CONTEXT_MODEL_DIR", str(directory))
    context_model_bundle.cache_clear()
    try:
        assert context_health()["active_policy"] == "context_actor_rl"
        request = PortfolioRequest(day=36, stock=[0, 0], pipeline=[[0, 0]] * 3,
                                   last_sales=[[1] * 7, [1] * 7])
        recommendation = recommend_context_actor(request)
        assert recommendation["policy_type"] == "context_actor_rl"
        assert recommendation["spend"] <= 2 * PortfolioConfig().budget_per_sku
        assert set(recommendation["orders"]) == {"item_a", "item_b"}
    finally:
        context_model_bundle.cache_clear()
