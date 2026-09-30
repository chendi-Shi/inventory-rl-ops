import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")

from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError

from inventory_rl.agent import DQNAgent
from inventory_rl.api import (
    ContextPortfolioRequest,
    PortfolioRequest,
    app,
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
from inventory_rl.context_artifact import (
    load_context_model,
    recommend_context,
    save_context_model,
)
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
        request = ContextPortfolioRequest(day=36, stock=[0, 0], pipeline=[[0, 0]] * 3,
                                          last_sales=[[1] * 7, [1] * 7],
                                          item_ids=list(data.item_ids))
        recommendation = recommend_context_actor(request)
        assert recommendation["policy_type"] == "context_actor_rl"
        assert recommendation["spend"] <= 2 * PortfolioConfig().budget_per_sku
        assert set(recommendation["orders"]) == {"item_a", "item_b"}
        assert recommendation["manifest_sha256"] == context_health()["manifest_sha256"]
        assert recommendation["sku_order_sha256"] == context_health()["sku_order_sha256"]
        assert recommendation["source_sha256"] == data.source_sha256
        assert recommendation["day"] == request.day
        assert recommendation["sku_order_verified"] is True
        assert context_health()["item_ids"] == list(data.item_ids)
    finally:
        context_model_bundle.cache_clear()


def test_context_request_checks_sku_order_and_integer_state(monkeypatch):
    directory = Path("artifacts/_test_context_order_bundle")
    data = M5Series("TX_3", ("item_a", "item_b"), np.ones((2, 40), dtype=np.int32),
                    30, 35, "fixture-sha256")
    save_context_model(np.zeros(9), data, PortfolioConfig(), directory, True,
                       baseline_cover=2.0, baseline_recent=False, baseline_beta=1.0)
    theta, manifest = load_context_model(directory)
    monkeypatch.setenv("CONTEXT_MODEL_DIR", str(directory))
    context_model_bundle.cache_clear()
    try:
        request = ContextPortfolioRequest(
            day=36, stock=[0, 0], pipeline=[[0, 0]] * 3,
            last_sales=[[1] * 7, [1] * 7],
            item_ids=["item_a", "item_b"],
            sku_order_sha256=manifest["sku_order_sha256"],
        )
        accepted = recommend_context_actor(request)
        assert accepted["policy_type"] == "context_actor_rl"
        assert accepted["sku_order_verified"] is True
        with TestClient(app) as client:
            response = client.post("/v4/portfolio/recommendations", json=request.model_dump())
            assert response.status_code == 200
            assert response.json()["sku_order_verified"] is True
            malformed = request.model_dump()
            malformed["stock"] = [0.5, 0]
            assert client.post("/v4/portfolio/recommendations", json=malformed).status_code == 422
            unlabeled = request.model_dump()
            unlabeled.pop("item_ids")
            unlabeled.pop("sku_order_sha256")
            assert client.post("/v4/portfolio/recommendations", json=unlabeled).status_code == 422
        request.item_ids = ["item_b", "item_a"]
        with pytest.raises(HTTPException) as exc:
            recommend_context_actor(request)
        assert exc.value.status_code == 422
        request.item_ids = ["item_a", "item_b"]
        request.sku_order_sha256 = "wrong"
        with pytest.raises(HTTPException) as exc:
            recommend_context_actor(request)
        assert exc.value.status_code == 422
        with pytest.raises(ValidationError):
            ContextPortfolioRequest(day=36, stock=[0.5, 0], pipeline=[[0, 0]] * 3,
                                    last_sales=[[1] * 7, [1] * 7],
                                    item_ids=["item_a", "item_b"])
        unlabeled_request = ContextPortfolioRequest(
            day=36, stock=[0, 0], pipeline=[[0, 0]] * 3,
            last_sales=[[1] * 7, [1] * 7],
        )
        with pytest.raises(HTTPException) as exc:
            recommend_context_actor(unlabeled_request)
        assert exc.value.status_code == 422
        with pytest.raises(ValueError, match="item_ids or sku_order_sha256"):
            recommend_context(theta, manifest, day=36, stock=np.zeros(2),
                              pipeline=np.zeros((3, 2)), last_sales=np.ones((2, 7)))
        with pytest.raises(ValueError, match="integer quantities"):
            recommend_context(theta, manifest, day=36, stock=np.array([0.5, 0]),
                              pipeline=np.zeros((3, 2)), last_sales=np.ones((2, 7)),
                              item_ids=["item_a", "item_b"])
        with pytest.raises(ValueError, match="capacity"):
            recommend_context(theta, manifest, day=36, stock=np.array([101, 0]),
                              pipeline=np.zeros((3, 2)), last_sales=np.ones((2, 7)),
                              item_ids=["item_a", "item_b"])
    finally:
        context_model_bundle.cache_clear()


@pytest.mark.parametrize("field,value,error", [
    ("mean_train", [float("nan"), 1.0], "training means"),
    ("lead_time", [1, 4], "lead times"),
    ("economics", {"budget_per_sku": 32.0, "capacity_per_sku": 50,
                   "unit_price": 10.0, "unit_cost": float("nan"),
                   "holding_cost": 0.15, "lost_sale_penalty": 2.0}, "economics"),
    ("promotion_eligible", "false", "promotion decision"),
])
def test_context_loader_rejects_invalid_manifest(field, value, error):
    directory = Path(f"artifacts/_test_context_invalid_{field}")
    data = M5Series("TX_3", ("item_a", "item_b"), np.ones((2, 40), dtype=np.int32),
                    30, 35, "fixture-sha256")
    save_context_model(np.zeros(9), data, PortfolioConfig(), directory, True,
                       baseline_cover=2.0, baseline_recent=False, baseline_beta=1.0)
    manifest_file = directory / "context_manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest[field] = value
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        load_context_model(directory)


def test_context_bundle_rejects_invalid_promotion_flag():
    directory = Path("artifacts/_test_context_invalid_release")
    data = M5Series("TX_3", ("item_a", "item_b"), np.ones((2, 40), dtype=np.int32),
                    30, 35, "fixture-sha256")
    with pytest.raises(ValueError, match="promotion_eligible"):
        save_context_model(np.zeros(9), data, PortfolioConfig(), directory, "false",
                           baseline_cover=2.0, baseline_recent=False, baseline_beta=1.0)


def test_context_api_reports_missing_model_as_unavailable(monkeypatch):
    monkeypatch.setenv("CONTEXT_MODEL_DIR", "artifacts/_missing_context_bundle")
    context_model_bundle.cache_clear()
    request = ContextPortfolioRequest(day=36, stock=[0, 0], pipeline=[[0, 0]] * 3,
                                      last_sales=[[1] * 7, [1] * 7],
                                      item_ids=["item_a", "item_b"])
    try:
        with pytest.raises(HTTPException) as exc:
            context_health()
        assert exc.value.status_code == 503
        with pytest.raises(HTTPException) as exc:
            recommend_context_actor(request)
        assert exc.value.status_code == 503
    finally:
        context_model_bundle.cache_clear()
