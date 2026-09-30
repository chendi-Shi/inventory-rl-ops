"""Optional FastAPI decision service. Set MODEL_DIR to a reviewed artifact."""

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, StrictInt

from inventory_rl.artifact import load_model
from inventory_rl.context_artifact import load_context_model, recommend_context
from inventory_rl.env import InventoryEnv
from inventory_rl.policy_search_artifact import load_actor_model, recommend_actor
from inventory_rl.portfolio_artifact import load_portfolio_model, recommend_portfolio


class DecisionRequest(BaseModel):
    day: int = Field(ge=0)
    stock: tuple[int, int, int]
    pipeline: list[tuple[int, int, int]]
    last_demand: tuple[float, float, float]


class PortfolioRequest(BaseModel):
    day: int = Field(ge=0)
    stock: list[int]
    pipeline: list[list[int]]
    last_sales: list[list[int]]


class ContextPortfolioRequest(PortfolioRequest):
    """v4 input requires proof of the positional SKU order."""

    day: StrictInt = Field(ge=0)
    stock: list[StrictInt]
    pipeline: list[list[StrictInt]]
    last_sales: list[list[StrictInt]]
    item_ids: list[str] | None = None
    sku_order_sha256: str | None = None


app = FastAPI(title="Inventory RL Decision Service", version="0.1.0")


@lru_cache(maxsize=1)
def model_bundle():
    return load_model(Path(os.environ.get("MODEL_DIR", "artifacts/run-42")))


@app.get("/health")
def health() -> dict:
    try:
        _, _, manifest = model_bundle()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"status": "ready", "model_sha256": manifest["model_sha256"]}


@app.post("/v1/recommendations")
def recommend(request: DecisionRequest) -> dict:
    try:
        agent, config, manifest = model_bundle()
        if len(request.pipeline) != max(config.lead_time):
            raise ValueError("pipeline length must match maximum lead time")
        if request.day >= config.horizon:
            raise ValueError("day is beyond the planning horizon")
        if any(x < 0 for x in request.stock) or any(x < 0 for row in request.pipeline for x in row):
            raise ValueError("stock and pipeline must be nonnegative")
        if any(x < 0 or not np.isfinite(x) for x in request.last_demand):
            raise ValueError("last demand must be finite and nonnegative")
        env = InventoryEnv(config)
        env.day = request.day
        env.stock = np.asarray(request.stock, dtype=np.int64)
        env.pipeline = np.asarray(request.pipeline, dtype=np.int64)
        env.last_demand = np.asarray(request.last_demand, dtype=np.float64)
        if int(env.stock.sum() + env.pipeline.sum()) > config.storage_capacity:
            raise ValueError("current inventory exceeds storage capacity")
        action_id = agent.act(env.observation(), env.valid_actions())
        order = env.actions[action_id]
        return {"action_id": action_id, "order": order.tolist(),
                "spend": float(order @ np.asarray(config.unit_cost)),
                "model_sha256": manifest["model_sha256"]}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@lru_cache(maxsize=1)
def portfolio_model_bundle():
    return load_portfolio_model(Path(os.environ.get("PORTFOLIO_MODEL_DIR", "artifacts/m5-ca1")))


@app.get("/v2/health")
def portfolio_health() -> dict:
    try:
        _, manifest = portfolio_model_bundle()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"status": "ready", "model_sha256": manifest["model_sha256"],
            "promotion_eligible": manifest["promotion_eligible"],
            "active_policy": manifest["active_policy"],
            "sku_count": len(manifest["item_ids"])}


@app.post("/v2/portfolio/recommendations")
def recommend_many(request: PortfolioRequest) -> dict:
    try:
        agent, manifest = portfolio_model_bundle()
        return recommend_portfolio(
            agent, manifest, day=request.day, stock=np.asarray(request.stock, dtype=float),
            pipeline=np.asarray(request.pipeline, dtype=float),
            last_sales=np.asarray(request.last_sales, dtype=float),
            force_rl=os.environ.get("ALLOW_UNPROMOTED") == "1",
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@lru_cache(maxsize=1)
def policy_search_bundle():
    return load_actor_model(Path(os.environ.get(
        "POLICY_SEARCH_MODEL_DIR", "models/wi1-policy-search"
    )))


@app.get("/v3/health")
def policy_search_health() -> dict:
    try:
        _, manifest = policy_search_bundle()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"status": "ready", "model_sha256": manifest["model_sha256"],
            "promotion_eligible": manifest["promotion_eligible"],
            "active_policy": manifest["active_policy"],
            "sku_count": len(manifest["item_ids"])}


@app.post("/v3/portfolio/recommendations")
def recommend_policy_search(request: PortfolioRequest) -> dict:
    try:
        theta, manifest = policy_search_bundle()
        return recommend_actor(
            theta, manifest, day=request.day,
            stock=np.asarray(request.stock, dtype=float),
            pipeline=np.asarray(request.pipeline, dtype=float),
            last_sales=np.asarray(request.last_sales, dtype=float),
            force_actor=os.environ.get("ALLOW_UNPROMOTED") == "1",
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@lru_cache(maxsize=1)
def context_model_bundle():
    return load_context_model(Path(os.environ.get(
        "CONTEXT_MODEL_DIR", "models/tx3-context"
    )))


@app.get("/v4/health")
def context_health() -> dict:
    try:
        _, manifest = context_model_bundle()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"status": "ready", "model_sha256": manifest["model_sha256"],
            "manifest_sha256": manifest["manifest_sha256"],
            "source_sha256": manifest["source_sha256"],
            "store_id": manifest["store_id"],
            "sku_order_sha256": manifest["sku_order_sha256"],
            "promotion_eligible": manifest["promotion_eligible"],
            "active_policy": manifest["active_policy"],
            "sku_count": len(manifest["item_ids"]),
            "item_ids": manifest["item_ids"]}


@app.post("/v4/portfolio/recommendations")
def recommend_context_actor(request: ContextPortfolioRequest) -> dict:
    try:
        theta, manifest = context_model_bundle()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    try:
        return recommend_context(
            theta, manifest, day=request.day,
            stock=np.asarray(request.stock),
            pipeline=np.asarray(request.pipeline),
            last_sales=np.asarray(request.last_sales),
            force_actor=os.environ.get("ALLOW_UNPROMOTED") == "1",
            item_ids=getattr(request, "item_ids", None),
            sku_order_sha256=getattr(request, "sku_order_sha256", None),
        )
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
