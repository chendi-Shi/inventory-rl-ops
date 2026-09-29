"""Optional FastAPI decision service. Set MODEL_DIR to a reviewed artifact."""

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from inventory_rl.artifact import load_model
from inventory_rl.env import InventoryEnv
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


app = FastAPI(title="Inventory RL Decision Service", version="0.1.0")


@lru_cache(maxsize=1)
def model_bundle():
    return load_model(Path(os.environ.get("MODEL_DIR", "artifacts/run-42")))


@app.get("/health")
def health() -> dict:
    try:
        _, _, manifest = model_bundle()
    except (OSError, ValueError, KeyError) as exc:
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
    except (OSError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@lru_cache(maxsize=1)
def portfolio_model_bundle():
    return load_portfolio_model(Path(os.environ.get("PORTFOLIO_MODEL_DIR", "artifacts/m5-ca1")))


@app.get("/v2/health")
def portfolio_health() -> dict:
    try:
        _, manifest = portfolio_model_bundle()
    except (OSError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"status": "ready", "model_sha256": manifest["model_sha256"],
            "promotion_eligible": manifest["promotion_eligible"],
            "sku_count": len(manifest["item_ids"])}


@app.post("/v2/portfolio/recommendations")
def recommend_many(request: PortfolioRequest) -> dict:
    try:
        agent, manifest = portfolio_model_bundle()
        if not manifest["promotion_eligible"] and os.environ.get("ALLOW_UNPROMOTED") != "1":
            raise HTTPException(status_code=409, detail="model failed offline promotion gate")
        return recommend_portfolio(
            agent, manifest, day=request.day, stock=np.asarray(request.stock, dtype=float),
            pipeline=np.asarray(request.pipeline, dtype=float),
            last_sales=np.asarray(request.last_sales, dtype=float),
        )
    except (OSError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
