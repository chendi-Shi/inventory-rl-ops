"""Retrospective, validation-tuned safety-stock baseline for the original M5 replay.

This diagnostic deliberately accepts only ``sales_train_validation.csv``. Its
days 1801--1913 have already been inspected in this project, so the result is
not a new holdout claim or a promotion decision.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from inventory_rl.context_artifact import load_context_model
from inventory_rl.m5 import M5Series, load_m5
from inventory_rl.m5_context_actor import replay_context
from inventory_rl.m5_experiment import block_bootstrap_ci, replay
from inventory_rl.portfolio import ORDER_CHOICES, PortfolioConfig, PortfolioEnv, allocate

QUANTILE_GRID = (0.70, 0.85, 0.95)
COVER_GRID = (1, 2, 3, 4)
BETA_GRID = (0.0, 1.0)
TRAINING_HISTORY_DAYS = 365
EXPECTED_SOURCE_DAYS = 1913


def training_safety_premium(data: M5Series, lead: np.ndarray, *, cover: int,
                            quantile: float, history_days: int = TRAINING_HISTORY_DAYS
                            ) -> np.ndarray:
    """Estimate SKU safety units from training-only lead-time sales windows.

    The empirical upper quantile is compared with mean lead-time sales. Only
    the positive difference is added to a dynamic mean-demand target during
    replay. This is a practical quantile approximation, not an optimality
    theorem for shared-budget, multi-period inventory control.
    """
    if type(cover) is not int or cover < 0:
        raise ValueError("cover must be a nonnegative integer")
    if not np.isfinite(quantile) or not 0 < quantile < 1:
        raise ValueError("quantile must be strictly between zero and one")
    if type(history_days) is not int or history_days < 1:
        raise ValueError("history_days must be positive")
    lead = np.asarray(lead)
    if (lead.shape != (data.n_sku,) or not np.isfinite(lead).all() or
            (lead < 1).any() or (lead != np.floor(lead)).any()):
        raise ValueError("invalid lead-time vector")
    start = max(0, data.train_end - history_days)
    training = data.sales[:, start:data.train_end]
    if training.shape[1] < int(lead.max()) + cover:
        raise ValueError("insufficient training history for requested coverage")
    premium = np.empty(data.n_sku, dtype=np.float64)
    for sku, lead_days in enumerate(lead):
        horizon = int(lead_days) + cover
        # Cumulative differences avoid materializing all overlapping windows.
        prefix = np.concatenate(([0.0], np.cumsum(training[sku], dtype=np.float64)))
        sums = prefix[horizon:] - prefix[:-horizon]
        target = float(np.quantile(sums, quantile))
        mean_target = float(training[sku].mean() * horizon)
        premium[sku] = max(0.0, target - mean_target)
    return premium


def safety_stock_scores(env: PortfolioEnv, premium: np.ndarray, *, cover: int,
                        recent: bool, beta: float) -> np.ndarray:
    """Score the same four pack sizes used by the original rule and actor."""
    premium = np.asarray(premium, dtype=np.float64)
    if (premium.shape != (env.data.n_sku,) or not np.isfinite(premium).all() or
            (premium < 0).any()):
        raise ValueError("invalid safety-stock premium")
    if type(cover) is not int or cover < 0:
        raise ValueError("cover must be a nonnegative integer")
    if type(recent) is not bool or not np.isfinite(beta) or beta < 0:
        raise ValueError("invalid score parameters")
    rate = env.last_sales.mean(axis=1) if recent else env.mean_train
    target = rate * (env.lead + cover) + premium
    desired = np.maximum(target - env.stock - env.pipeline.sum(axis=0), 0)
    return -((ORDER_CHOICES[None, :] - desired[:, None]) ** 2) / (rate[:, None] + 1) ** beta


def replay_safety_stock(data: M5Series, config: PortfolioConfig, start: int, end: int,
                        premium: np.ndarray, *, cover: int, recent: bool,
                        beta: float) -> dict:
    """Run the quantile rule through the unchanged simulator and allocator."""
    env = PortfolioEnv(data, config)
    env.reset(start, end)
    daily = []
    demand = sold = spend = 0.0
    while env.day < end:
        scores = safety_stock_scores(env, premium, cover=cover, recent=recent, beta=beta)
        orders = allocate(scores, stock=env.stock, pipeline=env.pipeline.sum(axis=0),
                          budget=env.budget, capacity=env.capacity,
                          unit_cost=config.unit_cost)
        _, reward, _, info = env.step(orders)
        daily.append(float(reward.sum()))
        demand += info["demand"]
        sold += info["sold"]
        spend += info["spend"]
    return {"profit": float(sum(daily)), "daily_profit": daily,
            "fill_rate": float(sold / max(demand, 1)),
            "mean_daily_spend": float(spend / (end - start)),
            "days": end - start, "sku_count": data.n_sku}


def select_safety_stock(data: M5Series, config: PortfolioConfig) -> tuple[dict, list[dict]]:
    """Tune only on days 1701--1800 under the default M5 split."""
    env = PortfolioEnv(data, config)
    premiums = {(quantile, cover): training_safety_premium(
        data, env.lead, cover=cover, quantile=quantile
    ) for quantile in QUANTILE_GRID for cover in COVER_GRID}
    candidates = []
    for quantile in QUANTILE_GRID:
        for cover in COVER_GRID:
            premium = premiums[(quantile, cover)]
            for recent in (False, True):
                for beta in BETA_GRID:
                    validation = replay_safety_stock(
                        data, config, data.train_end, data.validation_end,
                        premium, cover=cover, recent=recent, beta=beta
                    )
                    candidates.append({"quantile": quantile, "cover": cover,
                                       "recent_sales": recent, "beta": beta,
                                       "validation_profit": validation["profit"]})
    selected = max(candidates, key=lambda c: (c["validation_profit"], -c["cover"],
                                              -c["quantile"], -int(c["recent_sales"]),
                                              -c["beta"]))
    return selected, candidates


def run_safety_stock_diagnostic(data_path: Path, model_dir: Path, output: Path) -> dict:
    """Compare three policies on the *already inspected* d_1801--d_1913 period."""
    if data_path.name != "sales_train_validation.csv":
        raise ValueError("diagnostic accepts only sales_train_validation.csv")
    data = load_m5(data_path, store_id="TX_3", sku_count=64,
                   train_end=1700, validation_end=1800)
    if data.sales.shape[1] != EXPECTED_SOURCE_DAYS:
        raise ValueError("diagnostic requires exactly d_1 through d_1913")
    theta, manifest = load_context_model(model_dir)
    if (manifest["source_sha256"] != data.source_sha256 or
            manifest["store_id"] != data.store_id or
            manifest["item_ids"] != list(data.item_ids)):
        raise ValueError("frozen actor does not match the original M5 selection")
    config = PortfolioConfig(**manifest["economics"])
    env = PortfolioEnv(data, config)
    if (not np.allclose(manifest["mean_train"], env.mean_train) or
            not np.array_equal(manifest["lead_time"], env.lead)):
        raise ValueError("frozen actor's features do not match the simulator")

    selected, candidates = select_safety_stock(data, config)
    premium = training_safety_premium(data, env.lead, cover=selected["cover"],
                                      quantile=selected["quantile"])
    start, end = data.validation_end, EXPECTED_SOURCE_DAYS
    safety = replay_safety_stock(data, config, start, end, premium,
                                 cover=selected["cover"],
                                 recent=selected["recent_sales"], beta=selected["beta"])
    baseline = manifest["baseline_selection"]
    actor = replay_context(data, config, start, end, theta,
                           cover=baseline["cover"], recent=baseline["recent_sales"],
                           beta=baseline["beta"])
    rule = replay(data, config, start, end,
                  cover=baseline["cover"], recent=baseline["recent_sales"],
                  baseline_beta=baseline["beta"])
    report = {
        "analysis_status": "post-hoc retrospective baseline diagnostic; no promotion claim",
        "dataset": {"name": "M5 sales_train_validation", "source_sha256": data.source_sha256,
                    "store_id": data.store_id, "sku_count": data.n_sku,
                    "train_days": [1, data.train_end],
                    "validation_days": [data.train_end + 1, data.validation_end],
                    "retrospective_days": [start + 1, end]},
        "economics": asdict(config),
        "safety_stock_selection": selected,
        "safety_stock_candidates": candidates,
        "safety_premium_units": premium.tolist(),
        "frozen_actor_model_sha256": manifest["model_sha256"],
        "frozen_rule_selection": baseline,
        "retrospective": {"safety_stock": safety, "context_actor_rl": actor,
                          "strong_rule": rule,
                          "safety_minus_actor_profit": safety["profit"] - actor["profit"],
                          "safety_minus_rule_profit": safety["profit"] - rule["profit"],
                          "safety_minus_actor_ci95": block_bootstrap_ci(
                              safety["daily_profit"], actor["daily_profit"]),
                          "safety_minus_rule_ci95": block_bootstrap_ci(
                              safety["daily_profit"], rule["daily_profit"])},
        "caveat": ("Observed M5 sales proxy and synthetic economics, including lead times. "
                   "This baseline was designed after d_1801--d_1913 results were known; "
                   "it does not provide fresh out-of-sample evidence."),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/m5/sales_train_validation.csv"))
    parser.add_argument("--model", type=Path, default=Path("models/tx3-context"))
    parser.add_argument("--output", type=Path,
                        default=Path("docs/m5-safety-stock-retro-report.json"))
    args = parser.parse_args()
    report = run_safety_stock_diagnostic(args.data, args.model, args.output)
    results = report["retrospective"]
    print(json.dumps({"report": str(args.output),
                      "selected": report["safety_stock_selection"],
                      "safety_profit": results["safety_stock"]["profit"],
                      "actor_profit": results["context_actor_rl"]["profit"],
                      "rule_profit": results["strong_rule"]["profit"]}, indent=2))


if __name__ == "__main__":
    main()
