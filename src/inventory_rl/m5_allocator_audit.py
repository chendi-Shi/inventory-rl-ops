"""Post-hoc exact-versus-greedy allocation audit of frozen M5 policies.

Both M5 test windows have already been inspected. Exact allocation optimizes
the supplied one-day score, not realized multi-day profit. This diagnostic
never changes a model bundle or the serving allocator.
"""

import json
import os
import platform
from dataclasses import asdict
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from inventory_rl.m5 import M5Series, load_m5
from inventory_rl.m5_experiment import block_bootstrap_ci
from inventory_rl.m5_future_holdout import (
    EVALUATION_MD5,
    TX3_MODEL_SHA256,
    VALIDATION_SHA256,
    WI1_MODEL_SHA256,
    FrozenSpec,
    _digest,
    _load_frozen,
    _verify_frozen,
)
from inventory_rl.portfolio import (
    ORDER_CHOICES,
    PortfolioConfig,
    PortfolioEnv,
    allocate,
    allocate_exact,
    base_stock_scores,
    context_policy_scores,
    policy_search_scores,
)


def _require_close(actual: float, expected: float, label: str) -> None:
    if not np.isclose(actual, expected, rtol=0, atol=1e-7):
        raise ValueError(f"{label} does not reconcile")


def _scores(env: PortfolioEnv, theta: np.ndarray, baseline: dict,
            *, family: str, actor: bool) -> np.ndarray:
    position = env.pipeline.sum(axis=0)
    if actor and family == "policy_search":
        return policy_search_scores(
            env.stock, position, env.last_sales, env.mean_train, env.lead,
            env.day, theta, cover=baseline["cover"], recent=baseline["recent_sales"]
        )
    if actor and family == "context":
        return context_policy_scores(
            env.stock, position, env.last_sales, env.mean_train, env.lead,
            env.day, env.budget, env.config.unit_cost, theta,
            cover=baseline["cover"], recent=baseline["recent_sales"],
            beta=baseline["beta"]
        )
    return base_stock_scores(
        env.stock, position, env.last_sales, env.mean_train, env.lead,
        cover=baseline["cover"], recent=baseline["recent_sales"],
        beta=baseline.get("beta", 1.0)
    )


def _score_surplus(scores: np.ndarray, orders: np.ndarray) -> float:
    actions = np.searchsorted(ORDER_CHOICES, orders)
    return float(np.sum(scores[np.arange(len(orders)), actions] - scores[:, 0]))


def _shadow_decision(env: PortfolioEnv, scores: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray]:
    """Compare both optimizers at one identical state; time allocation only."""
    position = env.pipeline.sum(axis=0)
    args = {"stock": env.stock, "pipeline": position, "budget": env.budget,
            "capacity": env.capacity, "unit_cost": env.config.unit_cost}
    started = perf_counter_ns()
    greedy = allocate(scores, **args)
    greedy_ns = perf_counter_ns() - started
    started = perf_counter_ns()
    exact = allocate_exact(scores, **args)
    exact_ns = perf_counter_ns() - started
    for label, orders in (("greedy", greedy), ("exact", exact)):
        if (orders.shape != (env.data.n_sku,) or
                not np.isin(orders, ORDER_CHOICES).all() or
                orders.sum() * env.config.unit_cost > env.budget + 1e-9 or
                position.sum() + env.stock.sum() + orders.sum() > env.capacity):
            raise ValueError(f"{label} allocation is infeasible")
    greedy_surplus = _score_surplus(scores, greedy)
    exact_surplus = _score_surplus(scores, exact)
    gap = exact_surplus - greedy_surplus
    if gap < -1e-7 * max(1.0, abs(greedy_surplus), abs(exact_surplus)):
        raise ValueError("exact allocator scored below greedy on the same state")
    return {
        "m5_day": env.day + 1,
        "greedy_score_surplus": greedy_surplus,
        "exact_score_surplus": exact_surplus,
        "score_gap": float(max(0.0, gap)),
        "relative_gap_of_exact_positive_surplus": (
            float(max(0.0, gap) / exact_surplus) if exact_surplus > 0 else None
        ),
        "orders_changed": bool(np.any(greedy != exact)),
        "changed_sku_count": int(np.count_nonzero(greedy != exact)),
        "greedy_order_units": int(greedy.sum()),
        "exact_order_units": int(exact.sum()),
        "greedy_budget_slack": float(env.budget - greedy.sum() * env.config.unit_cost),
        "exact_budget_slack": float(env.budget - exact.sum() * env.config.unit_cost),
        "greedy_capacity_slack": int(env.capacity - position.sum() - env.stock.sum() -
                                     greedy.sum()),
        "exact_capacity_slack": int(env.capacity - position.sum() - env.stock.sum() -
                                    exact.sum()),
        "greedy_allocator_ns": int(greedy_ns),
        "exact_allocator_ns": int(exact_ns),
    }, greedy, exact


def _terminal(env: PortfolioEnv) -> dict:
    stock = int(env.stock.sum())
    pipeline = int(env.pipeline.sum())
    return {"on_hand_units": stock, "pipeline_units": pipeline,
            "position_units": stock + pipeline}


def _replay_path(data: M5Series, config: PortfolioConfig, theta: np.ndarray,
                 baseline: dict, *, family: str, actor: bool, allocator: str,
                 start: int, split: int, end: int) -> dict:
    if allocator not in ("greedy", "exact") or not start < split < end:
        raise ValueError("invalid allocation replay protocol")
    env = PortfolioEnv(data, config)
    started = perf_counter_ns()
    env.reset(start, end)
    daily = []
    terminal_old = None
    while env.day < end:
        scores = _scores(env, theta, baseline, family=family, actor=actor)
        shadow, greedy, exact = _shadow_decision(env, scores)
        orders = greedy if allocator == "greedy" else exact
        day = env.day
        _, reward, _, info = env.step(orders)
        daily.append({"m5_day": day + 1, "profit": float(reward.sum()),
                      "demand_units": info["demand"], "sold_units": info["sold"],
                      "spend": info["spend"], "order_units": int(orders.sum()),
                      "shadow": shadow})
        if env.day == split:
            terminal_old = _terminal(env)
    runtime_ns = perf_counter_ns() - started
    if terminal_old is None:
        raise ValueError("missing old-period terminal state")
    return {"allocator": allocator, "daily": daily,
            "old_terminal": terminal_old, "new_terminal": _terminal(env),
            "full_replay_ns": int(runtime_ns)}


def _period(path: dict, begin: int, end: int, *, terminal: dict,
            unit_cost: float) -> dict:
    rows = [row for row in path["daily"] if begin <= row["m5_day"] <= end]
    if len(rows) != end - begin + 1:
        raise ValueError("replay period has missing days")
    profits = [row["profit"] for row in rows]
    shadows = [row["shadow"] for row in rows]
    demands = sum(row["demand_units"] for row in rows)
    greedy_ns = np.asarray([row["greedy_allocator_ns"] for row in shadows])
    exact_ns = np.asarray([row["exact_allocator_ns"] for row in shadows])
    gap = np.asarray([row["score_gap"] for row in shadows])
    return {
        "days": [begin, end], "profit": float(sum(profits)),
        "daily_profit": profits,
        "fill_rate": float(sum(row["sold_units"] for row in rows) / max(demands, 1)),
        "p10_daily_profit": float(np.quantile(profits, 0.1)),
        "mean_daily_spend": float(sum(row["spend"] for row in rows) / len(rows)),
        "terminal": terminal,
        "shadow": {
            "trajectory_allocator": path["allocator"],
            "daily": shadows,
            "total_score_gap": float(gap.sum()),
            "mean_score_gap": float(gap.mean()),
            "median_score_gap": float(np.median(gap)),
            "p95_score_gap": float(np.quantile(gap, 0.95)),
            "max_score_gap": float(gap.max()),
            "changed_decision_count": sum(row["orders_changed"] for row in shadows),
            "changed_decision_fraction": float(
                sum(row["orders_changed"] for row in shadows) / len(shadows)
            ),
            "changed_sku_packs": sum(row["changed_sku_count"] for row in shadows),
            "greedy_budget_binding_decisions": sum(
                row["greedy_budget_slack"] < 4 * unit_cost for row in shadows
            ),
            "exact_budget_binding_decisions": sum(
                row["exact_budget_slack"] < 4 * unit_cost for row in shadows
            ),
            "greedy_capacity_binding_decisions": sum(
                row["greedy_capacity_slack"] < 4 for row in shadows
            ),
            "exact_capacity_binding_decisions": sum(
                row["exact_capacity_slack"] < 4 for row in shadows
            ),
            "timing_ns": {
                "greedy_median": float(np.median(greedy_ns)),
                "greedy_p95": float(np.quantile(greedy_ns, 0.95)),
                "exact_median": float(np.median(exact_ns)),
                "exact_p95": float(np.quantile(exact_ns, 0.95)),
            },
        },
    }


def _paired(a: dict, b: dict, *, left: str, right: str, begin: int) -> dict:
    if len(a["daily_profit"]) != len(b["daily_profit"]):
        raise ValueError("paired paths have different day counts")
    pairs = [{"m5_day": begin + index, "difference": float(x - y)}
             for index, (x, y) in enumerate(zip(a["daily_profit"], b["daily_profit"]))]
    return {"orientation": f"{left}_minus_{right}",
            "profit_difference": float(a["profit"] - b["profit"]),
            "profit_difference_pct_of_right": (
                float(100 * (a["profit"] - b["profit"]) / b["profit"])
                if b["profit"] else None),
            "paired_7day_block_ci95": block_bootstrap_ci(
                a["daily_profit"], b["daily_profit"]),
            "paired_daily_profit": pairs}


def _reconcile_greedy(path: dict, old_report: dict, future_row: dict,
                      *, name: str, future_name: str, start: int, split: int,
                      end: int) -> None:
    old = old_report["test"][name]
    tail = future_row[future_name]
    for label, expected, rows, first in (
        ("published old", old, path["daily"][:split - start], start + 1),
        ("published future", tail, path["daily"][split - start:], split + 1),
    ):
        if len(rows) != len(expected["daily_profit"]):
            raise ValueError(f"{name} {label} daily length mismatch")
        for offset, (actual, published) in enumerate(
            zip(rows, expected["daily_profit"]), start=first
        ):
            _require_close(actual["profit"], published,
                           f"{name} {label} day {offset} profit")
        _require_close(sum(row["profit"] for row in rows), expected["profit"],
                       f"{name} {label} total")
        if label == "published future":
            for metric, key in (("spend", "daily_spend"),
                                ("demand_units", "daily_demand"),
                                ("sold_units", "daily_sold")):
                values = expected[key]
                if len(rows) != len(values):
                    raise ValueError(f"{name} future {key} length mismatch")
                for actual, published in zip(rows, values):
                    _require_close(actual[metric], published,
                                   f"{name} future {metric}")
    if len(path["daily"]) != end - start:
        raise ValueError("full replay day count mismatch")


def _audit_store(spec: FrozenSpec, old: M5Series, new: M5Series,
                 theta: np.ndarray, manifest: dict, published: dict,
                 future_row: dict, *, old_sha: str, start: int, split: int,
                 end: int) -> dict:
    config, baseline = _verify_frozen(
        spec, old, new, theta, manifest, published,
        warmup_start=start, tail_start=split, end=end, old_sha256=old_sha
    )
    if (future_row["store_id"] != spec.store_id or
            future_row["model_sha256"] != spec.model_sha256 or
            future_row["sku_order"] != list(new.item_ids) or
            future_row["baseline_selection"] != baseline or
            future_row["economics"] != asdict(config) or
            future_row["warmup_days_reconciled"] != [start + 1, split] or
            future_row["evaluation_days"] != [split + 1, end]):
        raise ValueError(f"{spec.store_id} published future metadata mismatch")
    rule_name = "strong_rule" if spec.family == "context" else "base_stock"
    periods = {}
    for policy, actor in (("actor", True), ("rule", False)):
        periods[policy] = {}
        for allocator in ("greedy", "exact"):
            path = _replay_path(
                new, config, theta, baseline, family=spec.family,
                actor=actor, allocator=allocator, start=start,
                split=split, end=end
            )
            if allocator == "greedy":
                _reconcile_greedy(
                    path, published, future_row,
                    name="actor" if actor else rule_name,
                    future_name=policy, start=start, split=split, end=end
                )
            periods[policy][allocator] = {
                "old": _period(path, start + 1, split,
                               terminal=path["old_terminal"],
                               unit_cost=config.unit_cost),
                "new": _period(path, split + 1, end,
                               terminal=path["new_terminal"],
                               unit_cost=config.unit_cost),
                "full_replay_ns": path["full_replay_ns"],
            }
    comparisons = {window: {
        "actor_exact_minus_greedy": _paired(
            periods["actor"]["exact"][window], periods["actor"]["greedy"][window],
            left="actor_exact", right="actor_greedy",
            begin=start + 1 if window == "old" else split + 1
        ),
        "rule_exact_minus_greedy": _paired(
            periods["rule"]["exact"][window], periods["rule"]["greedy"][window],
            left="rule_exact", right="rule_greedy",
            begin=start + 1 if window == "old" else split + 1
        ),
        "actor_minus_rule_greedy": _paired(
            periods["actor"]["greedy"][window], periods["rule"]["greedy"][window],
            left="actor_greedy", right="rule_greedy",
            begin=start + 1 if window == "old" else split + 1
        ),
        "actor_minus_rule_exact": _paired(
            periods["actor"]["exact"][window], periods["rule"]["exact"][window],
            left="actor_exact", right="rule_exact",
            begin=start + 1 if window == "old" else split + 1
        ),
    } for window in ("old", "new")}
    return {"store_id": spec.store_id, "family": spec.family,
            "model_sha256": spec.model_sha256,
            "sku_order": list(new.item_ids),
            "baseline_selection": baseline,
            "economics": asdict(config),
            "greedy_reconciled": True,
            "paths": periods,
            "comparisons": comparisons}


def _run_allocator_audit(validation_path: Path, evaluation_path: Path,
                         future_report_path: Path, output: Path,
                         specs: tuple[FrozenSpec, ...], *, expected_old_sha: str,
                         expected_new_md5: str, train_end: int,
                         validation_end: int, start: int, split: int,
                         end: int) -> dict:
    """Configurable protocol engine for synthetic fixtures; never trains."""
    old_sha = _digest(validation_path, "sha256")
    if old_sha != expected_old_sha:
        raise ValueError("old validation CSV SHA-256 mismatch")
    new_md5 = _digest(evaluation_path, "md5")
    if new_md5 != expected_new_md5:
        raise ValueError("official evaluation CSV MD5 mismatch")
    if (not specs or start != validation_end or
            not 28 < train_end < validation_end < split < end or
            split - start < 7 or end - split < 7):
        raise ValueError("invalid allocator audit protocol")
    future = json.loads(future_report_path.read_text(encoding="utf-8"))
    new_sha = _digest(evaluation_path, "sha256")
    if (future["data"]["validation_sha256"] != old_sha or
            future["data"]["evaluation_md5"] != new_md5 or
            future["data"]["evaluation_sha256"] != new_sha or
            future["data"]["warmup_days"] != [start + 1, split] or
            future["data"]["scored_days"] != [split + 1, end] or
            len(future["stores"]) != len(specs)):
        raise ValueError("published future report source or split mismatch")
    indexed = {row["store_id"]: row for row in future["stores"]}
    if len(indexed) != len(specs) or set(indexed) != {spec.store_id for spec in specs}:
        raise ValueError("published future store set mismatch")
    stores = []
    for spec in specs:
        theta, manifest, published = _load_frozen(spec)
        old = load_m5(validation_path, store_id=spec.store_id,
                      sku_count=len(manifest["item_ids"]),
                      train_end=train_end, validation_end=validation_end)
        new = load_m5(evaluation_path, store_id=spec.store_id,
                      sku_count=len(manifest["item_ids"]),
                      train_end=train_end, validation_end=validation_end)
        stores.append(_audit_store(
            spec, old, new, theta, manifest, published, indexed[spec.store_id],
            old_sha=old_sha, start=start, split=split, end=end
        ))
    report = {
        "analysis_status": "post-hoc exact-allocation diagnostic; no new untouched holdout",
        "protocol": "docs/EXACT_ALLOCATOR_PLAN.md",
        "data": {"validation_sha256": old_sha, "evaluation_md5": new_md5,
                 "evaluation_sha256": new_sha, "old_days": [start + 1, split],
                 "new_days": [split + 1, end]},
        "optimizer": "exact multiple-choice one-resource dynamic program on supplied scores",
        "timing_method": ("perf_counter_ns; allocator calls timed separately once per "
                          "decision; full replay includes scoring, both selected and "
                          "shadow allocator calls, and simulator"),
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "os": platform.system(), "machine": platform.machine(),
                    "processor": platform.processor(), "logical_cpus": os.cpu_count(),
                    "allocator_timing_repeats_per_state": 1},
        "stores": stores,
        "caveat": ("All periods were already inspected; score optimality is not "
                   "multi-period profit optimality. M5 sales are a possibly censored "
                   "demand proxy and all economics are simulated. No bundle or "
                   "release decision changes."),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n",
                                        encoding="utf-8")
    return report


def run_allocator_audit(validation_path: Path, evaluation_path: Path,
                        output: Path, *, future_report_path: Path = Path(
                            "docs/m5-future-holdout-report.json")) -> dict:
    """Audit both frozen stores against the published old and new periods."""
    specs = (
        FrozenSpec("WI_1", "policy_search", Path("models/wi1-policy-search"),
                   Path("docs/m5-wi1-policy-search-report.json"), WI1_MODEL_SHA256),
        FrozenSpec("TX_3", "context", Path("models/tx3-context"),
                   Path("docs/m5-tx3-context-report.json"), TX3_MODEL_SHA256),
    )
    return _run_allocator_audit(
        validation_path, evaluation_path, future_report_path, output, specs,
        expected_old_sha=VALIDATION_SHA256, expected_new_md5=EVALUATION_MD5,
        train_end=1700, validation_end=1800, start=1800, split=1913, end=1941
    )
