"""Synthetic-only checks for the frozen TX_3 safety-rule future comparison."""

import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from inventory_rl.context_artifact import save_context_model
from inventory_rl.m5 import load_m5
from inventory_rl.m5_context_actor import replay_context
from inventory_rl.m5_experiment import replay
from inventory_rl.m5_future_holdout import FrozenSpec
from inventory_rl.m5_safety_future import _run_safety_future, run_safety_future
from inventory_rl.m5_safety_stock import replay_safety_stock, training_safety_premium
from inventory_rl.portfolio import PortfolioConfig, PortfolioEnv


def _write_sales(path: Path, *, days: int, changed_prefix: bool = False) -> None:
    with path.open("w", newline="", encoding="utf-8") as source:
        writer = csv.writer(source)
        writer.writerow(["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
                        + [f"d_{day}" for day in range(1, days + 1)])
        for sku in range(2):
            values = [3 + sku + day % 4 for day in range(days)]
            if changed_prefix and sku == 0:
                values[570] += 1
            writer.writerow([f"item_{sku}_TX_3_validation", f"item_{sku}",
                             "FOODS_1", "FOODS", "TX_3", "TX"] + values)


def _hash(path: Path, algorithm: str) -> str:
    return hashlib.new(algorithm, path.read_bytes()).hexdigest()


@pytest.fixture
def safety_fixture(request: pytest.FixtureRequest) -> dict:
    root = Path("artifacts") / f"_test_safety_future_{request.node.name}"
    root.mkdir(parents=True, exist_ok=True)
    output = root / "future"
    (output / "report.json").unlink(missing_ok=True)
    old_path = root / "sales_train_validation.csv"
    new_path = root / "sales_train_evaluation.csv"
    _write_sales(old_path, days=600)
    _write_sales(new_path, days=628)
    data = load_m5(old_path, store_id="TX_3", sku_count=2,
                   train_end=500, validation_end=550)
    config = PortfolioConfig()
    theta = np.array([0.1, -0.2, 0.1, 0.0, 0.1, 0.0, 0.1, 0.0, 0.1])
    baseline = {"cover": 3.0, "recent_sales": True, "beta": 1.0}
    model_dir = root / "context-model"
    save_context_model(theta, data, config, model_dir, True,
                       baseline_cover=3.0, baseline_recent=True,
                       baseline_beta=1.0)
    manifest = json.loads((model_dir / "context_manifest.json").read_text(encoding="utf-8"))
    actor = replay_context(data, config, 550, 600, theta,
                           cover=3.0, recent=True, beta=1.0)
    rule = replay(data, config, 550, 600,
                  cover=3.0, recent=True, baseline_beta=1.0)
    published_path = root / "published.json"
    published_path.write_text(json.dumps({
        "dataset": {"source_sha256": data.source_sha256,
                    "store_id": "TX_3", "sku_count": data.n_sku,
                    "train_days": [1, 500], "validation_days": [501, 550],
                    "test_days": [551, 600], "selected_item_ids": list(data.item_ids)},
        "economics": asdict(config),
        "baseline_selection": baseline,
        "model_selection": {"selected_theta": theta.tolist()},
        "test": {"actor": actor, "strong_rule": rule,
                 "active_policy": "context_actor_rl", "promotion_eligible": True},
    }), encoding="utf-8")
    env = PortfolioEnv(data, config)
    premium = training_safety_premium(data, env.lead, cover=2, quantile=0.7)
    safety = replay_safety_stock(data, config, 550, 600, premium,
                                 cover=2, recent=True, beta=1.0)
    retro_path = root / "retrospective.json"
    retro_path.write_text(json.dumps({
        "dataset": {"source_sha256": data.source_sha256,
                    "store_id": "TX_3", "sku_count": data.n_sku,
                    "train_days": [1, 500], "validation_days": [501, 550],
                    "retrospective_days": [551, 600]},
        "economics": asdict(config),
        "safety_stock_selection": {"quantile": 0.7, "cover": 2,
                                   "recent_sales": True, "beta": 1.0,
                                   "validation_profit": 1.0},
        "safety_premium_units": premium.tolist(),
        "frozen_actor_model_sha256": manifest["model_sha256"],
        "frozen_rule_selection": baseline,
        "retrospective": {"safety_stock": safety,
                          "context_actor_rl": actor, "strong_rule": rule},
    }), encoding="utf-8")
    spec = FrozenSpec("TX_3", "context", model_dir, published_path,
                      manifest["model_sha256"])
    return {"old_path": old_path, "new_path": new_path,
            "retro_path": retro_path, "spec": spec, "output": output}


def _run_fixture(fixture: dict) -> dict:
    return _run_safety_future(
        fixture["old_path"], fixture["new_path"], fixture["output"],
        fixture["spec"], fixture["retro_path"],
        expected_old_sha256=_hash(fixture["old_path"], "sha256"),
        expected_new_md5=_hash(fixture["new_path"], "md5"),
        train_end=500, validation_end=550,
        warmup_start=550, tail_start=600, end=628
    )


def test_safety_comparison_reconciles_three_warmups_and_scores_only_future(
    safety_fixture: dict,
):
    report = _run_fixture(safety_fixture)
    assert report["data"]["scored_days"] == [601, 628]
    assert report["data"]["evaluation_sha256"] == _hash(
        safety_fixture["new_path"], "sha256"
    )
    assert len(report["paired_daily_profit"]) == 28
    assert report["paired_daily_profit"][0]["m5_day"] == 601
    assert report["paired_daily_profit"][-1]["m5_day"] == 628
    for name in ("safety_stock", "context_actor_rl", "strong_rule"):
        metrics = report[name]
        assert metrics["days"] == 28
        assert len(metrics["daily_profit"]) == 28
        assert metrics["mean_daily_spend"] <= 2 * 32
        assert 0 <= metrics["fill_rate"] <= 1
    for comparison in report["comparisons"].values():
        assert len(comparison["paired_profit_difference_ci95_7day"]) == 2
    assert sum(x["safety_minus_actor"] for x in report["paired_daily_profit"]) == pytest.approx(
        report["comparisons"]["safety_minus_actor"]["simulated_profit_difference"]
    )
    assert (safety_fixture["output"] / "report.json").exists()


def test_safety_comparison_requires_official_hashes(safety_fixture: dict):
    with pytest.raises(ValueError, match="old validation CSV SHA-256 mismatch"):
        run_safety_future(safety_fixture["old_path"], safety_fixture["new_path"],
                          safety_fixture["output"])
    assert not (safety_fixture["output"] / "report.json").exists()


def test_safety_comparison_rejects_changed_history(safety_fixture: dict):
    _write_sales(safety_fixture["new_path"], days=628, changed_prefix=True)
    with pytest.raises(ValueError, match="sales prefix mismatch"):
        _run_fixture(safety_fixture)
    assert not (safety_fixture["output"] / "report.json").exists()


def test_safety_comparison_rejects_changed_training_premium(safety_fixture: dict):
    retro = json.loads(safety_fixture["retro_path"].read_text(encoding="utf-8"))
    retro["safety_premium_units"][0] += 1.0
    safety_fixture["retro_path"].write_text(json.dumps(retro), encoding="utf-8")
    with pytest.raises(ValueError, match="safety premium differs"):
        _run_fixture(safety_fixture)
    assert not (safety_fixture["output"] / "report.json").exists()


def test_safety_comparison_rejects_changed_warmup_profit(safety_fixture: dict):
    retro = json.loads(safety_fixture["retro_path"].read_text(encoding="utf-8"))
    retro["retrospective"]["safety_stock"]["daily_profit"][0] += 1.0
    safety_fixture["retro_path"].write_text(json.dumps(retro), encoding="utf-8")
    with pytest.raises(ValueError, match="published warm-up result"):
        _run_fixture(safety_fixture)
    assert not (safety_fixture["output"] / "report.json").exists()


def test_safety_comparison_rejects_reselected_parameters(safety_fixture: dict):
    retro = json.loads(safety_fixture["retro_path"].read_text(encoding="utf-8"))
    retro["safety_stock_selection"]["cover"] = 3
    safety_fixture["retro_path"].write_text(json.dumps(retro), encoding="utf-8")
    with pytest.raises(ValueError, match="predeclared validation selection"):
        _run_fixture(safety_fixture)
    assert not (safety_fixture["output"] / "report.json").exists()
