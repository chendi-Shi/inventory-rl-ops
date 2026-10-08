"""Synthetic checks of the retrospective allocator audit; no M5 download."""

import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from inventory_rl.m5 import load_m5
from inventory_rl.m5_allocator_audit import _run_allocator_audit, _shadow_decision
from inventory_rl.m5_experiment import replay
from inventory_rl.m5_future_holdout import FrozenSpec, _run_frozen_pair
from inventory_rl.m5_policy_search import replay_actor
from inventory_rl.policy_search_artifact import save_actor_model
from inventory_rl.portfolio import PortfolioConfig, PortfolioEnv


def _csv(path: Path, days: int) -> None:
    with path.open("w", newline="", encoding="utf-8") as source:
        writer = csv.writer(source)
        writer.writerow(["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
                        + [f"d_{day}" for day in range(1, days + 1)])
        for sku in range(2):
            writer.writerow([f"WI_1_{sku}_validation", f"item_{sku}", "FOODS_1",
                             "FOODS", "WI_1", "WI"] +
                            [2 + sku + day % 3 for day in range(days)])


def _hash(path: Path, algorithm: str) -> str:
    return hashlib.new(algorithm, path.read_bytes()).hexdigest()


@pytest.fixture
def audit_fixture(request: pytest.FixtureRequest) -> dict:
    root = Path("artifacts") / f"_test_allocator_audit_{request.node.name}"
    root.mkdir(parents=True, exist_ok=True)
    (root / "audit" / "report.json").unlink(missing_ok=True)
    old_path = root / "sales_train_validation.csv"
    new_path = root / "sales_train_evaluation.csv"
    _csv(old_path, 600)
    _csv(new_path, 628)
    old = load_m5(old_path, store_id="WI_1", sku_count=2,
                  train_end=500, validation_end=550)
    config = PortfolioConfig()
    theta = np.array([0.1, -0.2, 0.1, 0.0, 0.1])
    model_dir = root / "model"
    save_actor_model(theta, old, config, model_dir, True,
                     baseline_cover=2.0, baseline_recent=False)
    manifest = json.loads((model_dir / "actor_manifest.json").read_text(encoding="utf-8"))
    actor = replay_actor(old, config, 550, 600, theta, cover=2.0, recent=False)
    rule = replay(old, config, 550, 600, cover=2.0, recent=False)
    published = {
        "dataset": {"source_sha256": old.source_sha256, "store_id": "WI_1",
                    "sku_count": old.n_sku, "train_days": [1, 500],
                    "validation_days": [501, 550], "test_days": [551, 600],
                    "selected_item_ids": list(old.item_ids)},
        "economics": asdict(config),
        "baseline_selection": {"cover": 2.0, "recent_sales": False},
        "model_selection": {"selected_theta": theta.tolist()},
        "test": {"actor": actor, "base_stock": rule,
                 "active_policy": manifest["active_policy"],
                 "promotion_eligible": manifest["promotion_eligible"]},
    }
    report_path = root / "old-report.json"
    report_path.write_text(json.dumps(published), encoding="utf-8")
    spec = FrozenSpec("WI_1", "policy_search", model_dir, report_path,
                      manifest["model_sha256"])
    future_path = root / "future" / "report.json"
    _run_frozen_pair(
        old_path, new_path, future_path.parent, (spec,),
        expected_old_sha256=_hash(old_path, "sha256"),
        expected_new_md5=_hash(new_path, "md5"),
        train_end=500, validation_end=550, warmup_start=550,
        tail_start=600, end=628,
    )
    return {"old": old_path, "new": new_path, "spec": spec,
            "future": future_path, "output": root / "audit"}


def _audit(fixture: dict) -> dict:
    return _run_allocator_audit(
        fixture["old"], fixture["new"], fixture["future"], fixture["output"],
        (fixture["spec"],), expected_old_sha=_hash(fixture["old"], "sha256"),
        expected_new_md5=_hash(fixture["new"], "md5"),
        train_end=500, validation_end=550, start=550, split=600, end=628,
    )


def test_shadow_decision_finds_greedy_score_gap(audit_fixture: dict):
    data = load_m5(audit_fixture["old"], store_id="WI_1", sku_count=2,
                   train_end=500, validation_end=550)
    env = PortfolioEnv(data)
    env.reset(550, 600)
    scores = np.array([[0.0, 7.333, 13.333, 21.333],
                       [0.0, 4.5, 4.6, 4.7]])
    shadow, greedy, exact = _shadow_decision(env, scores)
    assert shadow["orders_changed"] is True
    assert shadow["score_gap"] > 0
    assert shadow["changed_sku_count"] > 0
    assert greedy.sum() * env.config.unit_cost <= env.budget
    assert exact.sum() * env.config.unit_cost <= env.budget
    assert shadow["exact_allocator_ns"] >= 0


def test_allocator_audit_reconciles_controls_and_reports_both_paths(
    audit_fixture: dict,
):
    report = _audit(audit_fixture)
    assert report["analysis_status"].startswith("post-hoc")
    assert (audit_fixture["output"] / "report.json").exists()
    store = report["stores"][0]
    assert store["greedy_reconciled"] is True
    future = json.loads(audit_fixture["future"].read_text(encoding="utf-8"))["stores"][0]
    assert store["paths"]["actor"]["greedy"]["new"]["daily_profit"] == pytest.approx(
        future["actor"]["daily_profit"]
    )
    for policy in ("actor", "rule"):
        for allocator in ("greedy", "exact"):
            path = store["paths"][policy][allocator]
            assert path["old"]["days"] == [551, 600]
            assert path["new"]["days"] == [601, 628]
            assert len(path["old"]["shadow"]["daily"]) == 50
            assert len(path["new"]["shadow"]["daily"]) == 28
            assert path["full_replay_ns"] > 0
            assert min(row["score_gap"] for row in path["new"]["shadow"]["daily"]) >= 0
            assert path["new"]["terminal"]["position_units"] >= 0
            assert 0 <= path["new"]["shadow"]["changed_decision_fraction"] <= 1
            assert path["new"]["shadow"]["median_score_gap"] >= 0
    assert len(store["comparisons"]["new"]["actor_exact_minus_greedy"][
        "paired_daily_profit"]) == 28
    assert "actor_minus_rule_exact" in store["comparisons"]["old"]


def test_allocator_audit_rejects_future_daily_mismatch(audit_fixture: dict):
    future = json.loads(audit_fixture["future"].read_text(encoding="utf-8"))
    future["stores"][0]["actor"]["daily_profit"][0] += 1
    audit_fixture["future"].write_text(json.dumps(future), encoding="utf-8")
    with pytest.raises(ValueError, match="published future day 601 profit does not reconcile"):
        _audit(audit_fixture)
    assert not (audit_fixture["output"] / "report.json").exists()


def test_allocator_audit_rejects_wrong_evaluation_digest(audit_fixture: dict):
    with pytest.raises(ValueError, match="official evaluation CSV MD5 mismatch"):
        _run_allocator_audit(
            audit_fixture["old"], audit_fixture["new"], audit_fixture["future"],
            audit_fixture["output"], (audit_fixture["spec"],),
            expected_old_sha=_hash(audit_fixture["old"], "sha256"),
            expected_new_md5="0" * 32, train_end=500, validation_end=550,
            start=550, split=600, end=628,
        )
    assert not (audit_fixture["output"] / "report.json").exists()
