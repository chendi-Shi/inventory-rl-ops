"""Synthetic-only tests of the predeclared future-period evaluation procedure."""

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
from inventory_rl.m5_future_holdout import FrozenSpec, _run_frozen_pair, run_future_holdout
from inventory_rl.m5_policy_search import replay_actor
from inventory_rl.policy_search_artifact import save_actor_model
from inventory_rl.portfolio import PortfolioConfig


def _write_fixture(path: Path, *, days: int, changed_prefix: bool = False) -> None:
    with path.open("w", newline="", encoding="utf-8") as source:
        writer = csv.writer(source)
        writer.writerow(["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
                        + [f"d_{day}" for day in range(1, days + 1)])
        for store in ("WI_1", "TX_3"):
            for sku in range(2):
                values = [2 + sku + day % 3 + (store == "TX_3")
                          for day in range(days)]
                if changed_prefix and store == "WI_1" and sku == 0:
                    values[570] += 1
                writer.writerow([f"{store}_{sku}_validation", f"item_{sku}",
                                 "FOODS_1", "FOODS", store, store[:2]] + values)


def _hash(path: Path, algorithm: str) -> str:
    return hashlib.new(algorithm, path.read_bytes()).hexdigest()


@pytest.fixture
def frozen_fixture(request: pytest.FixtureRequest) -> dict:
    root = Path("artifacts") / f"_test_future_{request.node.name}"
    root.mkdir(parents=True, exist_ok=True)
    output = root / "future"
    (output / "report.json").unlink(missing_ok=True)
    old_path = root / "sales_train_validation.csv"
    new_path = root / "sales_train_evaluation.csv"
    _write_fixture(old_path, days=600)
    _write_fixture(new_path, days=628)
    config = PortfolioConfig()
    specs = []
    for store, family in (("WI_1", "policy_search"), ("TX_3", "context")):
        data = load_m5(old_path, store_id=store, sku_count=2,
                       train_end=500, validation_end=550)
        model_dir = root / f"{store}-model"
        baseline = {"cover": 2.0, "recent_sales": False}
        if family == "policy_search":
            theta = np.array([0.1, -0.2, 0.1, 0.0, 0.1])
            save_actor_model(theta, data, config, model_dir, True,
                             baseline_cover=2.0, baseline_recent=False)
            manifest_path = model_dir / "actor_manifest.json"
            actor = replay_actor(data, config, 550, 600, theta,
                                 cover=2.0, recent=False)
            rule_name = "base_stock"
            rule = replay(data, config, 550, 600, cover=2.0, recent=False)
        else:
            theta = np.array([0.1, -0.2, 0.1, 0.0, 0.1, 0.0, 0.1, 0.0, 0.1])
            baseline["beta"] = 0.5
            save_context_model(theta, data, config, model_dir, True,
                               baseline_cover=2.0, baseline_recent=False,
                               baseline_beta=0.5)
            manifest_path = model_dir / "context_manifest.json"
            actor = replay_context(data, config, 550, 600, theta,
                                   cover=2.0, recent=False, beta=0.5)
            rule_name = "strong_rule"
            rule = replay(data, config, 550, 600, cover=2.0,
                          recent=False, baseline_beta=0.5)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        published = {
            "dataset": {"source_sha256": data.source_sha256,
                        "store_id": store, "sku_count": data.n_sku,
                        "train_days": [1, 500], "validation_days": [501, 550],
                        "test_days": [551, 600],
                        "selected_item_ids": list(data.item_ids)},
            "economics": asdict(config),
            "baseline_selection": baseline,
            "model_selection": {"selected_theta": theta.tolist()},
            "test": {"actor": actor, rule_name: rule,
                     "active_policy": manifest["active_policy"],
                     "promotion_eligible": manifest["promotion_eligible"]},
        }
        report_path = root / f"{store}-published.json"
        report_path.write_text(json.dumps(published), encoding="utf-8")
        specs.append(FrozenSpec(store, family, model_dir, report_path,
                                manifest["model_sha256"]))
    return {"old_path": old_path, "new_path": new_path,
            "specs": tuple(specs), "output": output}


def _run_fixture(fixture: dict) -> dict:
    return _run_frozen_pair(
        fixture["old_path"], fixture["new_path"], fixture["output"],
        fixture["specs"], expected_old_sha256=_hash(fixture["old_path"], "sha256"),
        expected_new_md5=_hash(fixture["new_path"], "md5"),
        train_end=500, validation_end=550, warmup_start=550,
        tail_start=600, end=628
    )


def test_frozen_pair_reconciles_warmup_and_scores_only_new_days(frozen_fixture: dict):
    report = _run_fixture(frozen_fixture)
    assert [row["store_id"] for row in report["stores"]] == ["WI_1", "TX_3"]
    assert report["data"]["scored_days"] == [601, 628]
    assert report["data"]["evaluation_sha256"] == _hash(
        frozen_fixture["new_path"], "sha256"
    )
    assert (frozen_fixture["output"] / "report.json").exists()
    for row in report["stores"]:
        assert row["warmup_days_reconciled"] == [551, 600]
        assert row["actor"]["days"] == row["rule"]["days"] == 28
        assert len(row["paired_daily_profit"]) == 28
        assert row["paired_daily_profit"][0]["m5_day"] == 601
        assert row["paired_daily_profit"][-1]["m5_day"] == 628
        assert sum(day["difference"] for day in row["paired_daily_profit"]) == pytest.approx(
            row["paired_profit_difference"]
        )
        assert len(row["paired_profit_difference_ci95_7day"]) == 2
        assert row["actor"]["fill_rate"] <= 1
        assert row["actor"]["mean_daily_spend"] <= 2 * 32


def test_future_period_rejects_wrong_official_checksums(frozen_fixture: dict):
    with pytest.raises(ValueError, match="old validation CSV SHA-256 mismatch"):
        run_future_holdout(frozen_fixture["old_path"], frozen_fixture["new_path"],
                           frozen_fixture["output"])
    assert not (frozen_fixture["output"] / "report.json").exists()


def test_future_period_rejects_wrong_evaluation_md5(frozen_fixture: dict):
    with pytest.raises(ValueError, match="official evaluation CSV MD5 mismatch"):
        _run_frozen_pair(
            frozen_fixture["old_path"], frozen_fixture["new_path"],
            frozen_fixture["output"], frozen_fixture["specs"],
            expected_old_sha256=_hash(frozen_fixture["old_path"], "sha256"),
            expected_new_md5="0" * 32, train_end=500, validation_end=550,
            warmup_start=550, tail_start=600, end=628
        )
    assert not (frozen_fixture["output"] / "report.json").exists()


def test_future_period_rejects_changed_history_even_with_matching_file_digest(
    frozen_fixture: dict,
):
    _write_fixture(frozen_fixture["new_path"], days=628, changed_prefix=True)
    with pytest.raises(ValueError, match="sales prefix mismatch"):
        _run_fixture(frozen_fixture)
    assert not (frozen_fixture["output"] / "report.json").exists()


def test_future_period_rejects_changed_model_hash(frozen_fixture: dict):
    first, second = frozen_fixture["specs"]
    frozen_fixture["specs"] = (FrozenSpec(first.store_id, first.family,
                                           first.model_dir, first.report_path,
                                           "0" * 64), second)
    with pytest.raises(ValueError, match="model SHA-256"):
        _run_fixture(frozen_fixture)
    assert not (frozen_fixture["output"] / "report.json").exists()


def test_future_period_rejects_published_warmup_mismatch(frozen_fixture: dict):
    first = frozen_fixture["specs"][0]
    published = json.loads(first.report_path.read_text(encoding="utf-8"))
    published["test"]["actor"]["daily_profit"][0] += 1.0
    first.report_path.write_text(json.dumps(published), encoding="utf-8")
    with pytest.raises(ValueError, match="published warm-up result"):
        _run_fixture(frozen_fixture)
    assert not (frozen_fixture["output"] / "report.json").exists()


def test_future_period_rejects_bundle_sku_order_change(frozen_fixture: dict):
    first = frozen_fixture["specs"][0]
    manifest_path = first.model_dir / "actor_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["item_ids"].reverse()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen SKU order mismatch"):
        _run_fixture(frozen_fixture)
    assert not (frozen_fixture["output"] / "report.json").exists()
