import csv
from pathlib import Path

import numpy as np
import pytest

from inventory_rl.agent import DQNAgent
from inventory_rl.m5 import load_m5
from inventory_rl.m5_experiment import block_bootstrap_ci, replay, run
from inventory_rl.m5_guided import run_guided
from inventory_rl.m5_hybrid import run_hybrid
from inventory_rl.portfolio import PortfolioEnv, allocate, residual_scores
from inventory_rl.portfolio_artifact import load_portfolio_model, recommend_portfolio


def write_m5_fixture(path: Path, days: int = 600) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
                        + [f"d_{day}" for day in range(1, days + 1)])
        for sku in range(9):
            sales = [2 + sku + day % 3 for day in range(days)]
            if sku == 8:
                sales[:500] = [1] * 500
                sales[500:] = [200] * (days - 500)
            writer.writerow([f"item_{sku}_CA_1_validation", f"item_{sku}", "FOODS_1",
                             "FOODS", "CA_1", "CA"] + sales)
        writer.writerow(["other_TX_1_validation", "other", "FOODS_1", "FOODS",
                         "TX_1", "TX"] + [999] * days)


def test_selection_uses_training_period_only():
    path = Path("artifacts/_test_m5.csv")
    write_m5_fixture(path)
    data = load_m5(path, sku_count=8, train_end=500, validation_end=550)
    assert data.sales.shape == (8, 600)
    assert "item_8" not in data.item_ids  # future spike must not affect selection
    assert "other" not in data.item_ids


def test_allocator_and_replay_constraints():
    path = Path("artifacts/_test_m5.csv")
    write_m5_fixture(path)
    data = load_m5(path, sku_count=8, train_end=500, validation_end=550)
    env = PortfolioEnv(data)
    env.reset(500, 520)
    scores = np.tile([0.0, 1.0, 2.0, 3.0], (8, 1))
    while env.day < env.end:
        orders = allocate(scores, stock=env.stock, pipeline=env.pipeline.sum(axis=0),
                          budget=env.budget, capacity=env.capacity,
                          unit_cost=env.config.unit_cost)
        assert orders.sum() * env.config.unit_cost <= env.budget
        assert env.stock.sum() + env.pipeline.sum() + orders.sum() <= env.capacity
        _, reward, _, info = env.step(orders)
        assert reward.shape == (8,)
        assert info["sold"] <= info["demand"]
    with pytest.raises(RuntimeError):
        env.step(np.zeros(8, dtype=np.int32))


def test_end_to_end_m5_pipeline_and_block_ci():
    path = Path("artifacts/_test_m5.csv")
    write_m5_fixture(path)
    report = run(path, Path("artifacts/_test_m5_run"), sku_count=8, seeds=(1,),
                 episodes=1, train_end=500, validation_end=550)
    assert report["dataset"]["test_days"] == [551, 600]
    assert report["model_selection"]["selected_seed"] == 1
    assert report["test"]["rl"]["days"] == 50
    assert len(report["test"]["paired_uplift_block_bootstrap_ci95"]) == 2
    assert block_bootstrap_ci([1.0] * 21, [0.0] * 21) == pytest.approx([21.0, 21.0])
    agent, manifest = load_portfolio_model(Path("artifacts/_test_m5_run"))
    result = recommend_portfolio(agent, manifest, day=551,
                                 stock=np.full(8, 10), pipeline=np.zeros((3, 8)),
                                 last_sales=np.ones((8, 7)))
    assert len(result["orders"]) == 8
    assert result["spend"] <= 8 * 32
    assert result["policy_type"] == manifest["active_policy"]
    fallback_manifest = {**manifest, "promotion_eligible": False, "active_policy": "base_stock"}
    fallback = recommend_portfolio(agent, fallback_manifest, day=551,
                                   stock=np.full(8, 10), pipeline=np.zeros((3, 8)),
                                   last_sales=np.ones((8, 7)))
    assert fallback["policy_type"] == "base_stock"
    assert fallback["spend"] <= 8 * 32
    demo = recommend_portfolio(agent, fallback_manifest, day=551,
                               stock=np.full(8, 10), pipeline=np.zeros((3, 8)),
                               last_sales=np.ones((8, 7)), force_rl=True)
    assert demo["policy_type"] == "rl_unpromoted"


def test_residual_zero_is_exact_rule_and_hybrid_report_is_auditable():
    rule = np.array([[0.0, -1.0, -4.0, -9.0], [0.0, -2.0, -8.0, -18.0]])
    q_values = np.array([[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 5.0, 8.0]])
    np.testing.assert_array_equal(residual_scores(rule, q_values, 0), rule)
    assert np.isfinite(residual_scores(rule, q_values, 0.25)).all()
    path = Path("artifacts/_test_m5.csv")
    write_m5_fixture(path)
    data = load_m5(path, sku_count=8, train_end=500, validation_end=550)
    config = PortfolioEnv(data).config
    rule_result = replay(data, config, 550, 600, cover=2.0, recent=False)
    zero_residual = replay(data, config, 550, 600, DQNAgent(8, 4), cover=2.0,
                           recent=False, hybrid_alpha=0.0)
    assert zero_residual["daily_profit"] == rule_result["daily_profit"]
    output = Path("artifacts/_test_m5_hybrid")
    report = run_hybrid(path, output, store_id="CA_1", sku_count=8, seeds=(1,),
                        episodes=1, alphas=(0.0, 0.25), train_end=500, validation_end=550)
    assert report["model_selection"]["candidate_count"] == 2
    assert report["model_selection"]["selected_alpha"] in (0.0, 0.25)
    assert report["test"]["candidate"]["days"] == 50
    _, manifest = load_portfolio_model(output)
    assert manifest["schema_version"] == 3
    assert manifest["hybrid_alpha"] == report["model_selection"]["selected_alpha"]


def test_guided_ablation_preserves_family_comparison_and_fallback():
    path = Path("artifacts/_test_m5.csv")
    write_m5_fixture(path)
    output = Path("artifacts/_test_m5_guided")
    report = run_guided(path, output, store_id="CA_1", sku_count=8, seeds=(1,),
                        episodes=1, alphas=(0.0, 0.25), train_end=500,
                        validation_end=550)
    assert report["model_selection"]["candidate_count"] == 4
    assert set(report["test"]["by_trainer"]) == {"guided", "unguided"}
    assert report["test"]["selected"]["result"]["days"] == 50
    assert report["test"]["active_policy"] == (
        "hybrid_rl" if report["test"]["promotion_eligible"] else "base_stock"
    )
    _, manifest = load_portfolio_model(output)
    assert manifest["hybrid_alpha"] == report["model_selection"]["selected_alpha"]
