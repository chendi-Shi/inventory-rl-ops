"""Training and evaluation entry points."""

import argparse
import json
from pathlib import Path

import numpy as np

from inventory_rl.agent import DQNAgent, Transition
from inventory_rl.artifact import load_model, save_model
from inventory_rl.env import InventoryConfig, InventoryEnv
from inventory_rl.evaluation import compare
from inventory_rl.m5_allocation import run_allocation
from inventory_rl.m5_allocator_audit import run_allocator_audit
from inventory_rl.m5_context_actor import run_context_development, run_context_final
from inventory_rl.m5_experiment import run as run_m5
from inventory_rl.m5_future_holdout import run_future_holdout
from inventory_rl.m5_guided import run_guided
from inventory_rl.m5_hybrid import run_hybrid
from inventory_rl.m5_policy_search import run_policy_search
from inventory_rl.m5_safety_future import run_safety_future
from inventory_rl.m5_sensitivity import run_sensitivity
from inventory_rl.m5_stress import run_stress
from inventory_rl.m5_transfer import run_transfer


def train(episodes: int, seed: int, output: Path) -> dict:
    if episodes < 1:
        raise ValueError("episodes must be positive")
    config = InventoryConfig()
    env = InventoryEnv(config, seed)
    agent = DQNAgent(len(env.observation()), len(env.actions), seed=seed)
    returns = []
    losses = []
    for episode in range(episodes):
        state = env.reset(seed + episode)
        total = 0.0
        epsilon = max(0.05, 1.0 - episode / max(episodes * 0.8, 1))
        while not env.done:
            action = agent.act(state, env.valid_actions(), epsilon)
            next_state, reward, done, _ = env.step(action)
            loss = agent.learn(Transition(
                state.copy(), action, reward, next_state.copy(), env.valid_actions().copy(), done
            ))
            if loss is not None:
                losses.append(loss)
            total += reward
            state = next_state
        returns.append(total)
    save_model(agent, config, output, {"train_episodes": episodes, "seed": seed})
    report = {
        "training": {"episodes": episodes, "seed": seed,
                     "last_10_mean_profit": float(np.mean(returns[-10:])),
                     "mean_huber_loss": float(np.mean(losses)) if losses else None},
        "holdout": compare(agent, config, list(range(10_000, 10_030))),
        "demand_surge_25pct": compare(
            agent, InventoryConfig(demand_multiplier=1.25), list(range(20_000, 20_030))
        ),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Inventory RL training and evaluation")
    commands = parser.add_subparsers(dest="command", required=True)
    train_cmd = commands.add_parser("train")
    train_cmd.add_argument("--episodes", type=int, default=200)
    train_cmd.add_argument("--seed", type=int, default=42)
    train_cmd.add_argument("--output", type=Path, default=Path("artifacts/run-42"))
    evaluate_cmd = commands.add_parser("evaluate")
    evaluate_cmd.add_argument("--model", type=Path, required=True)
    m5_cmd = commands.add_parser("m5-run", help="train and backtest a multi-SKU M5 policy")
    m5_cmd.add_argument("--data", type=Path, required=True)
    m5_cmd.add_argument("--output", type=Path, default=Path("artifacts/m5-ca1"))
    m5_cmd.add_argument("--store", default="CA_1")
    m5_cmd.add_argument("--skus", type=int, default=64)
    m5_cmd.add_argument("--episodes", type=int, default=60)
    m5_cmd.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    hybrid_cmd = commands.add_parser("m5-hybrid-run", help="evaluate baseline plus RL residual")
    hybrid_cmd.add_argument("--data", type=Path, required=True)
    hybrid_cmd.add_argument("--output", type=Path, default=Path("artifacts/m5-ca4-hybrid"))
    hybrid_cmd.add_argument("--store", default="CA_4")
    hybrid_cmd.add_argument("--skus", type=int, default=64)
    hybrid_cmd.add_argument("--episodes", type=int, default=60)
    hybrid_cmd.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    guided_cmd = commands.add_parser("m5-guided-run", help="compare guided and unguided M5 training")
    guided_cmd.add_argument("--data", type=Path, required=True)
    guided_cmd.add_argument("--output", type=Path, default=Path("artifacts/m5-tx1-guided"))
    guided_cmd.add_argument("--store", default="TX_1")
    guided_cmd.add_argument("--skus", type=int, default=64)
    guided_cmd.add_argument("--episodes", type=int, default=60)
    guided_cmd.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    allocation_cmd = commands.add_parser("m5-allocation-run", help="evaluate budget allocation scores")
    allocation_cmd.add_argument("--data", type=Path, required=True)
    allocation_cmd.add_argument("--output", type=Path, default=Path("artifacts/m5-tx2-allocation"))
    allocation_cmd.add_argument("--store", default="TX_2")
    allocation_cmd.add_argument("--skus", type=int, default=64)
    search_cmd = commands.add_parser("m5-policy-search-run",
                                 help="train a portfolio-return policy-search actor")
    search_cmd.add_argument("--data", type=Path, required=True)
    search_cmd.add_argument("--output", type=Path, default=Path("artifacts/m5-wi1-policy-search"))
    search_cmd.add_argument("--store", default="WI_1")
    search_cmd.add_argument("--skus", type=int, default=64)
    transfer_cmd = commands.add_parser("m5-transfer-run",
                                       help="test frozen actor on other M5 stores")
    transfer_cmd.add_argument("--data", type=Path, required=True)
    transfer_cmd.add_argument("--actor", type=Path, default=Path("models/wi1-policy-search"))
    transfer_cmd.add_argument("--output", type=Path, default=Path("artifacts/m5-wi-transfer"))
    transfer_cmd.add_argument("--stores", nargs="+", default=["WI_2", "WI_3"])
    transfer_cmd.add_argument("--skus", type=int, default=64)
    sensitivity_cmd = commands.add_parser("m5-sensitivity-run",
                                          help="post-hoc stronger-rule sensitivity")
    sensitivity_cmd.add_argument("--data", type=Path, required=True)
    sensitivity_cmd.add_argument("--actor", type=Path, default=Path("models/wi1-policy-search"))
    sensitivity_cmd.add_argument("--output", type=Path,
                                 default=Path("artifacts/m5-wi1-sensitivity"))
    sensitivity_cmd.add_argument("--skus", type=int, default=64)
    context_dev_cmd = commands.add_parser("m5-context-dev",
                                          help="train context actor without reading test")
    context_dev_cmd.add_argument("--data", type=Path, required=True)
    context_dev_cmd.add_argument("--output", type=Path, default=Path("artifacts/m5-wi2-context-dev"))
    context_dev_cmd.add_argument("--store", default="WI_2")
    context_dev_cmd.add_argument("--skus", type=int, default=64)
    context_cmd = commands.add_parser("m5-context-run",
                                      help="run fixed context actor holdout study")
    context_cmd.add_argument("--data", type=Path, required=True)
    context_cmd.add_argument("--output", type=Path, default=Path("artifacts/m5-tx3-context"))
    context_cmd.add_argument("--store", default="TX_3")
    context_cmd.add_argument("--skus", type=int, default=64)
    stress_cmd = commands.add_parser("m5-stress-run",
                                     help="post-hoc economics stress of frozen M5 policies")
    stress_cmd.add_argument("--data", type=Path, required=True)
    stress_cmd.add_argument("--output", type=Path, default=Path("artifacts/m5-stress"))
    stress_cmd.add_argument("--context-bundle", type=Path,
                            default=Path("models/tx3-context"))
    stress_cmd.add_argument("--policy-search-bundle", type=Path,
                            default=Path("models/wi1-policy-search"))
    future_cmd = commands.add_parser("m5-future-holdout",
                                     help="score frozen WI_1/TX_3 policies on M5 days 1914-1941")
    future_cmd.add_argument("--validation", type=Path,
                            default=Path("data/m5/sales_train_validation.csv"))
    future_cmd.add_argument("--evaluation", type=Path, required=True)
    future_cmd.add_argument("--output", type=Path,
                            default=Path("artifacts/m5-future-holdout"))
    safety_future_cmd = commands.add_parser(
        "m5-safety-future", help="compare frozen TX_3 safety rule on M5 days 1914-1941"
    )
    safety_future_cmd.add_argument("--validation", type=Path,
                                   default=Path("data/m5/sales_train_validation.csv"))
    safety_future_cmd.add_argument("--evaluation", type=Path, required=True)
    safety_future_cmd.add_argument("--output", type=Path,
                                   default=Path("artifacts/m5-safety-future"))
    allocator_audit_cmd = commands.add_parser(
        "m5-allocator-audit",
        help="post-hoc exact-versus-greedy audit of frozen WI_1/TX_3 policies",
    )
    allocator_audit_cmd.add_argument(
        "--validation", type=Path,
        default=Path("data/m5/sales_train_validation.csv"),
    )
    allocator_audit_cmd.add_argument("--evaluation", type=Path, required=True)
    allocator_audit_cmd.add_argument(
        "--future-report", type=Path,
        default=Path("docs/m5-future-holdout-report.json"),
    )
    allocator_audit_cmd.add_argument(
        "--output", type=Path, default=Path("artifacts/m5-allocator-audit"),
    )
    args = parser.parse_args()
    if args.command == "train":
        report = train(args.episodes, args.seed, args.output)
    elif args.command == "evaluate":
        agent, config, _ = load_model(args.model)
        report = compare(agent, config, list(range(10_000, 10_030)))
    elif args.command == "m5-run":
        report = run_m5(args.data, args.output, store_id=args.store, sku_count=args.skus,
                        episodes=args.episodes, seeds=tuple(args.seeds))
    elif args.command == "m5-hybrid-run":
        report = run_hybrid(args.data, args.output, store_id=args.store, sku_count=args.skus,
                            episodes=args.episodes, seeds=tuple(args.seeds))
    elif args.command == "m5-guided-run":
        report = run_guided(args.data, args.output, store_id=args.store, sku_count=args.skus,
                            episodes=args.episodes, seeds=tuple(args.seeds))
    elif args.command == "m5-allocation-run":
        report = run_allocation(args.data, args.output, store_id=args.store, sku_count=args.skus)
    elif args.command == "m5-policy-search-run":
        report = run_policy_search(args.data, args.output, store_id=args.store,
                                   sku_count=args.skus)
    elif args.command == "m5-transfer-run":
        report = run_transfer(args.data, args.actor, args.output,
                              stores=tuple(args.stores), sku_count=args.skus)
    elif args.command == "m5-sensitivity-run":
        report = run_sensitivity(args.data, args.actor, args.output,
                                 sku_count=args.skus)
    elif args.command == "m5-context-dev":
        report = run_context_development(args.data, args.output, store_id=args.store,
                                         sku_count=args.skus)
    elif args.command == "m5-context-run":
        report = run_context_final(args.data, args.output, store_id=args.store,
                                   sku_count=args.skus)
    elif args.command == "m5-future-holdout":
        report = run_future_holdout(args.validation, args.evaluation, args.output)
    elif args.command == "m5-safety-future":
        report = run_safety_future(args.validation, args.evaluation, args.output)
    elif args.command == "m5-allocator-audit":
        report = run_allocator_audit(
            args.validation, args.evaluation, args.output,
            future_report_path=args.future_report,
        )
    else:
        report = run_stress(args.data, args.output,
                            context_bundle=args.context_bundle,
                            policy_search_bundle=args.policy_search_bundle)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
