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
from inventory_rl.m5_experiment import run as run_m5
from inventory_rl.m5_guided import run_guided
from inventory_rl.m5_hybrid import run_hybrid


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
    else:
        report = run_allocation(args.data, args.output, store_id=args.store, sku_count=args.skus)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
