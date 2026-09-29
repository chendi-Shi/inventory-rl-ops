"""Portable model bundle with schema and accidental-corruption checks."""

import hashlib
import json
from pathlib import Path

import numpy as np

from inventory_rl.agent import DQNAgent
from inventory_rl.env import InventoryConfig, InventoryEnv

SCHEMA_VERSION = 1


def save_model(agent: DQNAgent, config: InventoryConfig, directory: Path,
               metadata: dict | None = None) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    model = directory / "model.npz"
    np.savez_compressed(model, **agent.online.params)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "algorithm": "Double DQN",
        "config": config.to_dict(),
        "state_dim": len(InventoryEnv(config).observation()),
        "action_dim": len(InventoryEnv(config).actions),
        "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "metadata": metadata or {},
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def load_model(directory: Path) -> tuple[DQNAgent, InventoryConfig, dict]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported model schema")
    model = directory / "model.npz"
    if hashlib.sha256(model.read_bytes()).hexdigest() != manifest["model_sha256"]:
        raise ValueError("model checksum mismatch")
    config_data = manifest["config"].copy()
    for key in ("initial_stock", "mean_demand", "lead_time", "unit_price", "unit_cost",
                "holding_cost", "lost_sale_penalty", "order_sizes"):
        config_data[key] = tuple(config_data[key])
    config = InventoryConfig(**config_data)
    env = InventoryEnv(config)
    agent = DQNAgent(len(env.observation()), len(env.actions))
    with np.load(model, allow_pickle=False) as weights:
        if set(weights.files) != set(agent.online.params):
            raise ValueError("unexpected model parameter names")
        for key, expected in agent.online.params.items():
            if weights[key].shape != expected.shape or not np.isfinite(weights[key]).all():
                raise ValueError(f"invalid model parameter: {key}")
            expected[...] = weights[key]
    agent.target.copy_from(agent.online)
    return agent, config, manifest
