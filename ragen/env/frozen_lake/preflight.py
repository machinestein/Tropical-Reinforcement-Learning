"""CPU replay checks and an audit of the original FrozenLake map distribution."""

import argparse
from collections import deque
import hashlib
import json
from pathlib import Path

from hydra import compose, initialize_config_dir

from .config import FrozenLakeEnvConfig
from .env import FrozenLakeEnv
from .tropic_config import FrozenLakeTropicEnvConfig
from .tropic_env import FrozenLakeTropicEnv


def shortest_path(desc):
    """Oracle for preflight diagnostics only; never supplies targets to training."""
    width = len(desc[0])
    start = next((r, c) for r, row in enumerate(desc) for c, tile in enumerate(row) if tile == "S")
    queue, seen = deque([(start, [])]), {start}
    while queue:
        (r, c), path = queue.popleft()
        if desc[r][c] == "G":
            return path
        for action, (dr, dc) in enumerate(((0, -1), (1, 0), (0, 1), (-1, 0)), start=1):
            cell = (r + dr, c + dc)
            if (0 <= cell[0] < len(desc) and 0 <= cell[1] < width
                    and desc[cell[0]][cell[1]] != "H" and cell not in seen):
                seen.add(cell)
                queue.append((cell, path + [action]))
    return None


def run_preflight(config, pool_size):
    train_seed, val_seed = int(config.seed.train), int(config.seed.val)
    count = int(config.es_manager.val.env_groups)
    if pool_size < config.es_manager.train.env_groups or count < 1:
        raise ValueError("Training pool must cover training groups and validation must be nonempty")
    train_seeds = list(range(train_seed, train_seed + pool_size))
    val_seeds = list(range(val_seed, val_seed + count))
    baseline_end = train_seed + config.es_manager.train.env_groups * config.trainer.total_training_steps
    train_end = max(train_seed + pool_size, baseline_end)
    if train_seed < val_seed + count and val_seed < train_end:
        raise ValueError("FrozenLake training and validation seed ranges overlap")
    task = config.custom_envs.CoordFrozenLake
    budget = int(task.max_actions_per_traj)
    rules = dict(task.env_config)
    baseline = FrozenLakeEnv(FrozenLakeEnvConfig(**rules))
    env = FrozenLakeTropicEnv(FrozenLakeTropicEnvConfig(**rules, max_steps=budget))
    peer = FrozenLakeTropicEnv(FrozenLakeTropicEnvConfig(**rules, max_steps=budget))
    records = []
    try:
        for seed in train_seeds + val_seeds:
            baseline.reset(seed=seed)
            env.reset(seed=seed)
            peer.reset(seed=seed)
            root = env.get_state()
            if root != peer.get_state() or env.render() != baseline.render():
                raise ValueError(f"FrozenLake map mismatch for seed {seed}")
            path = shortest_path(root.desc)
            if path is None:
                raise ValueError(f"Unreachable FrozenLake map for seed {seed}")
            peer.set_state(json.loads(json.dumps(root.to_dict())))
            for index, action in enumerate(path[:budget], start=1):
                actual = env.step(action)
                if actual != peer.step(action):
                    raise ValueError(f"FrozenLake snapshot replay differs for seed {seed}")
                original = baseline.step(action)
                if actual[:2] != original[:2] or actual[3] != original[3]:
                    raise ValueError(f"FrozenLake transition differs from baseline for seed {seed}")
                if actual[2] != (original[2] or index == budget):
                    raise ValueError("FrozenLake termination differs from the shared action budget")
            if len(path) <= budget and not actual[3]["success"]:
                raise ValueError("A shortest path did not reach the goal")
            digest = hashlib.sha256(json.dumps(root.desc).encode()).hexdigest()
            records.append({"seed": seed, "map_hash": digest, "shortest_path_length": len(path),
                            "solvable_within_budget": len(path) <= budget})
    finally:
        baseline.close()
        env.close()
        peer.close()
    train, val = records[:pool_size], records[pool_size:]
    train_hashes = {r["map_hash"] for r in train}
    val_hashes = {r["map_hash"] for r in val}
    return {
        "size": env.config.size, "safe_tile_probability": env.config.p,
        "success_rate": env.config.success_rate, "action_budget": budget,
        "training_pool": train, "validation": val,
        "baseline_training_seed_range_exclusive": [train_seed, baseline_end],
        "validation_solvable_fraction": sum(r["solvable_within_budget"] for r in val) / count,
        "validation_duplicate_maps": count - len(val_hashes),
        "train_pool_validation_shared_maps": len(train_hashes & val_hashes),
        "note": "Original seed distribution retained, without filtering duplicate or over-budget maps. "
                "Map overlap audit covers the TROPIC pool, not every future baseline training map.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pool-size", type=int, default=128)
    args, overrides = parser.parse_known_args()
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[3] / "config"), version_base=None):
        config = compose(config_name="_3_frozen_lake", overrides=overrides)
    manifest = run_preflight(config, args.pool_size)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"FrozenLake preflight passed: deterministic replay; validation solvable within budget: "
          f"{manifest['validation_solvable_fraction']:.1%}; duplicate validation maps: "
          f"{manifest['validation_duplicate_maps']}; pool/validation shared maps: "
          f"{manifest['train_pool_validation_shared_maps']}. No maps filtered.")


if __name__ == "__main__":
    main()
