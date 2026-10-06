"""CPU checks for the Sudoku protocol: determinism, solvability within budget, exact replay."""

import argparse
import hashlib
import json
from pathlib import Path

from hydra import compose, initialize_config_dir
import numpy as np

from .config import SudokuEnvConfig
from .tropic_env import SudokuTropicEnv


def puzzle_hash(grid):
    return hashlib.sha256(json.dumps(np.asarray(grid).tolist()).encode()).hexdigest()


def run_preflight(config, pool_size, budget, require_solvable):
    train_seed, val_seed = int(config.seed.train), int(config.seed.val)
    val_count = int(config.es_manager.val.env_groups)
    if pool_size < config.es_manager.train.env_groups:
        raise ValueError("TROPIC training pool must cover the training groups")
    train_seeds = list(range(train_seed, train_seed + pool_size))
    val_seeds = list(range(val_seed, val_seed + val_count))
    if set(train_seeds) & set(val_seeds):
        raise ValueError("Sudoku training and validation seed ranges overlap")
    env_config = SudokuEnvConfig(**dict(config.custom_envs.SimpleSudoku.env_config or {}))
    env, peer = SudokuTropicEnv(env_config), SudokuTropicEnv(env_config)
    blanks, hashes = {}, {}
    for seed in train_seeds + val_seeds:
        env.reset(seed=seed)
        peer.reset(seed=seed)
        if env.get_state() != peer.get_state():
            raise ValueError(f"Sudoku puzzle for seed {seed} is not deterministic")
        blanks[seed] = int(np.count_nonzero(env.current_grid == 0))
        hashes[seed] = puzzle_hash(env.initial_grid)
    if len(set(hashes[s] for s in val_seeds)) != len(val_seeds):
        raise ValueError("Validation seeds produced duplicate puzzles")
    worst = max(blanks.values())
    if require_solvable and worst > budget:
        raise ValueError(f"Action budget {budget} cannot fill {worst} blanks; TROPIC needs solvable episodes")
    env.reset(seed=val_seeds[0])
    root = env.get_state()
    placements = [(r, c, int(env.solution_grid[r, c])) for r, c in zip(*np.where(env.current_grid == 0))]
    for r, c, n in placements[:2]:
        env.step(f"place {n} at row {r + 1} col {c + 1}")
    peer.set_state(env.get_state().to_dict())
    if peer.render() != env.render() or peer.get_state().key("probe", budget) != env.get_state().key("probe", budget):
        raise ValueError("Sudoku snapshot replay does not reproduce the observation")
    _, _, done, info = peer.step("place 0 at row 1 col 1")
    if not done or info["action_is_valid"]:
        raise ValueError("Invalid placements must terminate TROPIC attempts")
    env.set_state(root.to_dict())
    success = False
    for r, c, n in placements[:budget]:
        _, _, done, info = env.step(f"{r + 1},{c + 1},{n}")
        success = bool(info["success"])
        if done:
            break
    if worst <= budget and not success:
        raise ValueError("Solving a validation puzzle from its solution did not report success")
    return {"grid_size": env_config.grid_size, "difficulty": env_config.difficulty, "action_budget": budget,
            "train_seed": train_seed, "eval_seed": val_seed, "training_pool_seeds": train_seeds,
            "validation_seeds": val_seeds, "max_blanks": worst, "solvable_within_budget": worst <= budget,
            "validation_puzzle_hashes": [hashes[s] for s in val_seeds]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pool-size", type=int, default=128)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--require-solvable", action="store_true")
    args, overrides = parser.parse_known_args()
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[3] / "config"), version_base=None):
        config = compose(config_name="_8_sudoku", overrides=overrides)
    manifest = run_preflight(config, args.pool_size, args.budget, args.require_solvable)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    note = "" if manifest["solvable_within_budget"] else " WARNING: puzzles are not solvable within this budget."
    print(f"Sudoku preflight passed: {manifest['max_blanks']} blanks at most, budget {args.budget}.{note}")


if __name__ == "__main__":
    main()
