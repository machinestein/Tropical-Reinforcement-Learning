"""CPU checks for the Countdown protocol: dataset, both policy interfaces, exact replay."""

import argparse
from fractions import Fraction
import json
from pathlib import Path

from hydra import compose, initialize_config_dir

from .config import CountdownEnvConfig
from .env import CountdownEnv, check_correctness, check_format
from .generate import build
from .tropic_env import CountdownStepEnv, fmt


def witness_expression(nums, steps):
    """Fold an operation sequence over the inputs into one parenthesised expression."""
    live = {}
    for v in nums:
        live.setdefault(Fraction(v), []).append(str(v))
    for step in steps:
        a, op, b = step.split(" ")
        a, b = Fraction(a), Fraction(b)
        ea, eb = live[a].pop(), live[b].pop()
        result = {"+": lambda: a + b, "-": lambda: a - b, "*": lambda: a * b, "/": lambda: a / b}[op]()
        live.setdefault(result, []).append(f"({ea} {op} {eb})")
    (expression,), = [v for v in live.values() if v]
    return expression


def run_preflight(config, pool_size, generate_missing=True):
    task = config.custom_envs.Countdown
    train_config = CountdownEnvConfig(**dict(task.env_config or {}))
    overrides = dict(config.es_manager.val.get("env_config_overrides", {}).get("Countdown", {}))
    val_config = CountdownEnvConfig(**{**dict(task.env_config or {}), **overrides})
    for path in {train_config.train_path, val_config.train_path}:
        if not Path(path).exists():
            if not generate_missing or "tropic_" not in Path(path).name:
                raise FileNotFoundError(f"Countdown data missing: {path}")
            build(Path(path).parent)
    if pool_size < config.es_manager.train.env_groups:
        raise ValueError("TROPIC training pool must cover the training groups")
    from dataclasses import replace
    step_budget = int(task.max_actions_per_traj) if task.env_type == "countdown_step" else 5
    train = CountdownStepEnv(replace(train_config, max_steps=step_budget))
    val = CountdownStepEnv(replace(val_config, max_steps=step_budget))
    single = CountdownEnv(val_config)
    same_file = Path(train_config.train_path).resolve() == Path(val_config.train_path).resolve()
    val_seed, val_count = int(config.seed.val), int(config.es_manager.val.env_groups)
    train_seed = int(config.seed.train)
    if val_count > len(val.data):
        raise ValueError(f"Requested {val_count} validation problems from {len(val.data)}")
    val_keys = {(tuple(sorted(int(v) for v in val.data[(val_seed + i) % len(val.data)]["nums"])),
                 int(val.data[(val_seed + i) % len(val.data)]["target"])) for i in range(val_count)}
    if len(val_keys) != val_count:
        raise ValueError("Validation seeds repeat problems")
    train_keys = {(tuple(sorted(int(v) for v in train.data[(train_seed + i) % len(train.data)]["nums"])),
                   int(train.data[(train_seed + i) % len(train.data)]["target"])) for i in range(pool_size)}
    overlap = len(train_keys & val_keys)
    if overlap and not same_file:
        raise ValueError("Countdown training pool and validation problems overlap")
    sizes = {}
    for i in range(val_count):
        row = val.data[(val_seed + i) % len(val.data)]
        sizes[len(row["nums"])] = sizes.get(len(row["nums"]), 0) + 1
    if max(sizes) - 1 > step_budget:
        raise ValueError("Action budget cannot use every number of the largest problems")
    checked = 0
    for i in range(min(val_count, 8)):
        seed = val_seed + i
        val.reset(seed=seed)
        row = val.data[val.index]
        steps = json.loads(row["solution"]) if "solution" in row else None
        if steps is None:
            continue
        root = val.get_state()
        for step in steps[:-1]:
            _, _, done, info = val.step(step)
            if done or not info["action_is_valid"]:
                raise ValueError(f"Witness prefix rejected for seed {seed}")
        peer = CountdownStepEnv(replace(val_config, max_steps=step_budget))
        peer.set_state(val.get_state().to_dict())
        if peer.render() != val.render() or peer.get_state().key("probe", 5) != val.get_state().key("probe", 5):
            raise ValueError("Countdown snapshot replay does not reproduce the observation")
        _, reward, done, info = peer.step(steps[-1])
        if not (done and info["success"] and reward == float(val_config.score)):
            raise ValueError(f"Witness does not solve seed {seed} in the step interface")
        single.reset(seed=seed)
        expression = witness_expression([int(v) for v in row["nums"]], steps)
        if not (check_format(expression, [int(v) for v in row["nums"]]) and check_correctness(expression, int(row["target"]))
                and single.compute_reward(expression, single.data[single.index]) == single.config.score):
            raise ValueError(f"Witness expression rejected by the original single-turn scorer for seed {seed}")
        val.set_state(root.to_dict())
        _, _, done, info = val.step("1 + 99999")
        if not done or info["action_is_valid"]:
            raise ValueError("Invalid operations must terminate TROPIC attempts")
        checked += 1
    return {"train_path": train_config.train_path, "val_path": val_config.train_path,
            "train_problems": len(train.data), "val_problems": len(val.data),
            "train_seed": train_seed, "eval_seed": val_seed, "pool_size": pool_size,
            "validation_sizes": sizes, "witnesses_checked": checked, "same_file": same_file,
            "train_val_problem_overlap": overlap}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pool-size", type=int, default=128)
    parser.add_argument("--config-name", default="_4_countdown_tropic")
    args, overrides = parser.parse_known_args()
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[3] / "config"), version_base=None):
        config = compose(config_name=args.config_name, overrides=overrides)
    manifest = run_preflight(config, args.pool_size)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Countdown preflight passed: {manifest['train_problems']} training / {manifest['val_problems']} validation "
          f"problems, sizes {manifest['validation_sizes']}, {manifest['witnesses_checked']} witnesses verified on both interfaces.")


if __name__ == "__main__":
    main()
