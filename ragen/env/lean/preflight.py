"""Check the Lean data protocol and live verifier before allocating model workers."""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re

from hydra import compose, initialize_config_dir

from .config import LeanEnvConfig
from .env import LeanEnv
from .tropic_env import LeanTropicEnv


def statement_key(record):
    statement = re.sub(r"^(theorem|lemma)\s+\S+", r"\1 _", record["formal_statement"])
    fields = [record.get("imports", ""), record.get("preamble", ""), " ".join(statement.split())]
    return hashlib.sha256(json.dumps(fields).encode()).hexdigest()


def dataset_manifest(train, val, eval_problems, pool_size):
    def partition(records):
        names = [record["name"] for record in records]
        statements = [statement_key(record) for record in records]
        if len(set(names)) != len(names) or len(set(statements)) != len(statements):
            raise ValueError("Lean partition contains duplicate names or statements")
        for record in records:
            LeanTropicEnv.theorem_name(record)
        return {"count": len(records), "names": names,
                "sha256": hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()}

    manifest = {"train": partition(train), "val": partition(val)}
    if set(manifest["train"]["names"]) & set(manifest["val"]["names"]):
        raise ValueError("Lean train/validation theorem names overlap")
    if {statement_key(row) for row in train} & {statement_key(row) for row in val}:
        raise ValueError("Lean train/validation theorem statements overlap")
    if not 1 <= eval_problems <= len(val):
        raise ValueError(f"EVAL_PROBLEMS must be between 1 and {len(val)} (distinct validation theorems)")
    if not 8 <= pool_size <= len(train):
        raise ValueError(f"TRAIN_POOL_SIZE must be between 8 and {len(train)}")
    return manifest


def check_server(config):
    env = LeanTropicEnv(config)
    try:
        env.current_theorem = {"name": "ragen_preflight", "imports": "", "preamble": "",
                               "formal_statement": "theorem ragen_preflight : True := by",
                               "natural_language_statement": ""}
        partial = env._run_lean_query(["skip"])
        if not partial.accepted or partial.success or not env._goals(partial):
            raise RuntimeError("Kimina must report partial proofs as unsolved goals")
        good = env._run_lean_query(["exact True.intro"])
        if not good.success:
            raise RuntimeError(f"Kimina failed the valid-proof/axiom-audit probe: {good.message_objects}")
        for tactic in ("exact False.elim True.intro", "sorry"):
            bad = env._run_lean_query([tactic])
            if bad.success or bad.accepted:
                raise RuntimeError(f"Kimina incorrectly accepted the negative probe: {tactic}")
        env.reset(theorem_idx=0)
    finally:
        env.close()


def run_preflight(config, pool_size=128):
    train_config = LeanEnvConfig(**config.custom_envs.Lean.env_config)
    val_config = replace(train_config, **config.es_manager.val.env_config_overrides.Lean)
    train, val = LeanEnv(train_config), LeanEnv(val_config)
    try:
        manifest = dataset_manifest(train._dataset, val._dataset, config.es_manager.val.env_groups, pool_size)
        check_server(train_config)
        manifest.update(dataset=train_config.dataset_name_or_path, revision=train_config.dataset_revision,
                        train_partition=train_config.dataset_partition, val_partition=val_config.dataset_partition,
                        eval_seed=config.seed.val, eval_attempts=config.es_manager.val.group_size,
                        eval_names=[val._dataset[(config.seed.val + i) % len(val._dataset)]["name"]
                                    for i in range(config.es_manager.val.env_groups)])
        return manifest
    finally:
        train.close()
        val.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pool-size", type=int, default=128)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    config_dir = str(Path(__file__).resolve().parents[3] / "config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(config_name="_7_lean_light", overrides=args.overrides)
    manifest = run_preflight(config, args.pool_size)
    with args.output.open("w") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    print(f"Lean preflight passed: {manifest['train']['count']} training, "
          f"{manifest['val']['count']} held-out theorems; verifier probes passed.")


if __name__ == "__main__":
    main()
