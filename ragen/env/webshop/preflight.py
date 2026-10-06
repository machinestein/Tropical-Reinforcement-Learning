"""CPU checks for WebShop data, real search and portable TROPIC replay."""

import argparse
import json
from pathlib import Path

from hydra import compose, initialize_config_dir

from .state import fingerprint
from .tropic_config import WebShopTropicEnvConfig
from .tropic_env import WebShopTropicEnv


def goal_indices(env, seed, count, mode):
    size = len(env.server.goals)
    if size <= 1500:
        raise ValueError("Original WebShop splits require more than 1500 goals")
    span, offset = {"train": (size - 1500, 1500), "val": (1000, 500), "test": (500, 0)}[mode]
    if not 1 <= count <= span:
        raise ValueError(f"Requested {count} distinct goals from the {span}-goal {mode} split")
    return [env._get_permuted_index((seed + i) % span + offset) for i in range(count)]


def run_preflight(config, pool_size):
    if pool_size < config.es_manager.train.env_groups:
        raise ValueError("TROPIC training pool must cover the training groups")
    env_config = WebShopTropicEnvConfig(**dict(config.custom_envs.WebShop.env_config or {}))
    env = WebShopTropicEnv(env_config)
    peer = None
    try:
        train = goal_indices(env, config.seed.train, pool_size, "train")
        validation = goal_indices(env, config.seed.val, config.es_manager.val.env_groups, "val")
        all_train = {env._get_permuted_index(i) for i in range(1500, len(env.server.goals))}
        if all_train.intersection(validation) or len(set(validation)) != len(validation):
            raise ValueError("WebShop training and validation goal indices overlap")
        peer = WebShopTropicEnv(env_config)
        env.reset(seed=config.seed.train)
        root = env.get_state()
        peer.set_state(root.to_dict())
        if env.session == peer.session or env.render() != peer.render():
            raise ValueError("WebShop snapshots require isolated sessions and identical root observations")
        goal = env.server.user_sessions[env.session]["goal"]
        query = str(goal.get("query") or goal.get("name") or "shirt").strip()
        _, _, _, info = env.step(f"search[{query}]")
        if not info["action_is_valid"]:
            raise ValueError("WebShop search preflight returned an invalid action")
        searched = env.get_state()
        peer.set_state(searched.to_dict())
        if peer.get_state().key("probe", 9) != searched.key("probe", 9):
            raise ValueError("WebShop search is not reproducible across sessions")
        products = [action for action in env.get_available_actions()
                    if action.startswith("click[") and action[6:-1].upper() in env.server.product_item_dict]
        if not products:
            raise ValueError("WebShop search returned no catalog products; check the Lucene index")
        env.step(products[0])
        peer.set_state(env.get_state().to_dict())
        _, reward, done, info = peer.step("click[buy now]")
        if not done or not info["action_is_valid"] or bool(reward) != (info["raw_reward"] >= 1.0):
            raise ValueError("WebShop purchase/reward preflight failed")
        peer.reset(seed=config.seed.train)
        if peer.get_state().state != root.state:
            raise ValueError("WebShop reset retained terminal purchase state")
        return {
            "dataset": env_config.dataset, "num_goals": len(env.server.goals),
            "train_seed": config.seed.train, "eval_seed": config.seed.val,
            "training_pool_goal_indices": train, "validation_goal_indices": validation,
            "evaluation_attempts": config.es_manager.val.group_size,
            "preflight_catalog_id": env.catalog_id,
            "validation_goal_hashes": [fingerprint(env.server.goals[i]) for i in validation],
            "warning": "Probe-process catalog only: the original simulator randomises prices/goals at startup. "
                       "This manifest does not certify identical catalog realisations in separate training processes.",
        }
    finally:
        env.close()
        if peer is not None:
            peer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pool-size", type=int, default=128)
    args, overrides = parser.parse_known_args()
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[3] / "config"), version_base=None):
        config = compose(config_name="_6_webshop", overrides=overrides)
    manifest = run_preflight(config, args.pool_size)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print("WebShop data/search/purchase/replay preflight passed.")


if __name__ == "__main__":
    main()
