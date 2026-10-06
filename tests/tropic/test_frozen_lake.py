"""Exact replay, genuine unsampled composition and terminal safety on real FrozenLake."""

import copy
from dataclasses import replace
import json
from pathlib import Path
import re

from hydra import compose, initialize_config_dir
import pytest
import torch

from verl import DataProto
from conftest import CharacterTokenizer
from ragen.env.frozen_lake.config import FrozenLakeEnvConfig
from ragen.env.frozen_lake.env import FrozenLakeEnv
from ragen.env.frozen_lake.preflight import run_preflight, shortest_path
from ragen.env.frozen_lake.state import FrozenLakeSnapshot
from ragen.env.frozen_lake.tropic_config import FrozenLakeTropicEnvConfig
from ragen.env.frozen_lake.tropic_env import FrozenLakeTropicEnv
from ragen.llm_agent.agent_proxy import LLMAgentProxy
from ragen.tropic.frozen_lake_collector import FrozenLakeTropicCollector
from ragen.tropic.frozen_lake_graph import FrozenLakeFragmentGraph


ROOT = Path(__file__).resolve().parents[2]
MAP = ["SFFF", "FFFG", "FHFF", "FFFF"]


def load_config(name="_3_frozen_lake_tropic"):
    with initialize_config_dir(config_dir=str(ROOT / "config"), version_base=None):
        return compose(config_name=name)


def test_prompt_has_neutral_format_and_explicit_coordinate_and_grid_rules():
    from ragen.llm_agent.ctx_manager import ContextManager
    cfg = load_config()
    context = ContextManager(cfg, CharacterTokenizer(), mode="val")
    instruction = context.prefix_lookup[0]
    assert not re.search(r"<answer>\s*(?:Left|Down|Right|Up)\s*</answer>", instruction)
    assert "<answer>ACTION</answer>" in instruction
    for rule in ("(row, column)", "Left moves to (r, c-1)", "Down to (r+1, c)",
                 "Right to (r, c+1)", "Up to (r-1, c)", "no wraparound",
                 "P is the current player", "G is the goal", "O is a hole", "_ is safe ice",
                 "FIRST move"):
        assert rule in instruction
    assert cfg.agent_proxy.enable_think and cfg.agent_proxy.max_actions_per_turn == 1
    assert cfg.custom_envs.CoordFrozenLake.max_actions_per_traj == 10


@pytest.fixture
def lake_config():
    cfg = load_config()
    cfg.es_manager.train.env_groups = 1
    cfg.es_manager.train.group_size = 2
    cfg.es_manager.train.env_configs.n_groups = [1]
    cfg.es_manager.train.seed_pool_size = 2
    cfg.es_manager.val.env_groups = 1
    cfg.es_manager.val.group_size = 2
    cfg.es_manager.val.env_configs.n_groups = [1]
    cfg.actor_rollout_ref.rollout.max_model_len = 10000
    cfg.tropic.waves_per_iteration = 1
    return cfg


@pytest.mark.parametrize("seed", [123, 125, 10000, 10001])
def test_baseline_maps_transitions_and_json_snapshot_replay(seed):
    baseline = FrozenLakeEnv(FrozenLakeEnvConfig(success_rate=1, observation_format="grid_coord"))
    env, peer = FrozenLakeTropicEnv(), FrozenLakeTropicEnv()
    try:
        assert env.reset(seed=seed) == baseline.reset(seed=seed)
        peer.reset(seed=seed + 1)
        path = shortest_path(env.get_state().desc)
        assert path
        for action in path[:env.config.max_steps]:
            snapshot = json.loads(json.dumps(env.get_state().to_dict()))
            assert peer.set_state(snapshot) == env.render()
            actual = env.step(action)
            assert peer.step(action) == actual
            original = baseline.step(action)
            assert actual[:2] == original[:2] and actual[3] == original[3]
            assert env.get_state() == peer.get_state()
            assert actual[2] == (original[2] or env.num_env_steps == env.config.max_steps)
            if actual[2]:
                with pytest.raises(RuntimeError, match="termination"):
                    peer.step(1)
                break
    finally:
        baseline.close()
        env.close()
        peer.close()


def test_key_preserves_map_position_depth_budget_and_terminal_status():
    state = FrozenLakeSnapshot(MAP.copy(), 5, 2, 10, False)
    key = state.key("p", 10)
    assert key == FrozenLakeSnapshot.from_dict(json.loads(json.dumps(state.to_dict()))).key("p", 10)
    assert key != state.key("other", 10)
    assert key != state.key("p", 9)
    for change in [dict(position=4), dict(num_env_steps=3), dict(max_steps=12), dict(done=True),
                   dict(desc=["SFFF", "FFFG", "FFHF", "FFFF"])]:
        assert key != replace(state, **change).key("p", 10)


def test_commuting_actions_join_and_observation_is_history_independent():
    left, right = FrozenLakeTropicEnv(), FrozenLakeTropicEnv()
    root = FrozenLakeSnapshot(MAP.copy(), 0, 0, 10, False)
    try:
        left.set_state(root)
        right.set_state(root)
        for action in [3, 2]:
            left.step(action)
        for action in [2, 3]:
            right.step(action)
        assert left.get_state().key("map", 10) == right.get_state().key("map", 10)
        assert left.render() == right.render()
        left.step(1)
        assert root.position == 0 and right.get_state().position == 5
    finally:
        left.close()
        right.close()


@pytest.mark.parametrize("actions,success", [([3, 2, 3, 3], True), ([2, 3, 2], False), ([0], False), ([1] * 10, False)])
def test_goal_hole_invalid_action_and_budget_are_terminal(actions, success):
    env, peer = FrozenLakeTropicEnv(), FrozenLakeTropicEnv()
    try:
        env.set_state(FrozenLakeSnapshot(MAP.copy(), 0, 0, 10, False))
        for action in actions:
            _, reward, done, info = env.step(action)
        assert done and info["success"] is success and reward == float(success)
        assert env.num_env_steps == len(actions)
        assert env.get_state().done
        peer.set_state(env.get_state().to_dict())
        with pytest.raises(RuntimeError, match="termination"):
            peer.step(1)
        if actions == [0]:
            assert not info["action_is_valid"]
    finally:
        env.close()
        peer.close()


@pytest.mark.parametrize("kwargs", [dict(success_rate=.8), dict(render_mode="rgb_array"), dict(max_steps=0)])
def test_unsupported_rules_rejected(kwargs):
    with pytest.raises(ValueError):
        FrozenLakeTropicEnvConfig(**kwargs)


@pytest.mark.parametrize("change", [dict(desc=["SFF", "FFG"]), dict(position=16), dict(num_env_steps=11),
                                    dict(max_steps=11), dict(desc=["SFFX", "FFFG", "FHFF", "FFFF"]),
                                    dict(position=7), dict(num_env_steps=10), dict(desc=["SFFF"] * 4)])
def test_incompatible_snapshots_rejected(change):
    env = FrozenLakeTropicEnv()
    try:
        with pytest.raises(ValueError, match="snapshot"):
            env.set_state(replace(FrozenLakeSnapshot(MAP.copy(), 0, 0, 10, False), **change))
    finally:
        env.close()


class LakeProxy(LLMAgentProxy):
    scripts = {0: ["Right", "Down", "Right", "Right"], 1: ["Down", "Right", "Down"]}

    def generate_sequences(self, inputs):
        if inputs.meta_info.get("skip_generation"):
            return inputs
        rows = []
        for env_id in inputs.non_tensor_batch["env_ids"]:
            depth = self.train_es_manager.envs[env_id]["env"].num_env_steps
            action = self.scripts[int(env_id)][depth]
            rows.append(self.tokenizer.encode(f"Move {action}.</think><answer>{action}</answer><|im_end|>"))
        responses = torch.zeros(len(rows), max(map(len, rows)), dtype=torch.long)
        mask = torch.zeros_like(responses)
        for i, row in enumerate(rows):
            responses[i, :len(row)] = torch.tensor(row)
            mask[i, :len(row)] = 1
        return DataProto.from_dict(tensors={
            "responses": responses, "attention_mask": torch.cat([inputs.batch["attention_mask"], mask], dim=-1),
            "rollout_log_probs": torch.full_like(responses, -.25, dtype=torch.float32),
        }, non_tensors=inputs.non_tensor_batch, meta_info=inputs.meta_info)


@pytest.fixture
def lake_proxy(lake_config, monkeypatch):
    def reset(env, seed=None, mode=None):
        return env.set_state(FrozenLakeSnapshot(MAP.copy(), 0, 0, env.config.max_steps, False))
    monkeypatch.setattr(FrozenLakeTropicEnv, "reset", reset)
    proxy = LakeProxy(lake_config, None, CharacterTokenizer())
    yield proxy
    proxy.train_es_manager.close()
    proxy.val_es_manager.close()


def score(rows):
    return [-len(row.completion_ids) * .5 for row in rows]


def test_real_collector_composes_unsampled_path_and_never_certifies_holes(lake_config, lake_proxy):
    collector = FrozenLakeTropicCollector(lake_config, lake_proxy, score)
    try:
        bases, metrics = collector.collect(1)
        graph = next(iter(collector.graphs.values()))
        assert isinstance(graph, FrozenLakeFragmentGraph)
        actions = {tuple(graph.edges[k].action for k in p) for p in graph.solutions}
        assert actions == {(3, 2, 3, 3), (2, 3, 3, 3)}
        assert metrics["tropic/root_new_solutions"] == 1
        assert metrics["tropic/composition_new_solutions"] == 1
        assert metrics.get("tropic/failed_verifications", 0) == 0
        assert len(bases[graph.problem_id]) == 2
        assert metrics["tropic/selected_path_length_mean"] == 4
        assert metrics["tropic/selected_path_revisit_fraction"] == 0
        assert metrics["tropic/selected_wall_action_fraction"] == 0
        assert metrics["tropic/archive_replacements"] == 0
        assert all(collector.verify(graph, path) for path in graph.solutions)
        assert any(len(event["edge_origins"]) > 1 for event in collector.events if event["origin"] == "composition")
        prefixes, _ = graph.paths()
        hole = next(n for n in graph.nodes.values() if n.snapshot["position"] == 9)
        path = prefixes[hole.key][0]
        assert not collector.verify(graph, path)
        prefix, snapshot = collector._prepare_start(graph, path)
        assert len(prefix) == 2 and not snapshot["done"]
        assert snapshot["num_env_steps"] == 2
        assert collector.metrics["terminal_frontiers_trimmed"] == 1
        edge = graph.edges[bases[graph.problem_id][0][0]]
        edge.prompt_ids = (999,)
        assert not collector.verify(graph, bases[graph.problem_id][0])
    finally:
        collector.close()


def test_composition_disabled_control(lake_config, lake_proxy):
    lake_config.tropic.composition_enabled = False
    collector = FrozenLakeTropicCollector(lake_config, lake_proxy, score)
    try:
        _, metrics = collector.collect(1)
        assert metrics["tropic/retained_solutions"] == 1
        assert metrics.get("tropic/composition_candidates", 0) == 0
    finally:
        collector.close()


def test_selected_path_diagnostics_count_wall_moves_and_cell_revisits(lake_config, lake_proxy):
    lake_config.tropic.composition_enabled = False
    lake_proxy.scripts = {0: ["Left", "Right", "Down", "Right", "Right"],
                          1: ["Right", "Down", "Right", "Right"]}
    collector = FrozenLakeTropicCollector(lake_config, lake_proxy, score)
    try:
        _, metrics = collector.collect(1)
        assert metrics["tropic/selected_paths"] == 2
        assert metrics["tropic/selected_path_length_mean"] == 4.5
        assert metrics["tropic/selected_path_revisit_fraction"] == .5
        assert metrics["tropic/selected_wall_action_fraction"] == pytest.approx(1 / 9)
    finally:
        collector.close()


def test_full_archive_replacement_is_verified_and_logged_per_iteration(lake_config, lake_proxy):
    lake_config.tropic.composition_enabled = False
    lake_config.tropic.max_solutions_per_problem = 1
    lake_config.es_manager.train.seed_pool_size = 1
    lake_proxy.scripts = {i: ["Left", "Right", "Down", "Right", "Right"] for i in range(2)}
    collector = FrozenLakeTropicCollector(lake_config, lake_proxy, score)
    try:
        _, metrics = collector.collect(1)
        assert metrics["tropic/archive_replacements"] == 0
        assert metrics["tropic/archive_full_problems"] == 1
        lake_proxy.scripts = {i: ["Right", "Down", "Right", "Right"] for i in range(2)}
        bases, metrics = collector.collect(2)
        assert metrics["tropic/archive_replacements"] == 1
        assert metrics["tropic/selected_path_length_mean"] == 4
        assert metrics["tropic/selected_path_revisit_fraction"] == 0
        for problem, paths in bases.items():
            assert all(collector.verify(collector.graphs[problem], p) for p in paths)
        _, metrics = collector.collect(3)
        assert metrics["tropic/archive_replacements"] == 0
    finally:
        collector.close()


def test_validation_restarts_at_roots_and_logs_nested_pass_at_k(lake_config, lake_proxy):
    collector = FrozenLakeTropicCollector(lake_config, lake_proxy, score)
    try:
        collector.collect(1)
        archived = copy.deepcopy(collector.state_dict())
        manager = lake_proxy.val_es_manager
        for _ in range(2):
            pending = manager.reset()
            assert all(e["status"].seed == lake_config.seed.val for e in manager.envs)
            assert all(e["env"].num_env_steps == 0 for e in manager.envs)
            for depth in range(4):
                inputs = [{"env_id": state["env_id"],
                           "actions": [lake_proxy.scripts[state["env_id"]][depth]],
                           "llm_response": "", "llm_raw_response": ""} for state in pending]
                pending = manager.step(inputs)
            assert not pending
            states = manager.get_rollout_states()
            assert [s["metrics"]["CoordFrozenLake/success"] for s in states] == [1., 0.]
            assert all(s["metrics"]["CoordFrozenLake/pass@1"] == .5 for s in states)
            assert all(s["metrics"]["CoordFrozenLake/pass@2"] == 1. for s in states)
            assert collector.state_dict() == archived
    finally:
        collector.close()


def test_graph_checkpoint_and_second_wave_restarts(lake_config, lake_proxy):
    lake_config.tropic.waves_per_iteration = 2
    lake_config.es_manager.train.seed_pool_size = 1
    collector = FrozenLakeTropicCollector(lake_config, lake_proxy, score)
    restored = FrozenLakeTropicCollector(lake_config, lake_proxy, score)
    try:
        _, metrics = collector.collect(1)
        assert metrics["tropic/terminal_frontiers_trimmed"] > 0
        assert metrics["tropic/restart_attempts"] > 0
        restored.load_state_dict(json.loads(json.dumps(collector.state_dict())))
        assert all(isinstance(g, FrozenLakeFragmentGraph) for g in restored.graphs.values())
        assert restored.collect(2) == collector.collect(2)
    finally:
        collector.close()
        restored.close()


@pytest.mark.parametrize("scripts", [{0: ["Down"], 1: ["Down"]},
                                    {0: ["Right || Down"], 1: ["invalid"]}])
def test_failed_attempts_never_become_targets(lake_config, lake_proxy, scripts):
    lake_proxy.scripts = scripts
    lake_config.agent_proxy.max_turn = 1
    collector = FrozenLakeTropicCollector(lake_config, lake_proxy, score)
    try:
        bases, metrics = collector.collect(1)
        assert not any(bases.values())
        assert metrics["tropic/positive_problems"] == 0
        assert metrics["tropic/selected_path_length_mean"] == 0
        assert metrics["tropic/selected_path_revisit_fraction"] == 0
        assert metrics["tropic/selected_wall_action_fraction"] == 0
        if scripts[1] == ["invalid"]:
            assert metrics["tropic/invalid_decisions"] == 2
            assert metrics["tropic/retained_edges"] == 0
    finally:
        collector.close()


def test_preflight_audits_original_maps_and_does_not_filter():
    cfg = load_config("_3_frozen_lake")
    cfg.es_manager.val.env_groups = 4
    before = copy.deepcopy(cfg)
    manifest = run_preflight(cfg, 8)
    assert cfg == before
    assert len(manifest["training_pool"]) == 8
    assert len(manifest["validation"]) == 4
    assert manifest["success_rate"] == 1 and manifest["action_budget"] == 10
    assert 0 <= manifest["validation_solvable_fraction"] <= 1
    assert shortest_path(["SH", "HG"]) is None
    cfg.seed.val = 10010  
    with pytest.raises(ValueError, match="overlap"):
        run_preflight(cfg, 8)


def test_config_guards_and_trainer_dispatch(lake_config, lake_proxy, monkeypatch):
    from ragen.trainer.agent_trainer import RayAgentTrainer
    from ragen.trainer.tropic_trainer import RayTropicTrainer
    from ragen.tropic.config import validate_tropic_config
    validate_tropic_config(lake_config)
    trainer = object.__new__(RayTropicTrainer)
    trainer.config, trainer.agent_proxy = lake_config, lake_proxy
    monkeypatch.setattr(RayAgentTrainer, "init_agent_proxy", lambda self: None)
    trainer.init_agent_proxy()
    try:
        assert isinstance(trainer.collector, FrozenLakeTropicCollector)
    finally:
        trainer.collector.close()
    lake_config.custom_envs.CoordFrozenLake.env_config.success_rate = .8
    with pytest.raises(ValueError, match="deterministic"):
        validate_tropic_config(lake_config)
