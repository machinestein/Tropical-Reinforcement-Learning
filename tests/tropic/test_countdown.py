"""Countdown TROPIC contracts: generator, step environment, exact snapshots, composition."""

from fractions import Fraction
import json
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest

from conftest import CharacterTokenizer
from test_collector import ScriptedProxy, score
from ragen.env.countdown.config import CountdownEnvConfig
from ragen.env.countdown.env import CountdownEnv, check_correctness, check_format
from ragen.env.countdown.generate import generate, write
from ragen.env.countdown.preflight import run_preflight, witness_expression
from ragen.env.countdown.tropic_env import CountdownStepEnv
from ragen.tropic.countdown_collector import CountdownTropicCollector
from ragen.env.countdown.state import CountdownSnapshot

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    directory = tmp_path_factory.mktemp("countdown")
    val = generate(11, 4)              
    train = generate(12, 8, exclude=val)
    write(val, directory / "tropic_val.parquet")
    write(train, directory / "tropic_train.parquet")
    return directory, train, val


def step_env(data, path="tropic_train.parquet", max_steps=5):
    return CountdownStepEnv(CountdownEnvConfig(train_path=str(data[0] / path), max_instances=10_000,
                                               solvable_filter=False, max_steps=max_steps))


def test_generator_is_deterministic_balanced_and_deduplicated(data):
    _, train, val = data
    assert generate(11, 4) == val
    assert sorted(len(r["nums"]) for r in val) == [3] * 4 + [4] * 4 + [5] * 4 + [6] * 4
    keys = {(tuple(sorted(r["nums"])), r["target"]) for r in train + val}
    assert len(keys) == len(train) + len(val)
    for row in val:
        assert all(1 <= v <= 20 for v in row["nums"]) and 1 <= row["target"] <= 100
        assert len(row["solution"]) == len(row["nums"]) - 1


def test_witness_solves_in_both_interfaces(data):
    env, single = step_env(data, "tropic_val.parquet"), CountdownEnv(CountdownEnvConfig(
        train_path=str(data[0] / "tropic_val.parquet"), max_instances=10_000, solvable_filter=False))
    for seed in range(16):
        env.reset(seed=seed); single.reset(seed=seed)
        row = env.data[env.index]; steps = json.loads(row["solution"])
        for step in steps:
            _, reward, done, info = env.step(step)
        assert done and info["success"] and reward == 1.0
        expression = witness_expression([int(v) for v in row["nums"]], steps)
        assert check_format(expression, [int(v) for v in row["nums"]]) and check_correctness(expression, int(row["target"]))
        assert single.compute_reward(expression, single.data[single.index]) == single.config.score


def test_operand_multiset_and_exact_arithmetic():
    env = CountdownStepEnv.__new__(CountdownStepEnv)
    env.config = CountdownEnvConfig(max_steps=5)
    env.target, env.nums, env.remaining = 7, [3, 3, 2], [Fraction(3), Fraction(3), Fraction(2)]
    env.num_env_steps, env._done, env._success = 0, False, False
    env.render_cache = None
    _, _, done, info = env.step("3 / 2")
    assert info["action_is_valid"] and not done and sorted(env.get_state().remaining) == ["3", "3/2"]
    _, _, done, info = env.step("3 * 3")          
    assert done and not info["action_is_valid"]
    with pytest.raises(RuntimeError):
        env.step("3 + 3/2")


@pytest.mark.parametrize("action", ["5 / 0", "1/0 + 2", "2 * 1/0", "abc", "5 + 5 + 5", "99 - 1"])
def test_bad_operations_terminate_without_success(data, action):
    env = step_env(data)
    env.reset(seed=0)
    if action == "5 / 0":
        env.remaining = [Fraction(5), Fraction(0)]
    _, reward, done, info = env.step(action)
    assert done and reward == 0 and not info["action_is_valid"] and not info["success"]


def test_commuting_operations_join_but_depth_and_values_distinguish(data):
    a, b = step_env(data), step_env(data)
    a.reset(seed=1); b.reset(seed=1)
    nums = a.nums
    assert len(nums) >= 4
    x, y, z, w = (str(v) for v in nums[:4])
    a.step(f"{x} + {y}"); a.step(f"{z} * {w}")
    b.step(f"{z} * {w}"); b.step(f"{x} + {y}")
    assert a.get_state().key("p", 5) == b.get_state().key("p", 5)
    assert a.render() == b.render()
    assert a.get_state() == b.get_state()
    assert a.get_state().key("p", 5) != a.get_state().key("p", 6)
    c = step_env(data); c.reset(seed=1); c.step(f"{x} - {y}"); c.step(f"{z} * {w}")
    assert c.get_state().key("p", 5) != a.get_state().key("p", 5) or int(x) - int(y) == int(x) + int(y)


def test_snapshot_roundtrip_is_exact_and_render_is_pure(data):
    env, peer = step_env(data), step_env(data)
    env.reset(seed=2); env.step(f"{env.nums[0]} + {env.nums[1]}")
    snapshot = env.get_state()
    peer.reset(seed=3)
    assert peer.set_state(snapshot.to_dict()) == env.render()
    assert peer.get_state() == snapshot and peer.num_env_steps == 1
    with pytest.raises(ValueError):
        peer.set_state({**snapshot.to_dict(), "num_env_steps": 3})
    reversed_snapshot = {**snapshot.to_dict(), "remaining": list(reversed(snapshot.remaining))}
    assert peer.set_state(reversed_snapshot) == env.render()
    assert peer.get_state() == snapshot


def test_preflight_checks_data_and_both_interfaces(data):
    with initialize_config_dir(config_dir=str(ROOT / "config"), version_base=None):
        cfg = compose(config_name="_4_countdown_tropic", overrides=[
            f"custom_envs.Countdown.env_config.train_path={data[0] / 'tropic_train.parquet'}",
            f"es_manager.val.env_config_overrides.Countdown.train_path={data[0] / 'tropic_val.parquet'}",
            "es_manager.val.env_groups=16", "es_manager.val.env_configs.n_groups=[16]", "seed.train=10000"])
    manifest = run_preflight(cfg, 8, generate_missing=False)
    assert manifest["witnesses_checked"] == 8 and manifest["train_val_problem_overlap"] == 0
    assert manifest["validation_sizes"] == {3: 4, 4: 4, 5: 4, 6: 4}
    with pytest.raises(ValueError, match="repeat|Requested"):
        cfg.es_manager.val.env_groups = 17
        run_preflight(cfg, 8, generate_missing=False)


@pytest.fixture
def countdown_config(data):
    with initialize_config_dir(config_dir=str(ROOT / "config"), version_base=None):
        cfg = compose(config_name="_4_countdown_tropic")
    cfg.custom_envs.Countdown.env_config.train_path = str(data[0] / "tropic_train.parquet")
    cfg.custom_envs.Countdown.env_config.max_instances = 10_000
    cfg.es_manager.val.env_config_overrides.Countdown.train_path = str(data[0] / "tropic_val.parquet")
    cfg.es_manager.val.env_config_overrides.Countdown.max_instances = 10_000
    cfg.seed.train = 0
    cfg.es_manager.train.env_groups = cfg.es_manager.train.seed_pool_size = 1
    cfg.es_manager.train.env_configs.n_groups = [1]
    cfg.es_manager.train.group_size = 2
    cfg.es_manager.val.env_groups = cfg.es_manager.val.group_size = 1
    cfg.es_manager.val.env_configs.n_groups = [1]
    cfg.tropic.waves_per_iteration = 1
    cfg.actor_rollout_ref.rollout.max_model_len = 10000
    return cfg


@pytest.mark.parametrize('variant', ['full', 'no_composition', 'successful_only'])
def test_collection_composes_an_operation_order_never_sampled(data, countdown_config, variant):
    probe = step_env(data); probe.reset(seed=0)
    steps = json.loads(probe.data[probe.index]["solution"])
    seed = 0
    while True:
        probe.reset(seed=seed); steps = json.loads(probe.data[probe.index]["solution"])
        a1, _, b1 = steps[0].split(" "); a2, _, b2 = steps[1].split(" ")
        live = [str(v) for v in probe.nums]
        if len(steps) >= 3 and all(t in live for t in (a1, b1, a2, b2)) and len({a1, b1, a2, b2}) == 4:
            break
        seed += 1
    countdown_config.seed.train = seed
    countdown_config.custom_envs.Countdown.max_actions_per_traj = countdown_config.agent_proxy.max_turn = len(steps)
    if variant == 'no_composition':
        countdown_config.tropic.composition_enabled = False
    elif variant == 'successful_only':
        OmegaConf.update(countdown_config, 'tropic.successful_fragments_only', True, force_add=True)

    class CountdownProxy(ScriptedProxy):
        scripts = {0: [steps[0], steps[1], "1 / 0"] + steps[2:],                
                   1: [steps[1], steps[0]] + steps[2:] + ["1 / 0"] * 3}         

    proxy = CountdownProxy(countdown_config, None, CharacterTokenizer())
    scored_origins = set()
    def scorer(rows):
        scored_origins.update(origin for edge in rows for origin in edge.rollout_ids)
        return score(rows)
    collector = CountdownTropicCollector(countdown_config, proxy, scorer)
    try:
        bases, metrics = collector.collect(1)
        graph = next(iter(collector.graphs.values()))
        solutions = {tuple(graph.edges[k].action for k in p) for p in graph.solutions}
        assert tuple([steps[1], steps[0]] + steps[2:]) in solutions
        if variant == 'full':
            assert tuple(steps) in solutions                
            assert metrics['tropic/composition_new_solutions'] >= 1
            assert len(bases[graph.problem_id]) == 2
        else:
            assert tuple(steps) not in solutions
            assert metrics.get('tropic/composition_new_solutions', 0) == 0
            assert len(bases[graph.problem_id]) == 1
        if variant == 'successful_only':
            assert metrics['tropic/success_only_discarded_transitions'] == 2
            assert metrics['tropic/success_only_admitted_transitions'] == len(steps)
            assert scored_origins == {'1:0:1'}
            assert all(edge.rollout_ids == ['1:0:1'] for edge in graph.edges.values())
        assert metrics['tropic/root_verified'] == 1
        assert metrics.get("tropic/failed_verifications", 0) == 0 and metrics["tropic/invalid_decisions"] == 1
        assert all(collector.verify(graph, path) for path in graph.solutions)
    finally:
        collector.close(); proxy.train_es_manager.close(); proxy.val_es_manager.close()


@pytest.mark.parametrize('successful_only', [False, True])
def test_both_commuting_prefixes_can_continue_from_the_same_prompt(countdown_config, monkeypatch, successful_only):
    if successful_only:
        OmegaConf.update(countdown_config, 'tropic.successful_fragments_only', True, force_add=True)
    def reset(env, seed=None, mode=None):
        return env.set_state(CountdownSnapshot(23, [2, 3, 7, 11], ["2", "3", "7", "11"], 0, 5))

    monkeypatch.setattr(CountdownStepEnv, "reset", reset)

    class BothSucceed(ScriptedProxy):
        scripts = {0: ["2 + 3", "7 + 11", "5 + 18"],
                   1: ["7 + 11", "2 + 3", "18 + 5"]}

    proxy = BothSucceed(countdown_config, None, CharacterTokenizer())
    collector = CountdownTropicCollector(countdown_config, proxy, score)
    try:
        _, metrics = collector.collect(1)
        assert metrics["tropic/root_verified"] == 2
        assert metrics.get("tropic/failed_verifications", 0) == 0
        graph = next(iter(collector.graphs.values()))
        joins = [node for node in graph.nodes.values() if node.depth == 2]
        assert len(joins) == 1 and joins[0].snapshot["remaining"] == ["5", "18"]
        if successful_only:
            assert metrics['tropic/success_only_admitted_transitions'] == 6
            assert metrics['tropic/success_only_discarded_transitions'] == 0
            assert metrics['tropic/composition_new_solutions'] >= 1
            solutions = {tuple(graph.edges[k].action for k in path) for path in graph.solutions}
            assert ('2 + 3', '7 + 11', '18 + 5') in solutions  
        assert all(collector.verify(graph, path) for path in graph.solutions)
    finally:
        collector.close(); proxy.train_es_manager.close(); proxy.val_es_manager.close()


@pytest.mark.parametrize('frontier_enabled', [True, False])
def test_terminal_frontier_restarts_at_root(countdown_config, monkeypatch, frontier_enabled):
    from ragen.tropic.graph import FragmentGraph

    def reset(env, seed=None, mode=None):
        return env.set_state(CountdownSnapshot(99, [2, 3, 7], ["2", "3", "7"], 0, 5))

    monkeypatch.setattr(CountdownStepEnv, "reset", reset)
    def terminal_frontier(graph, budget, eta):
        assert frontier_enabled, 'Root-only sampling must never choose a frontier'
        prefixes, _ = graph.paths()
        return next(paths[0] for key, paths in prefixes.items() if graph.nodes[key].depth == 2)

    monkeypatch.setattr(FragmentGraph, "choose_frontier", terminal_frontier)
    countdown_config.tropic.waves_per_iteration = 2
    countdown_config.tropic.frontier_enabled = frontier_enabled

    class NeverSucceeds(ScriptedProxy):
        scripts = {0: ["2 + 3", "5 + 7"], 1: ["2 + 3", "5 + 7"]}

    proxy = NeverSucceeds(countdown_config, None, CharacterTokenizer())
    collector = CountdownTropicCollector(countdown_config, proxy, score)
    try:
        _, metrics = collector.collect(1)
        assert metrics.get('tropic/terminal_restart_fallbacks', 0) == int(frontier_enabled)
        assert metrics["tropic/root_attempts"] == 4
        assert metrics.get("tropic/restart_attempts", 0) == 0
        assert metrics["tropic/positive_problems"] == 0
    finally:
        collector.close(); proxy.train_es_manager.close(); proxy.val_es_manager.close()


@pytest.mark.parametrize('last_action,valid_steps', [('5 * 7', 2), ('1/0 + 7', 1)])
def test_success_only_failed_countdown_attempts_leave_no_frontiers(countdown_config, monkeypatch, last_action, valid_steps):
    def reset(env, seed=None, mode=None):
        return env.set_state(CountdownSnapshot(12, [2, 3, 7], ['2', '3', '7'], 0, 5))

    monkeypatch.setattr(CountdownStepEnv, 'reset', reset)
    OmegaConf.update(countdown_config, 'tropic.successful_fragments_only', True, force_add=True)
    countdown_config.tropic.waves_per_iteration = 2

    class FailingProxy(ScriptedProxy):
        scripts = {0: ['2 + 3', last_action], 1: ['2 + 3', last_action]}

    proxy = FailingProxy(countdown_config, None, CharacterTokenizer())
    def scorer(rows):
        pytest.fail('Failed Countdown fragments must not be rescored')
    collector = CountdownTropicCollector(countdown_config, proxy, scorer)
    try:
        bases, metrics = collector.collect(1)
        graph = next(iter(collector.graphs.values()))
        assert not any(bases.values()) and not graph.edges and not graph.solutions
        assert set(graph.nodes) == {graph.root}
        assert metrics['tropic/root_attempts'] == 4 and metrics.get('tropic/restart_attempts', 0) == 0
        assert metrics['tropic/success_only_discarded_transitions'] == 4 * valid_steps
        assert metrics['tropic/success_only_admitted_transitions'] == 0
    finally:
        collector.close(); proxy.train_es_manager.close(); proxy.val_es_manager.close()
