"""CPU contract tests with a scripted Kimina response, not a substitute for Lean."""

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

from hydra import compose, initialize_config_dir
import pytest

from ragen.env.lean import env as lean_module
from ragen.env.lean.config import LeanEnvConfig
from ragen.env.lean.env import LeanEnv
from ragen.env.lean.preflight import check_server, dataset_manifest
from ragen.env.lean.tropic_env import LeanTropicEnv
from ragen.llm_agent.es_manager import EnvStateManager
from ragen.tropic.lean_collector import LeanTropicCollector
from conftest import CharacterTokenizer
from test_collector import ScriptedProxy, score


ROOT = Path(__file__).resolve().parents[2]


def theorem(name="demo", statement="True", **kwargs):
    return {"name": name, "imports": "", "preamble": "",
            "formal_statement": f"theorem {name} : {statement} := by",
            "natural_language_statement": "", **kwargs}


def reply(payload):
    return SimpleNamespace(results=[SimpleNamespace(response=payload, error=None, diagnostics={}, time=0.01)])


@pytest.fixture
def backend(monkeypatch):
    state = SimpleNamespace(reject_join=False, calls=[])
    monkeypatch.setattr(lean_module, "KiminaClient", object)
    monkeypatch.setattr(LeanEnv, "_load_dataset", lambda env: [
        theorem(f"{env.config.dataset_partition or 'train'}_{i}", statement=f"True /\\ True -- {i}\n")
        for i in range(8)])

    def compile_proof(env, code):
        state.calls.append(code)
        body = code.split(":= by\n")[1].split("\n#print")[0]
        steps = [line.strip() for line in body.splitlines() if line.strip()]
        name = env.current_theorem["name"]
        solved = steps[-1:] in (["all_goals trivial"], ["exact True.intro"])
        invalid = any(step in ("bad", "exact False.elim True.intro") for step in steps)
        if state.reject_join and steps == ["apply And.intro", "all_goals trivial"]:
            invalid = True
        if "sorry" in steps:
            return reply({"env": 1, "sorries": [{"goal": "True"}], "messages": []})
        if invalid:
            return reply({"env": 1, "messages": [{"severity": "error", "data": "tactic failed"}]})
        if solved:
            return reply({"env": 1, "messages": [{"severity": "info",
                                                   "data": f"'{name}' does not depend on any axioms"}]})
        goals = "unsolved goals\nTrue /\\ True" if steps == ["skip"] else "unsolved goals\nTrue\n\nTrue"
        return reply({"env": 1, "messages": [{"severity": "error", "data": goals}]})

    monkeypatch.setattr(LeanEnv, "_call_lean_server", compile_proof)
    return state


@pytest.fixture
def lean_config():
    with initialize_config_dir(config_dir=str(ROOT / "config"), version_base=None):
        cfg = compose(config_name="_7_lean_tropic")
    cfg.es_manager.train.env_groups = 1
    cfg.es_manager.train.env_configs.n_groups = [1]
    cfg.es_manager.train.group_size = 2
    cfg.es_manager.train.seed_pool_size = 1
    cfg.es_manager.val.env_groups = 1
    cfg.es_manager.val.env_configs.n_groups = [1]
    cfg.es_manager.val.group_size = 1
    cfg.custom_envs.Lean.max_actions_per_traj = 2
    cfg.agent_proxy.max_turn = 2
    cfg.tropic.waves_per_iteration = 1
    cfg.actor_rollout_ref.rollout.max_model_len = 10000
    return cfg


class LeanProxy(ScriptedProxy):
    scripts = {0: ["constructor", "all_goals trivial"], 1: ["apply And.intro", "skip"]}


@pytest.fixture
def lean_proxy(backend, lean_config):
    proxy = LeanProxy(lean_config, None, CharacterTokenizer())
    yield proxy
    proxy.train_es_manager.close()
    proxy.val_es_manager.close()


def test_snapshot_roundtrip_and_goal_projection_requires_replay(backend):
    env = LeanTropicEnv(LeanEnvConfig())
    env.reset(seed=0)
    root = env.get_state()
    env.step("constructor")
    a = env.get_state()
    env.set_state(root)
    env.step("apply And.intro")
    b = env.get_state()
    assert a.tactics != b.tactics
    assert a.key("p", 30) == b.key("p", 30)
    assert a.key("p", 30) != a.key("other", 30)
    assert a.key("p", 30) != a.key("p", 31)
    env.set_state(json.loads(json.dumps(a.to_dict())))
    assert env.tactic_history == ["constructor"]
    assert env.get_state() == a
    assert "Proof transcript" not in env.render()
    assert env.step("all_goals trivial")[3]["success"]
    assert "constructor\n  all_goals trivial" in backend.calls[-1]
    assert backend.calls[-1].rstrip().endswith(f"#print axioms {env.current_theorem['name']}")
    with pytest.raises(ValueError, match="snapshot"):
        env.set_state(replace(a, num_env_steps=7))
    with pytest.raises(ValueError, match="partition"):
        env.set_state(replace(a, theorem=theorem("foreign")))


def test_collection_certifies_an_unsampled_proof(lean_config, lean_proxy):
    collector = LeanTropicCollector(lean_config, lean_proxy, score)
    try:
        bases, metrics = collector.collect(1)
        graph = next(iter(collector.graphs.values()))
        paths = {tuple(graph.edges[k].action for k in path) for path in graph.solutions}
        assert ("constructor", "all_goals trivial") in paths
        assert ("apply And.intro", "all_goals trivial") in paths
        assert metrics["tropic/composition_new_solutions"] >= 1
        assert len(bases[graph.problem_id]) == 2
        assert all(collector.verify(graph, path) for path in graph.solutions)
        restored = LeanTropicCollector(lean_config, lean_proxy, score)
        restored.load_state_dict(json.loads(json.dumps(collector.state_dict())))
        assert restored.state_dict() == collector.state_dict()
        restored.close()
    finally:
        collector.close()


def test_identical_printed_goals_do_not_certify_an_invalid_join(backend, lean_config, lean_proxy):
    backend.reject_join = True
    collector = LeanTropicCollector(lean_config, lean_proxy, score)
    try:
        _, metrics = collector.collect(1)
        graph = next(iter(collector.graphs.values()))
        paths = {tuple(graph.edges[k].action for k in path) for path in graph.solutions}
        assert paths == {("constructor", "all_goals trivial")}
        assert metrics["tropic/failed_verifications"] >= 1
        assert graph.failed_joins
    finally:
        collector.close()


def test_frontier_uses_selected_prefix_not_merged_node_snapshot(lean_config, lean_proxy):
    collector = LeanTropicCollector(lean_config, lean_proxy, score)
    try:
        collector.collect(1)
        graph = next(iter(collector.graphs.values()))
        edge = next(e for e in graph.edges.values() if e.action == "apply And.intro")
        graph.nodes[edge.target].snapshot["tactics"] = ["constructor"]
        path, snapshot = collector._prepare_start(graph, (edge.key,))
        assert path == (edge.key,)
        assert snapshot["tactics"] == ["apply And.intro"]
    finally:
        collector.close()


def test_two_wave_collection_restarts_verified_concrete_prefixes(lean_config, lean_proxy, monkeypatch):
    lean_config.tropic.waves_per_iteration = 2
    lean_proxy.scripts = {0: ['constructor', 'skip'], 1: ['apply And.intro', 'skip']}
    original_generate = lean_proxy.generate_sequences
    decisions = 0

    def generate(inputs):
        nonlocal decisions
        output = original_generate(inputs)
        if not inputs.meta_info.get('skip_generation'):
            decisions += 1
            if decisions == 2:
                lean_proxy.scripts = LeanProxy.scripts
        return output

    monkeypatch.setattr(lean_proxy, 'generate_sequences', generate)
    collector = LeanTropicCollector(lean_config, lean_proxy, score)
    try:
        _, metrics = collector.collect(1)
        assert metrics['tropic/restart_attempts'] > 0
        assert metrics['tropic/restart_verified'] > 0
    finally:
        collector.close()


def test_invalid_tactics_terminate_only_tropic_and_are_not_fragments(backend, lean_config, lean_proxy):
    original, tropic = LeanEnv(), LeanTropicEnv()
    original.reset(seed=0)
    tropic.reset(seed=0)
    assert original.step("bad")[2] is False
    assert tropic.step("bad")[2] is True
    lean_proxy.scripts = {0: ["bad"], 1: ["bad"]}
    collector = LeanTropicCollector(lean_config, lean_proxy, score)
    try:
        _, metrics = collector.collect(1)
        assert metrics["tropic/retained_edges"] == 0
        assert metrics["tropic/invalid_decisions"] == 2
    finally:
        collector.close()


@pytest.mark.parametrize("axioms,accepted", [([], True), (["propext", "Classical.choice"], True),
                                          (["sorryAx"], False), (["customAxiom"], False)])
def test_completed_proofs_require_axiom_audit(backend, monkeypatch, axioms, accepted):
    env = LeanTropicEnv()
    env.reset(seed=0)
    name = env.current_theorem["name"]
    monkeypatch.setattr(LeanEnv, "_call_lean_server", lambda *args: reply({"messages": [
        {"severity": "info", "data": f"'{name}' depends on axioms: [{', '.join(axioms)}]"}]}))
    assert env.step("all_goals trivial")[3]["success"] is accepted


def test_missing_audit_and_sorry_are_not_success(backend, monkeypatch):
    env = LeanTropicEnv()
    env.reset(seed=0)
    assert not env.step("sorry")[3]["success"]
    monkeypatch.setattr(LeanEnv, "_call_lean_server", lambda *args: reply({"env": 1, "messages": []}))
    assert not env._run_lean_query(["all_goals trivial"]).success


@pytest.mark.parametrize("response", [{}, None])
def test_malformed_server_payload_fails_closed(backend, monkeypatch, response):
    env = LeanTropicEnv()
    env.reset(seed=0)
    monkeypatch.setattr(LeanEnv, "_call_lean_server", lambda *args: reply(response))
    with pytest.raises(RuntimeError, match="payload|service"):
        env.step("all_goals trivial")


def test_server_outage_is_not_silently_recorded_as_training_failure(backend, monkeypatch):
    env = LeanTropicEnv()
    env.reset(seed=0)
    calls, sleeps = [], []
    def unavailable(*args):
        calls.append(1)
        raise ConnectionError("connection refused")
    monkeypatch.setattr(LeanEnv, "_call_lean_server", unavailable)
    monkeypatch.setattr("ragen.env.lean.tropic_env.time.sleep", sleeps.append)
    with pytest.raises(RuntimeError, match="service failed after 4 attempts"):
        env.step("all_goals trivial")
    assert len(calls) == 5  
    assert sleeps == [2.0, 4.0, 8.0]  


def test_request_rejected_by_healthy_server_is_a_failed_attempt_not_an_outage(backend, monkeypatch):
    env = LeanTropicEnv()
    env.reset(seed=0)
    scripted, proofs = LeanEnv._call_lean_server, []
    def rejecting(self, proof_text):
        if "ragen_health_probe" in proof_text:
            return scripted(self, proof_text)  
        proofs.append(proof_text)
        raise RuntimeError("Request failed after 1 retries")  
    monkeypatch.setattr(LeanEnv, "_call_lean_server", rejecting)
    monkeypatch.setattr("ragen.env.lean.tropic_env.time.sleep", lambda seconds: None)
    _, reward, done, info = env.step("all_goals trivial")
    assert len(proofs) == 4 and done and not info["accepted"] and not info["success"]
    assert any("healthy server" in m["data"] for m in info["message_objects"])
    assert env.tactic_history == []  


def test_info_tree_request_is_optional(monkeypatch):
    from types import SimpleNamespace
    from kimina_client import Infotree
    monkeypatch.setattr(lean_module, "KiminaClient", object)
    monkeypatch.setattr(LeanEnv, "_load_dataset", lambda env: [theorem("t")])
    seen = {}
    def fake_client(**kwargs):
        seen.update(kwargs)
        return reply({"env": 1, "messages": []})
    for flag, expected in ((True, Infotree.tactics), (False, None)):
        env = LeanTropicEnv(LeanEnvConfig(request_infotree=flag))
        env._client = SimpleNamespace(api_check=fake_client)
        env.current_theorem = theorem("t")
        LeanEnv._call_lean_server(env, "theorem t : True := by\n  exact True.intro\n")
        assert seen["infotree"] == expected


def test_transient_server_error_is_retried_and_step_succeeds(backend, monkeypatch):
    env = LeanTropicEnv()
    env.reset(seed=0)
    scripted = LeanEnv._call_lean_server
    failures = iter([True, False])
    def flaky(self, proof_text):
        if next(failures):
            raise ConnectionError("connection reset")
        return scripted(self, proof_text)
    monkeypatch.setattr(LeanEnv, "_call_lean_server", flaky)
    monkeypatch.setattr("ragen.env.lean.tropic_env.time.sleep", lambda seconds: None)
    _, _, done, info = env.step("all_goals trivial")
    assert info["accepted"] and info["success"] and done


def test_tactic_timeout_terminates_attempt_without_training_target(backend, monkeypatch):
    env = LeanTropicEnv()
    env.reset(seed=0)
    def timeout(*args):
        raise TimeoutError("timed out")
    monkeypatch.setattr(LeanEnv, "_call_lean_server", timeout)
    _, _, done, info = env.step("all_goals trivial")
    assert done and not info['accepted'] and not info['success']


def test_lean_parse_error_terminates_attempt_without_crashing(backend, monkeypatch):
    env = LeanTropicEnv()
    env.reset(seed=0)
    monkeypatch.setattr(LeanEnv, "_call_lean_server", lambda *args: reply({"message": "unexpected token"}))
    _, _, done, info = env.step("bad")
    assert done and not info['accepted'] and not info['success']


def test_multiline_tactics_cannot_dedent_out_of_proof(backend):
    env = LeanTropicEnv()
    env.reset(seed=0)
    assert env._construct_proof(["constructor\nall_goals trivial"]).endswith(
        "  constructor\n  all_goals trivial\n")


def test_mode_overrides_keep_metric_tag_and_select_distinct_theorems(backend, lean_config):
    lean_config.es_manager.val.env_groups = 3
    lean_config.es_manager.val.env_configs.n_groups = [3]
    lean_config.es_manager.val.group_size = 2
    manager = EnvStateManager(lean_config, mode="val")
    try:
        manager.reset()
        names = [e['env'].current_theorem['name'] for e in manager.envs]
        assert len(set(names)) == 3
        assert names[::2] == names[1::2]
        assert all(name.startswith("test_") for name in names)
        assert all(e['tag'] == 'Lean' for e in manager.envs)
        manager.reset()
        assert names == [e['env'].current_theorem['name'] for e in manager.envs]
    finally:
        manager.close()


def test_dataset_partitions_are_explicit_and_cache_keys_do_not_mix(monkeypatch):
    import datasets
    monkeypatch.setattr(lean_module, "KiminaClient", object)
    monkeypatch.setattr(lean_module, "_DATASET_CACHE", {})
    rows = [theorem("a", split="valid"), theorem("b", split="test")]
    calls = []
    def load(*args, **kwargs):
        calls.append((args, kwargs))
        return rows
    monkeypatch.setattr(datasets, "load_dataset", load)
    original = LeanEnv()
    train = LeanEnv(LeanEnvConfig(dataset_partition="valid", dataset_revision="pinned"))
    val = LeanEnv(LeanEnvConfig(dataset_partition="test", sample_mode="sequential"))
    assert len(original._dataset) == 2
    assert train._dataset[0]['name'] == 'a'
    assert val._dataset[0]['name'] == 'b'
    assert calls[1][1]['revision'] == 'pinned'
    assert 'revision' not in calls[0][1]
    assert LeanEnvConfig().dataset_partition is None
    assert LeanEnvConfig().sample_mode == 'random'


def test_data_manifest_detects_overlap_and_oversampling():
    train = [theorem(f"a{i}", statement=f"{i} = {i}") for i in range(8)]
    val = [theorem("heldout", statement="True")]
    assert dataset_manifest(train, val, 1, 8)['val']['count'] == 1
    with pytest.raises(ValueError, match="names overlap"):
        dataset_manifest(train, [train[0]], 1, 8)
    with pytest.raises(ValueError, match="statements overlap"):
        dataset_manifest(train, [theorem("renamed", statement="0 = 0")], 1, 8)
    with pytest.raises(ValueError, match="EVAL_PROBLEMS"):
        dataset_manifest(train, val, 2, 8)
    with pytest.raises(ValueError, match="TRAIN_POOL_SIZE"):
        dataset_manifest(train, val, 1, 9)


def test_preflight_exercises_live_contract(backend):
    check_server(LeanEnvConfig())
    assert any("exact True.intro" in code for code in backend.calls)
    assert any("sorry" in code for code in backend.calls)
