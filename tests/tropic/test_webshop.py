"""TROPIC contracts using the real RAGEN adapter and a scripted simulator backend."""

from collections import defaultdict
import copy
import importlib
import json
from pathlib import Path
import re
import sys
from types import ModuleType, SimpleNamespace

from hydra import compose, initialize_config_dir
import pytest

from conftest import CharacterTokenizer
from test_collector import ScriptedProxy, score


@pytest.fixture
def webshop_backend(monkeypatch):
    class SimServer:
        def __init__(self):
            self.goals = [{"id": i, "asin": "ITEM", "query": "shirts", "instruction_text": f"goal {i}",
                           "goal_options": {"color": "red", "size": "large"}} for i in range(2004)]
            self.product_item_dict = {"ITEM": {"name": "shirt"}, "OTHER": {"name": "other shirt"}}
            self.product_prices = {"ITEM": 10., "OTHER": 20.}
            self.show_attrs = False
            self.assigned_instruction_text = None
            self.user_sessions = {}

        def get_page_name(self, url):
            return url.split("/")[3]

    server = SimServer()

    def parse_action(action):
        match = re.match(r"(.+)\[(.+)\]", action)
        return match.groups() if match else (action, None)

    class WebAgentTextEnv:
        def __init__(self, **kwargs):
            self.server = kwargs.get("server") or server
            self.session_prefix = kwargs["session_prefix"]
            self.session = None
            self.browser = SimpleNamespace(current_url=None)
            self.reset()

        def reset(self, session=None, instruction_text=None):
            self.session = self.session_prefix + str(session)
            goal = self.server.goals[session]
            self.server.user_sessions[self.session] = {"goal": goal, "done": False, "keywords": None,
                "page": None, "asin": None, "asins": set(), "options": {}, "actions": defaultdict(int)}
            self.browser.current_url = f"http://local/index/{self.session}"
            self.instruction_text = goal["instruction_text"]
            return self.observation, {}

        @property
        def observation(self):
            entry = self.server.user_sessions[self.session]
            return self.instruction_text + " page=" + self.server.get_page_name(self.browser.current_url) + str(entry["asin"])

        def get_instruction_text(self):
            return self.instruction_text

        def get_available_actions(self):
            page = self.server.get_page_name(self.browser.current_url)
            clickables = {"index": ["search"], "search_results": ["item", "other", "back to search"],
                          "item_page": ["buy now", "red", "blue", "large", "small", "back to search"],
                          "done": []}[page]
            return {"has_search_bar": page == "index", "clickables": clickables}

        def step(self, action):
            name, arg = parse_action(action)
            arg = arg.lower() if arg else arg
            entry = self.server.user_sessions[self.session]
            page = self.server.get_page_name(self.browser.current_url)
            reward, done = 0., False
            if name == "search" and arg:
                entry.update(keywords=arg.split(), page=1, asin=None, options={})
                entry["actions"]["search"] += 1
                page = "search_results"
            elif name == "click" and arg in WebAgentTextEnv.get_available_actions(self)["clickables"]:
                if arg in ("item", "other"):
                    entry["asin"] = arg.upper()
                    entry["asins"].add(arg.upper())
                    entry["actions"]["asin"] += 1
                    page = "item_page"
                elif arg in ("red", "blue", "large", "small"):
                    entry["options"]["color" if arg in ("red", "blue") else "size"] = arg
                    entry["actions"]["options"] += 1
                elif arg == "buy now":
                    reward = 1. if entry["asin"] == "ITEM" and entry["options"] == entry["goal"]["goal_options"] else .5
                    entry.update(done=True, reward=reward)
                    entry["actions"]["purchase"] += 1
                    page, done = "done", True
                elif arg == "back to search":
                    page = "index"
                    entry.update(keywords=None, page=None, asin=None, options={}, asins=set(), actions=defaultdict(int))
            self.browser.current_url = f"http://local/{page}/{self.session}"
            return self.observation, reward, done, {}

        def close(self):
            pass

    package = ModuleType("webshop_minimal")
    package.__path__ = []
    package.WebAgentTextEnv = WebAgentTextEnv
    package.init_basedir = lambda dataset: None
    utils = ModuleType("webshop_minimal.utils")
    utils.DEFAULT_FILE_PATH = ""
    engine = ModuleType("webshop_minimal.engine")
    engine.parse_action = parse_action
    backend_module = ModuleType("webshop_minimal.env")
    backend_module.SimServer = SimServer
    for name, module in {"webshop_minimal": package, "webshop_minimal.utils": utils,
                         "webshop_minimal.engine": engine, "webshop_minimal.env": backend_module}.items():
        monkeypatch.setitem(sys.modules, name, module)
    for name in list(sys.modules):
        if name.startswith("ragen.env.webshop") or name == "ragen.tropic.webshop_collector":
            monkeypatch.delitem(sys.modules, name)
    env_module = importlib.import_module("ragen.env.webshop.tropic_env")
    collector_module = importlib.import_module("ragen.tropic.webshop_collector")
    from ragen.env import REGISTERED_ENVS, REGISTERED_ENV_CONFIGS
    monkeypatch.setitem(REGISTERED_ENVS, "webshop_tropic", env_module.WebShopTropicEnv)
    monkeypatch.setitem(REGISTERED_ENV_CONFIGS, "webshop_tropic", env_module.WebShopTropicEnvConfig)
    yield SimpleNamespace(env=env_module.WebShopTropicEnv, config=env_module.WebShopTropicEnvConfig,
                          collector=collector_module.WebShopTropicCollector, server=server)
    for name in list(sys.modules):
        if name.startswith("ragen.env.webshop") or name == "ragen.tropic.webshop_collector":
            sys.modules.pop(name, None)


def advance(env, actions):
    for action in actions:
        env.step(action)


PREFIX = ["search[shirts]", "click[item]"]
FORWARD = PREFIX + ["click[red]", "click[large]"]
REVERSE = PREFIX + ["click[large]", "click[red]"]
REPEATED_OPTIONS = PREFIX + ["click[red]"] * 3 + ["click[large]"]
RESEARCHED = ["search[shirts]", "click[back to search]"] + FORWARD


def test_snapshot_roundtrip_and_session_isolation(webshop_backend):
    a, b = webshop_backend.env(), webshop_backend.env()
    a.reset(seed=10000)
    advance(a, FORWARD)
    snapshot = a.get_state()
    b.set_state(json.loads(json.dumps(snapshot.to_dict())))
    assert a.session != b.session
    assert a.render() == b.render()
    assert b.get_state().key("task", 9) == snapshot.key("task", 9)
    b.step("click[blue]")
    assert a.get_state().state == snapshot.state
    b.close()
    assert a.session in webshop_backend.server.user_sessions
    a.close()
    assert not webshop_backend.server.user_sessions


def test_options_are_visible_and_commuting_prefixes_can_join(webshop_backend):
    a, b = webshop_backend.env(), webshop_backend.env()
    a.reset(seed=10000)
    b.reset(seed=10000)
    advance(a, FORWARD)
    advance(b, REVERSE)
    assert a.get_state().actions != b.get_state().actions
    assert a.get_state().key("task", 9) == b.get_state().key("task", 9)
    assert '"color":"red"' in a.render() and '"size":"large"' in a.render()
    changed = copy.deepcopy(a.get_state())
    changed.state["session"]["options"]["color"] = "blue"
    assert changed.key("task", 9) != a.get_state().key("task", 9)


def test_state_prompt_tracks_budget_options_and_available_search(webshop_backend):
    env = webshop_backend.env(webshop_backend.config(max_steps=7))
    root = env.reset(seed=10000)
    assert "Actions remaining: 7" in root
    assert "search[<content>]" in root
    assert "no search bar" not in root
    assert env.instruction_text in root
    advance(env, FORWARD)
    observation = env.render()
    assert "Current page: item_page" in observation
    assert "Actions remaining: 3" in observation
    assert 'Selected options: {"color":"red","size":"large"}' in observation
    assert "no search bar" in observation
    assert "do not select them again" in observation
    assert "within 10 actions" not in observation
    assert "doesn't have to match perfectly" not in observation
    assert "click[size] then click[color]" not in observation
    assert "click[red]" in env.get_available_actions()
    _, _, done, info = env.step("click[red]")
    assert info["action_is_valid"] and not done
    terminal, reward, done, info = env.step("click[buy now]")
    assert done and reward == 1.0 and info["success"]
    assert terminal == "Shopping episode finished. Success: 1."


def test_join_ignores_counters_without_mutating_replay_snapshots(webshop_backend):
    env, peer = webshop_backend.env(), webshop_backend.env()
    snapshots = []
    for prefix in (REPEATED_OPTIONS, RESEARCHED):
        env.reset(seed=10000)
        advance(env, prefix)
        snapshots.append(env.get_state())
    a, b = snapshots
    assert a.num_env_steps == b.num_env_steps == 6
    assert a.state["session"]["actions"] != b.state["session"]["actions"]
    before = copy.deepcopy([a.to_dict(), b.to_dict()])
    assert a.key("task", 9) == b.key("task", 9)
    assert [a.to_dict(), b.to_dict()] == before
    for snapshot in snapshots:
        peer.set_state(snapshot.to_dict())
        assert peer.get_state().to_dict() == snapshot.to_dict()
    tampered = copy.deepcopy(b)
    tampered.state["session"]["actions"]["search"] += 1
    assert tampered.key("task", 9) == b.key("task", 9)
    with pytest.raises(ValueError, match="root replay"):
        peer.set_state(tampered)


def test_join_retains_visited_products_and_remaining_budget(webshop_backend):
    env = webshop_backend.env()
    env.reset(seed=10000)
    advance(env, FORWARD)
    snapshot = env.get_state()
    changed = copy.deepcopy(snapshot)
    changed.state["session"]["asins"].append("OTHER")
    assert changed.key("task", 9) != snapshot.key("task", 9)
    changed = copy.deepcopy(snapshot)
    changed.num_env_steps += 1
    assert changed.key("task", 9) != snapshot.key("task", 9)
    assert snapshot.key("task", 8) != snapshot.key("task", 9)


@pytest.mark.parametrize("field,value", [("catalog_id", "different"), ("max_steps", 10),
                                         ("num_env_steps", 3), ("mode", "val")])
def test_incompatible_snapshots_rejected(webshop_backend, field, value):
    env = webshop_backend.env()
    env.reset(seed=10000)
    snapshot = env.get_state()
    setattr(snapshot, field, value)
    with pytest.raises(ValueError, match="incompatible"):
        env.set_state(snapshot)


def test_tampered_snapshot_fails_replay(webshop_backend):
    env = webshop_backend.env()
    env.reset(seed=10000)
    advance(env, FORWARD)
    snapshot = env.get_state()
    snapshot.state["session"]["options"]["color"] = "blue"
    with pytest.raises(ValueError, match="root replay"):
        env.set_state(snapshot)


@pytest.mark.parametrize("options,success", [([], False), (["click[red]", "click[large]"], True)])
def test_only_perfect_purchase_is_success_and_reset_clears_done(webshop_backend, options, success):
    env = webshop_backend.env()
    env.reset(seed=10000)
    root = env.get_state()
    advance(env, PREFIX + options)
    _, reward, done, info = env.step("click[buy now]")
    assert done and bool(reward) is success and bool(info["success"]) is success
    assert info["success_purchase"]
    peer = webshop_backend.env()
    peer.set_state(env.get_state().to_dict())
    assert peer.render() == env.render()
    with pytest.raises(RuntimeError, match="termination"):
        env.step("click[buy now]")
    env.reset(seed=10000)
    assert env.get_state().state == root.state


@pytest.mark.parametrize("action", ["search[<r>]", "search[ ]", "click[missing]", "click[item] trailing", "garbage"])
def test_invalid_and_random_search_terminate_without_success(webshop_backend, action):
    env = webshop_backend.env()
    env.reset(seed=10000)
    _, reward, done, info = env.step(action)
    assert done and not reward and not info["action_is_valid"]
    assert env.num_env_steps == 1


def test_action_budget_and_goal_boundaries(webshop_backend):
    from ragen.env.webshop.preflight import goal_indices
    env = webshop_backend.env(webshop_backend.config(max_steps=2))
    assert set(goal_indices(env, 10000, 128, "train")).isdisjoint(goal_indices(env, 123, 512, "val"))
    with pytest.raises(ValueError, match="distinct"):
        goal_indices(env, 0, 1001, "val")
    env.reset(seed=10000)
    advance(env, PREFIX)
    assert env.num_env_steps == 2 and env.get_state().state["done"]
    with pytest.raises(RuntimeError):
        env.step("click[buy now]")


def test_preflight_exercises_search_purchase_and_replay(webshop_backend):
    from ragen.env.webshop.preflight import run_preflight
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[2] / "config"), version_base=None):
        cfg = compose(config_name="_6_webshop")
    manifest = run_preflight(cfg, 128)
    assert manifest["num_goals"] == 2004
    assert len(manifest["validation_goal_indices"]) == 512
    assert not webshop_backend.server.user_sessions


class WebShopProxy(ScriptedProxy):
    scripts = {0: FORWARD + ["click[buy now]"] * 5,
               1: REVERSE + ["click[small]"] + ["click[buy now]"] * 4}


@pytest.fixture
def webshop_proxy(webshop_backend):
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[2] / "config"), version_base=None):
        cfg = compose(config_name="_6_webshop_tropic")
    cfg.es_manager.train.env_groups = cfg.es_manager.train.seed_pool_size = 1
    cfg.es_manager.train.env_configs.n_groups = [1]
    cfg.es_manager.train.group_size = 2
    cfg.es_manager.val.env_groups = cfg.es_manager.val.group_size = 1
    cfg.es_manager.val.env_configs.n_groups = [1]
    cfg.tropic.waves_per_iteration = 1
    proxy = WebShopProxy(cfg, None, CharacterTokenizer())
    yield proxy, cfg
    proxy.train_es_manager.close()
    proxy.val_es_manager.close()


def test_collection_composes_a_purchase_not_sampled_end_to_end(webshop_backend, webshop_proxy):
    proxy, cfg = webshop_proxy
    collector = webshop_backend.collector(cfg, proxy, score)
    try:
        bases, metrics = collector.collect(1)
        graph = next(iter(collector.graphs.values()))
        paths = {tuple(graph.edges[key].action for key in path) for path in graph.solutions}
        assert tuple(FORWARD + ["click[buy now]"]) in paths
        assert tuple(REVERSE + ["click[buy now]"]) in paths
        assert metrics["tropic/root_verified"] == 1
        assert metrics["tropic/composition_new_solutions"] >= 1
        assert metrics["tropic/selected_path_length_mean"] == 5
        assert metrics["tropic/selected_repeated_option_action_fraction"] == 0
        assert metrics["tropic/selected_repeated_option_path_fraction"] == 0
        assert len(bases[graph.problem_id]) == 2
        failed = next(edge for edge in graph.edges.values()
                      if graph.nodes[edge.target].snapshot["state"]["done"] and not edge.success)
        prefixes, _ = graph.paths()
        failed_path = prefixes[failed.source][0] + (failed.key,)
        _, start = collector._prepare_start(graph, failed_path)
        assert not start["state"]["done"]
        assert collector.metrics["terminal_frontiers_trimmed"] == 1
        for path in graph.solutions:
            assert collector.verify(graph, path)
        assert any(len(event["edge_origins"]) > 1 for event in collector.events if event["origin"] == "composition")
        path = next(path for path in graph.solutions if graph.edges[path[2]].action == "click[large]")
        prefix, snapshot = collector._prepare_start(graph, path[:-1])
        assert prefix == path[:-1] and snapshot["actions"] == REVERSE
        graph.edges[path[0]].prompt_ids = (999,)
        assert not collector.verify(graph, path)
    finally:
        collector.close()


@pytest.mark.parametrize("composition", [False, True])
def test_composition_across_different_counters_is_root_verified(webshop_backend, webshop_proxy, composition):
    proxy, cfg = webshop_proxy
    cfg.tropic.composition_enabled = composition
    proxy.scripts = {0: REPEATED_OPTIONS + ["click[buy now]"] * 3,
                     1: RESEARCHED + ["click[small]"] + ["click[buy now]"] * 2}
    collector = webshop_backend.collector(cfg, proxy, score)
    try:
        _, metrics = collector.collect(1)
        graph = next(iter(collector.graphs.values()))
        paths = {tuple(graph.edges[key].action for key in path): path for path in graph.solutions}
        unsampled = tuple(RESEARCHED + ["click[buy now]"])
        assert tuple(REPEATED_OPTIONS + ["click[buy now]"]) in paths
        assert (unsampled in paths) is composition
        assert metrics.get("tropic/composition_new_solutions", 0) == int(composition)
        assert metrics.get("tropic/failed_verifications", 0) == 0
        assert metrics["tropic/selected_path_length_mean"] == 7
        assert metrics["tropic/selected_repeated_option_action_fraction"] == pytest.approx(2 / (14 if composition else 7))
        assert metrics["tropic/selected_repeated_option_path_fraction"] == (.5 if composition else 1.)
        if composition:
            path = paths[unsampled]
            assert collector.verify(graph, path)
            assert collector.verifier.get_state().state["session"]["actions"]["options"] == 2
            prefix, snapshot = collector._prepare_start(graph, path[:-1])
            assert prefix == path[:-1] and snapshot["actions"] == RESEARCHED
            peer = proxy.train_es_manager.envs[0]["env"]
            peer.set_state(snapshot)
            assert peer.get_state().to_dict() == snapshot
            assert peer.step("click[buy now]")[3]["success"]
    finally:
        collector.close()


def test_frontier_and_checkpoint_roundtrip(webshop_backend, webshop_proxy):
    proxy, cfg = webshop_proxy
    cfg.tropic.waves_per_iteration = 2
    collector = webshop_backend.collector(cfg, proxy, score)
    restored = webshop_backend.collector(cfg, proxy, score)
    try:
        _, metrics = collector.collect(1)
        assert metrics["tropic/restart_attempts"] > 0
        restored.load_state_dict(json.loads(json.dumps(collector.state_dict())))
        assert restored.state_dict() == collector.state_dict()
        for graph in restored.graphs.values():
            assert all(restored.verify(graph, path) for path in graph.solutions)
    finally:
        collector.close()
        restored.close()
