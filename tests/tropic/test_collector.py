import json
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl import DataProto
from ragen.env.sokoban.env import SokobanEnv
from ragen.env.sokoban.state import SokobanSnapshot
from ragen.llm_agent.agent_proxy import LLMAgentProxy
from ragen.tropic.collector import TropicCollector
from conftest import CharacterTokenizer


class ScriptedProxy(LLMAgentProxy):
    scripts = {0: ['Up', 'Right', 'Left', 'Up', 'Right'],
               1: ['Right', 'Up', 'Down', 'Left', 'Down']}

    def generate_sequences(self, inputs):
        if inputs.meta_info.get('skip_generation'):
            return inputs
        rows = []
        for env_id in inputs.non_tensor_batch['env_ids']:
            depth = self.train_es_manager.envs[env_id]['env'].num_env_steps
            action = self.scripts[int(env_id)][depth]
            text = f'At this state choose {action}.</think><answer>{action}</answer><|im_end|>'
            rows.append(self.tokenizer.encode(text))
        width = max(map(len, rows))
        responses = torch.zeros(len(rows), width, dtype=torch.long)
        mask = torch.zeros_like(responses)
        for i, row in enumerate(rows):
            responses[i, :len(row)] = torch.tensor(row)
            mask[i, :len(row)] = 1
        return DataProto.from_dict(tensors={
            'responses': responses,
            'attention_mask': torch.cat([inputs.batch['attention_mask'], mask], dim=-1),
            'rollout_log_probs': torch.full_like(responses, -.25, dtype=torch.float32),
        }, non_tensors=inputs.non_tensor_batch, meta_info=inputs.meta_info)


@pytest.fixture
def proxy(config, monkeypatch):
    def reset(env, seed=None, mode=None):
        fixed = np.ones((5, 5), dtype=int)
        fixed[[0, -1], :] = 0
        fixed[:, [0, -1]] = 0
        fixed[1, 3] = 2
        state = fixed.copy()
        state[3, 1] = 5
        state[1, 2] = 4
        return env.set_state(SokobanSnapshot(fixed.tolist(), state.tolist(), [3, 1],
                                            [[[1, 3], [1, 2]]], 0, 0, 0., None, None, env.max_steps, 1))
    monkeypatch.setattr(SokobanEnv, 'reset', reset)
    instance = ScriptedProxy(config, None, CharacterTokenizer())
    yield instance
    instance.train_es_manager.close()
    instance.val_es_manager.close()


def score(rows):
    return [-len(row.completion_ids) * .5 for row in rows]


def test_real_sokoban_collection_certifies_unsampled_composition(config, proxy):
    collector = TropicCollector(config, proxy, score)
    try:
        bases, metrics = collector.collect(1)
        graph = next(iter(collector.graphs.values()))
        actions = {tuple(graph.edges[k].action for k in p) for p in graph.solutions}
        assert (1, 4, 3, 1, 4) in actions
        assert (4, 1, 3, 1, 4) in actions  
        assert metrics['tropic/root_verified'] == 1
        assert metrics['tropic/composition_new_solutions'] >= 1
        assert metrics.get('tropic/failed_verifications', 0) == 0
        assert len(bases[graph.problem_id]) == 2
        assert all(collector.verify(graph, path) for path in graph.solutions)
        edge = next(iter(graph.edges.values()))
        assert edge.behavior_log_probs[0] == pytest.approx(-.25 * len(edge.completion_ids))
        assert edge.log_prob == pytest.approx(-.5 * len(edge.completion_ids))
        assert any(len(e['edge_origins']) > 1 for e in collector.events if e['origin'] == 'composition')
    finally:
        collector.close()


def test_composition_off_is_verified_replay_control(config, proxy):
    config.tropic.composition_enabled = False
    collector = TropicCollector(config, proxy, score)
    try:
        _, metrics = collector.collect(1)
        assert metrics['tropic/retained_solutions'] == 1
        assert metrics.get('tropic/composition_candidates', 0) == 0
    finally:
        collector.close()


def test_frontiers_preserve_depth_and_memory_resume(config, proxy):
    config.tropic.waves_per_iteration = 2
    config.es_manager.train.seed_pool_size = 1
    collector = TropicCollector(config, proxy, score)
    restored = TropicCollector(config, proxy, score)
    try:
        _, metrics = collector.collect(1)
        assert metrics['tropic/restart_attempts'] > 0
        snapshot = json.loads(json.dumps(collector.state_dict()))
        restored.load_state_dict(snapshot)
        expected_bases, expected_metrics = collector.collect(2)
        actual_bases, actual_metrics = restored.collect(2)
        assert actual_bases == expected_bases
        assert actual_metrics == expected_metrics
        for graph in restored.graphs.values():
            assert all(graph.nodes[e.target].depth == graph.nodes[e.source].depth + 1 for e in graph.edges.values())
            assert max(n.depth for n in graph.nodes.values()) <= 5
    finally:
        collector.close()
        restored.close()


def test_malformed_actions_terminate_without_entering_graph(config, proxy):
    proxy.scripts = {0: ['Up || Right'], 1: ['not_an_action']}
    collector = TropicCollector(config, proxy, score)
    try:
        bases, metrics = collector.collect(1)
        assert not any(bases.values())
        assert metrics['tropic/invalid_decisions'] == 2
        assert metrics['tropic/retained_edges'] == 0
        assert metrics['tropic/model_decisions'] == 2
    finally:
        collector.close()


def test_replay_rejects_changed_exact_context(config, proxy):
    collector = TropicCollector(config, proxy, score)
    try:
        bases, _ = collector.collect(1)
        graph = next(iter(collector.graphs.values()))
        path = bases[graph.problem_id][0]
        graph.edges[path[0]].prompt_ids = (999,)
        assert not collector.verify(graph, path)
    finally:
        collector.close()


def test_recurring_training_seeds_leave_validation_roots_unchanged(config, proxy):
    train = proxy.train_es_manager
    seeds = []
    for _ in range(3):
        train.reset()
        seeds.append([entry['status'].seed for entry in train.envs])
    assert seeds == [[10000, 10000], [10001, 10001], [10000, 10000]]
    val = proxy.val_es_manager
    val.reset()
    seed = val.envs[0]['status'].seed
    val.reset()
    assert val.envs[0]['status'].seed == seed == config.seed.val


def test_root_diagnostics_keep_exact_reasoning_without_actions(config, proxy):
    collector = TropicCollector(config, proxy, score)
    try:
        collector.collect(1)
        batch = collector.root_diagnostics_batch
        assert batch.meta_info['root_only_diagnostics']
        assert proxy.tokenizer.decode(batch.non_tensor_batch['first_turn_reasoning_ids'][0]) == 'At this state choose Up.'
        assert proxy.tokenizer.decode(batch.non_tensor_batch['first_turn_reasoning_ids'][1]) == 'At this state choose Right.'
        assert batch.non_tensor_batch['first_turn_prompt_ids'][0] == batch.non_tensor_batch['first_turn_prompt_ids'][1]
        collector.collect(2)
        assert collector.root_diagnostics_batch is None
    finally:
        collector.close()


@pytest.mark.parametrize('failed_script', [
    ['Right', 'Up', 'Down', 'Left', 'Down'],
    ['Up', 'Right', 'Down', 'Down', 'Down'],  
    ['Up', 'not_an_action'],
])
def test_success_only_never_retains_or_scores_failed_rollout_fragments(config, proxy, failed_script):
    OmegaConf.update(config, 'tropic.successful_fragments_only', True, force_add=True)
    proxy.scripts = {0: ScriptedProxy.scripts[0], 1: failed_script}
    scored_origins = set()
    def scorer(rows):
        scored_origins.update(origin for edge in rows for origin in edge.rollout_ids)
        return score(rows)
    collector = TropicCollector(config, proxy, scorer)
    try:
        bases, metrics = collector.collect(1)
        graph = next(iter(collector.graphs.values()))
        assert metrics['tropic/root_attempts'] == 2
        assert metrics['tropic/root_verified'] == 1
        assert metrics.get('tropic/composition_new_solutions', 0) == 0
        assert metrics['tropic/success_only_discarded_transitions'] == (1 if len(failed_script) == 2 else 5)
        assert metrics['tropic/success_only_admitted_transitions'] == 5
        assert len(graph.edges) == 5 and len(graph.nodes) == 6
        assert len(bases[graph.problem_id]) == 1
        assert scored_origins == {'1:0:0'}
        assert all(edge.rollout_ids == ['1:0:0'] for edge in graph.edges.values())
        assert all(collector.verify(graph, path) for path in bases[graph.problem_id])
    finally:
        collector.close()


def test_success_only_all_failed_rollouts_leave_only_root_and_no_scoring(config, proxy):
    OmegaConf.update(config, 'tropic.successful_fragments_only', True, force_add=True)
    proxy.scripts = {0: ['Down'] * 5, 1: ['Down'] * 5}
    def scorer(rows):
        pytest.fail('Failed fragments must not be rescored')
    collector = TropicCollector(config, proxy, scorer)
    try:
        bases, metrics = collector.collect(1)
        graph = next(iter(collector.graphs.values()))
        assert not any(bases.values())
        assert not graph.edges and not graph.solutions
        assert set(graph.nodes) == {graph.root}
        assert metrics['tropic/success_only_buffered_transitions'] == 10
        assert metrics['tropic/success_only_discarded_transitions'] == 10
        assert metrics['tropic/success_only_admitted_transitions'] == 0
    finally:
        collector.close()


@pytest.mark.parametrize('capacity', [4, 5])
def test_success_only_capacity_is_atomic_and_not_consumed_by_failed_fragments(config, proxy, capacity):
    OmegaConf.update(config, 'tropic.successful_fragments_only', True, force_add=True)
    config.tropic.max_edges_per_problem = capacity
    proxy.scripts = {0: ScriptedProxy.scripts[1], 1: ScriptedProxy.scripts[0]}
    collector = TropicCollector(config, proxy, score)
    try:
        bases, metrics = collector.collect(1)
        graph = next(iter(collector.graphs.values()))
        if capacity == 5:
            assert len(graph.edges) == 5
            assert len(bases[graph.problem_id]) == 1
            assert metrics['tropic/success_only_admitted_transitions'] == 5
            assert all(edge.rollout_ids == ['1:0:1'] for edge in graph.edges.values())
        else:
            assert not graph.edges and set(graph.nodes) == {graph.root}
            assert not any(bases.values())
            assert metrics['tropic/success_only_capacity_rejected_rollouts'] == 1
            assert metrics['tropic/success_only_discarded_transitions'] == 10
    finally:
        collector.close()


def test_success_only_failed_replay_cannot_mutate_memory_or_shared_edge_provenance(config, proxy, monkeypatch):
    OmegaConf.update(config, 'tropic.successful_fragments_only', True, force_add=True)
    config.es_manager.train.seed_pool_size = 1
    collector = TropicCollector(config, proxy, score)
    try:
        collector.collect(1)
        graph = next(iter(collector.graphs.values()))
        origins = {key: list(edge.rollout_ids) for key, edge in graph.edges.items()}
        nodes = set(graph.nodes)
        solutions = dict(graph.solutions)
        monkeypatch.setattr(collector, 'verify', lambda *args, **kwargs: False)
        _, metrics = collector.collect(2)
        assert metrics['tropic/success_only_failed_verifications'] == 1
        assert metrics['tropic/success_only_discarded_transitions'] == 10
        assert {key: edge.rollout_ids for key, edge in graph.edges.items()} == origins
        assert set(graph.nodes) == nodes and graph.solutions == solutions
    finally:
        collector.close()


def test_success_only_still_composes_fragments_from_two_successful_rollouts(config, proxy):
    OmegaConf.update(config, 'tropic.successful_fragments_only', True, force_add=True)
    config.custom_envs.CoordSokoban.max_actions_per_traj = 7
    config.agent_proxy.max_turn = 7
    for entry in proxy.train_es_manager.envs:
        entry['max_actions_per_traj'] = 7
    proxy.scripts = {0: ['Up', 'Right', 'Left', 'Up', 'Right'],
                     1: ['Right', 'Up', 'Down', 'Up', 'Left', 'Up', 'Right']}
    collector = TropicCollector(config, proxy, score)
    try:
        _, metrics = collector.collect(1)
        graph = next(iter(collector.graphs.values()))
        assert metrics['tropic/root_verified'] == 2
        assert metrics['tropic/composition_new_solutions'] >= 1
        assert metrics['tropic/success_only_discarded_transitions'] == 0
        actions = {tuple(graph.edges[k].action for k in path) for path in graph.solutions}
        assert (4, 1, 3, 1, 4) in actions  
        assert all(collector.verify(graph, path) for path in graph.solutions)
    finally:
        collector.close()


def test_success_only_memory_survives_resume_and_rejects_other_retention_modes(config, proxy):
    OmegaConf.update(config, 'tropic.successful_fragments_only', True, force_add=True)
    config.es_manager.train.seed_pool_size = 1
    config.tropic.waves_per_iteration = 2
    collector = TropicCollector(config, proxy, score)
    restored = TropicCollector(config, proxy, score)
    try:
        collector.collect(1)
        snapshot = json.loads(json.dumps(collector.state_dict()))
        restored.load_state_dict(snapshot)
        assert restored.collect(2) == collector.collect(2)
        assert restored.state_dict() == collector.state_dict()
        with pytest.raises(ValueError, match='retention policy'):
            restored.load_state_dict({**snapshot, 'successful_fragments_only': False})
        restored.successful_fragments_only = False
        with pytest.raises(ValueError, match='retention policy'):
            restored.load_state_dict(snapshot)
    finally:
        collector.close()
        restored.close()


def test_success_only_verifies_and_retains_successful_frontier_continuations(config, proxy, monkeypatch):
    OmegaConf.update(config, 'tropic.successful_fragments_only', True, force_add=True)
    config.es_manager.train.seed_pool_size = 1
    config.tropic.waves_per_iteration = 2
    collector = TropicCollector(config, proxy, score)
    try:
        graph = collector._graph(config.seed.train)
        monkeypatch.setattr(graph, 'choose_frontier', lambda *args: next(iter(graph.solutions))[:2])
        bases, metrics = collector.collect(1)
        assert metrics['tropic/restart_attempts'] == 2
        assert metrics['tropic/restart_verified'] == 1
        assert metrics['tropic/success_only_verified_rollouts'] == 2
        assert all(collector.verify(graph, path) for path in bases[graph.problem_id])
    finally:
        collector.close()
