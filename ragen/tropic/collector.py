"""Sokoban collection using RAGEN's generation and environment interfaces."""

from collections import defaultdict
import copy
import math

from verl import DataProto
from ragen.llm_agent.phi import is_response_end
from ragen.env.sokoban.env import SokobanEnv
from .graph import Edge, FragmentGraph, Node


class TropicCollector:
    environment_class = SokobanEnv
    graph_class = FragmentGraph
    normalize_action = staticmethod(int)

    def __init__(self, config, proxy, score_edges):
        self.config = config
        self.options = config.tropic
        self.successful_fragments_only = self.options.get('successful_fragments_only', False)
        self.proxy = proxy
        self.score_edges = score_edges
        manager = proxy.train_es_manager
        entry = manager.envs[0]
        if any(not isinstance(e['env'], self.environment_class) or e['tag'] != entry['tag'] for e in manager.envs):
            raise ValueError(f"TROPIC collector requires one {self.environment_class.__name__} task configuration")
        self.budget = entry['max_actions_per_traj']
        self.verifier = self.environment_class(copy.deepcopy(entry['config']))
        self.graphs = {}
        self.cursor = 0
        self.metrics = defaultdict(float)
        self.events = []
        self.root_diagnostics_batch = None

    def _node(self, problem, snapshot, observation):
        return Node(snapshot.key(problem, self.budget), snapshot.num_env_steps, snapshot.to_dict(), observation)

    def _graph(self, seed):
        problem = str(seed)
        if problem not in self.graphs:
            self.verifier.reset(seed=seed)
            root = self._node(problem, self.verifier.get_state(), self.verifier.render())
            self.graphs[problem] = self.graph_class(problem, root, self.options.max_edges_per_problem,
                                                  self.options.max_solutions_per_problem, self.options.top_l)
        return self.graphs[problem]

    def history(self, graph, path):
        history = [{"state": graph.nodes[graph.root].observation, "actions_left": self.budget}]
        for key in path:
            edge = graph.edges[key]
            target = graph.nodes[edge.target]
            history[-1].update(actions=[edge.action], reward=edge.reward,
                               info={"success": edge.success, "action_is_valid": True},
                               llm_response=edge.response, llm_raw_response=edge.raw_response)
            history.append({"state": target.observation, "actions_left": self.budget - target.depth})
        return history

    def verify(self, graph, path, require_success=True):
        self.metrics["verification_calls"] += 1
        self.verifier.set_state(graph.nodes[graph.root].snapshot)
        ctx = self.proxy.train_ctx_manager
        prefix = []
        success = False
        for i, key in enumerate(path):
            edge = graph.edges[key]
            if self.verifier.get_state().key(graph.problem_id, self.budget) != edge.source:
                return False
            state = {"env_id": 0, "group_id": 0, "history": self.history(graph, prefix)}
            prompt = ctx.get_lm_inputs([state], prepare_for_update=False)
            ids = prompt.batch['input_ids'][0][prompt.batch['attention_mask'][0].bool()].tolist()
            if tuple(ids) != tuple(edge.prompt_ids):
                return False
            _, actions = ctx._parse_response(edge.raw_response)
            mapped = self.proxy.train_es_manager._extract_map_valid_actions(
                self.proxy.train_es_manager.envs[0], actions)
            if mapped != [edge.action]:
                return False
            _, reward, done, info = self.verifier.step(edge.action)
            self.metrics["verification_env_steps"] += 1
            if not self._valid_transition(info):
                return False
            if self.verifier.get_state().key(graph.problem_id, self.budget) != edge.target:
                return False
            if not math.isclose(reward, edge.reward, abs_tol=1e-8):
                return False
            success = bool(info['success'])
            if success != edge.success or (done and i != len(path) - 1):
                return False
            prefix.append(key)
        return bool(path) and (success or not require_success) and len(path) <= self.budget

    def _valid_transition(self, info):
        return True

    def _prepare_start(self, graph, path):
        node = graph.nodes[graph.edges[path[-1]].target if path else graph.root]
        return path, node.snapshot

    def _rescore(self, graphs, version, only_new=False):
        rows = [edge for graph in graphs for edge in graph.edges.values()
                if not only_new or edge.policy_version != version]
        if not rows:
            return
        scores = self.score_edges(rows)
        if len(scores) != len(rows) or not all(math.isfinite(float(s)) for s in scores):
            raise ValueError("Fragment rescoring returned missing or non-finite likelihoods")
        for edge, score in zip(rows, scores):
            edge.log_prob, edge.policy_version = float(score), version
        self.metrics['rescored_edges'] += len(rows)
        self.metrics['rescored_completion_tokens'] += sum(len(e.completion_ids) for e in rows)
        self.metrics['rescored_input_tokens'] += sum(len(e.prompt_ids) + len(e.completion_ids) for e in rows)

    def _record(self, lm_inputs, lm_outputs, env_inputs, before, manager):
        if lm_outputs.batch is None or 'responses' not in lm_outputs.batch:
            raise ValueError("TROPIC requires exact generated token IDs; text-only wrappers are unsupported")
        for row, item in enumerate(env_inputs):
            env_id = item['env_id']
            entry = manager.envs[env_id]
            graph = self.active_graphs[env_id]
            response_ids = lm_outputs.batch['responses'][row]
            response_mask = lm_outputs.batch['attention_mask'][row, -len(response_ids):].bool()
            completion = tuple(response_ids[response_mask].tolist())
            prompt_ids = lm_inputs.batch['input_ids'][row][lm_inputs.batch['attention_mask'][row].bool()]
            self.metrics['generated_tokens'] += len(completion)
            self.metrics['generation_prompt_tokens'] += len(prompt_ids)
            self.metrics['model_decisions'] += 1
            turn = manager.rollout_cache[env_id]['history'][-2]
            if len(turn.get('actions', [])) != 1 or not completion or not self._valid_transition(turn.get('info', {})):
                self.metrics['invalid_decisions'] += 1
                self.active_paths[env_id] = None
                continue
            snapshot = entry['env'].get_state()
            source = self._node(graph.problem_id, before[env_id], turn['state'])
            target = self._node(graph.problem_id, snapshot, entry['env'].render())
            behavior = None
            if 'rollout_log_probs' in lm_outputs.batch:
                behavior = float(lm_outputs.batch['rollout_log_probs'][row][response_mask].sum())
            edge = Edge(source.key, target.key, self.normalize_action(turn['actions'][0]), tuple(prompt_ids.tolist()),
                        completion, turn['llm_response'], turn['llm_raw_response'], float(turn['reward']),
                        bool(turn['info'].get('success', False)),
                        [self.rollout_ids[env_id]], [behavior],
                        stop_reason='eos' if is_response_end(self.proxy.tokenizer, completion[-1]) else 'length')
            if self.successful_fragments_only:
                self.staged_transitions[env_id].append((source, target, edge))
                self.metrics['success_only_buffered_transitions'] += 1
                key = edge.key
            else:
                key = graph.add_edge(source, target, edge)
            self.metrics['sampled_env_steps'] += 1
            if key is None:
                self.active_paths[env_id] = None
            elif self.active_paths[env_id] is not None:
                self.active_paths[env_id].append(key)
                if edge.success:
                    pending = (graph, tuple(self.active_paths[env_id]), self.origins[env_id])
                    if self.successful_fragments_only:
                        self.staged_successes.append((*pending, env_id))
                    else:
                        self.pending.append(pending)

    def _admit_successful_rollouts(self):
        """Commit complete, root-verified sampled successes before rescoring/joins.

        A temporary graph isolates both new edges and provenance updates on shared
        edges. A failed replay or capacity rejection leaves persistent memory intact.
        """
        admitted = 0
        for graph, path, origin, env_id in self.staged_successes:
            transitions = self.staged_transitions[env_id]
            new_keys = {edge.key for _, _, edge in transitions} - graph.edges.keys()
            if len(graph.edges) + len(new_keys) > graph.max_edges:
                graph.rejected_edges += len(new_keys)
                self.metrics['success_only_capacity_rejected_rollouts'] += 1
                continue
            trial = copy.copy(graph)
            trial.nodes = dict(graph.nodes)
            trial.edges = dict(graph.edges)
            for source, target, edge in transitions:
                key = edge.key
                if key in graph.edges and trial.edges[key] is graph.edges[key]:
                    trial.edges[key] = copy.deepcopy(graph.edges[key])
                trial.add_edge(source, target, edge)
            if not self.verify(trial, path):
                self.metrics['failed_verifications'] += 1
                self.metrics['success_only_failed_verifications'] += 1
                continue
            graph.nodes, graph.edges = trial.nodes, trial.edges
            self.pending.append((graph, path, origin))
            admitted += len(transitions)
            self.metrics['success_only_verified_rollouts'] += 1
        buffered = sum(len(rows) for rows in self.staged_transitions.values())
        self.metrics['success_only_admitted_transitions'] += admitted
        self.metrics['success_only_discarded_transitions'] += buffered - admitted
        self.staged_transitions.clear()
        self.staged_successes.clear()

    def _certify(self, graph, path, origin):
        if not self.verify(graph, path):
            self.metrics['failed_verifications'] += 1
            if origin == 'composition':
                graph.reject_join(path)
            return False
        return self._archive_verified(graph, path, origin)

    def _archive_verified(self, graph, path, origin):
        """Archive a path after its exact root replay has passed."""
        self.metrics[origin + '_verified'] += 1
        fresh = graph.certify(path, origin)
        if fresh:
            self.metrics[origin + '_new_solutions'] += 1
        origins = sorted({graph.edges[k].rollout_ids[0] for k in path})
        self.events.append({"problem": graph.problem_id, "origin": origin, "path": list(path),
                            "actions": [graph.edges[k].action for k in path],
                            "new_solution": fresh, "archived": tuple(path) in graph.solutions,
                            "edge_origins": origins})
        return fresh

    def collect(self, iteration):
        self.metrics = defaultdict(float)
        self.events = []
        self.root_diagnostics_batch = None
        manager = self.proxy.train_es_manager
        pool = int(manager.config.seed_pool_size)
        base_seed = int(self.config.seed.train)
        seeds = [base_seed + (self.cursor + i) % pool for i in range(manager.env_groups)]
        self.cursor = (self.cursor + manager.env_groups) % pool
        graphs = [self._graph(seed) for seed in seeds]
        self._rescore(graphs, iteration)
        for wave in range(self.options.waves_per_iteration):
            starts, self.active_graphs, self.active_paths, self.origins, self.rollout_ids = [], {}, {}, {}, {}
            self.pending = []
            if self.successful_fragments_only:
                self.staged_transitions = defaultdict(list)
                self.staged_successes = []
            for group, (seed, graph) in enumerate(zip(seeds, graphs)):
                path = graph.choose_frontier(self.budget, self.options.frontier_eta) if wave and self.options.frontier_enabled else ()
                path, snapshot = self._prepare_start(graph, path)
                origin = 'restart' if path else 'root'
                for sample in range(manager.group_size):
                    env_id = group * manager.group_size + sample
                    starts.append({"seed": seed, "snapshot": snapshot, "history": self.history(graph, path)})
                    self.active_graphs[env_id] = graph
                    self.active_paths[env_id] = list(path)
                    self.origins[env_id] = origin
                    self.rollout_ids[env_id] = f"{iteration}:{wave}:{env_id}"
                    self.metrics[origin + '_attempts'] += 1
            freq = self.config.collapse_detection.compute_freq
            diagnose = wave == 0 and (iteration == 1 or iteration % freq == 0)
            request = DataProto(meta_info={"do_sample": True, "validate": False, "compute_collapse": diagnose})
            self.proxy.rollout(request, starts=starts, transition_recorder=self._record, return_states=True)
            if wave == 0:
                self.root_diagnostics_batch = self.proxy.root_diagnostics_batch
            if self.successful_fragments_only:
                self._admit_successful_rollouts()
            self._rescore(graphs, iteration, only_new=True)
            for graph, path, origin in self.pending:
                if self.successful_fragments_only:
                    self._archive_verified(graph, path, origin)
                else:
                    self._certify(graph, path, origin)
            if self.options.composition_enabled:
                for graph in graphs:
                    for path in graph.candidates(self.options.max_candidates_per_refresh):
                        self.metrics['composition_candidates'] += 1
                        self._certify(graph, path, 'composition')
        bases = {g.problem_id: g.select_basis(self.options.basis_size, self.options.coverage_enabled) for g in graphs}
        self.metrics.update({
            "positive_problems": sum(bool(p) for p in bases.values()),
            "selected_paths": sum(len(p) for p in bases.values()),
            "retained_edges": sum(len(g.edges) for g in self.graphs.values()),
            "retained_solutions": sum(len(g.solutions) for g in self.graphs.values()),
            "rejected_edges": sum(g.rejected_edges for g in self.graphs.values()),
        })
        return bases, {"tropic/" + k: v for k, v in self.metrics.items()}

    def state_dict(self):
        state = {"version": 1, "cursor": self.cursor, "graphs": [g.to_dict() for g in self.graphs.values()]}
        if self.successful_fragments_only:
            state['successful_fragments_only'] = True
        return state

    def load_state_dict(self, state):
        if state['version'] != 1:
            raise ValueError("Unsupported TROPIC checkpoint version")
        if state.get('successful_fragments_only', False) != self.successful_fragments_only:
            raise ValueError("TROPIC checkpoint fragment-retention policy differs; start a fresh run")
        self.cursor = state['cursor']
        self.graphs = {g['problem_id']: self.graph_class.from_dict(g) for g in state['graphs']}

    def close(self):
        self.verifier.close()
