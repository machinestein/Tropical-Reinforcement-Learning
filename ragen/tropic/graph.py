"""Bounded, time-unrolled fragment graphs with explicit path provenance."""

from collections import defaultdict
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math


@dataclass
class Node:
    key: str
    depth: int
    snapshot: dict
    observation: str
    prompt_ids: tuple = ()


@dataclass
class Edge:
    source: str
    target: str
    action: int | str
    prompt_ids: tuple
    completion_ids: tuple
    response: str
    raw_response: str
    reward: float
    success: bool
    rollout_ids: list = field(default_factory=list)
    behavior_log_probs: list = field(default_factory=list)
    log_prob: float | None = None
    policy_version: int = -1
    stop_reason: str = "unknown"

    @property
    def key(self):
        fields = [self.source, self.target, self.action, self.prompt_ids, self.completion_ids]
        return hashlib.sha256(json.dumps(fields, separators=(",", ":")).encode()).hexdigest()


class FragmentGraph:
    def __init__(self, problem_id, root, max_edges=8192, max_solutions=64, top_l=4):
        if min(max_edges, max_solutions, top_l) < 1:
            raise ValueError("Graph limits must be positive")
        self.problem_id = problem_id
        self.root = root.key
        self.nodes = {root.key: root}
        self.edges = {}
        self.solutions = {}
        self.rehearsals = {}
        self.frontier_visits = {}
        self.depth_visits = {}
        self.max_edges = max_edges
        self.max_solutions = max_solutions
        self.top_l = top_l
        self.rejected_edges = 0
        self.failed_joins = set()

    def add_edge(self, source, target, edge):
        if edge.source != source.key or edge.target != target.key or target.depth != source.depth + 1:
            raise ValueError("A fragment must advance exactly one decision in the DAG")
        if not edge.prompt_ids or not edge.completion_ids:
            raise ValueError("Fragments require exact prompt and completion tokens")
        existing = self.nodes.get(source.key)
        if existing and existing.prompt_ids and tuple(existing.prompt_ids) != tuple(edge.prompt_ids):
            raise ValueError("The same graph state produced different policy prompts")
        key = edge.key
        if key in self.edges:
            retained = self.edges[key]
            for origin, score in zip(edge.rollout_ids, edge.behavior_log_probs):
                if origin not in retained.rollout_ids and len(retained.rollout_ids) < 8:
                    retained.rollout_ids.append(origin)
                    retained.behavior_log_probs.append(score)
            return key
        if len(self.edges) >= self.max_edges:
            self.rejected_edges += 1
            return None
        source.prompt_ids = tuple(edge.prompt_ids)
        self.nodes[source.key] = source
        self.nodes.setdefault(target.key, target)
        self.edges[key] = edge
        return key

    def score(self, path):
        scores = [self.edges[k].log_prob for k in path]
        return sum(scores) if all(v is not None and math.isfinite(v) for v in scores) else -math.inf

    def _top(self, paths):
        return sorted(set(paths), key=lambda p: (-self.score(p), p))[:self.top_l]

    def paths(self):
        outgoing = defaultdict(list)
        for key, edge in self.edges.items():
            if edge.log_prob is not None and math.isfinite(edge.log_prob):
                outgoing[edge.source].append(key)
        ordered = sorted(self.nodes, key=lambda k: (self.nodes[k].depth, k))
        prefixes = {self.root: [()]}
        for state in ordered:
            for key in outgoing[state]:
                target = self.edges[key].target
                options = prefixes.get(target, []) + [p + (key,) for p in prefixes.get(state, [])]
                prefixes[target] = self._top(options)
        terminals = {self.edges[p[-1]].target for p in self.solutions if p}
        suffixes = {s: [()] for s in terminals}
        for state in reversed(ordered):
            options = suffixes.get(state, [])
            for key in outgoing[state]:
                options = options + [(key,) + p for p in suffixes.get(self.edges[key].target, [])]
            if options:
                suffixes[state] = self._top(options)
        return prefixes, suffixes

    def candidates(self, limit=64):
        prefixes, suffixes = self.paths()
        certified_edges = {e for path in self.solutions for e in path}
        joined = {p + q for state in prefixes for p in prefixes[state]
                  for q in suffixes.get(state, []) if p and q}
        joined.difference_update(self.solutions)
        joined = {p for p in joined if self.path_key(p) not in self.failed_joins}
        return sorted(joined, key=lambda p: (-len(set(p) - certified_edges), -self.score(p), p))[:limit]

    @staticmethod
    def path_key(path):
        return hashlib.sha256(json.dumps(list(path)).encode()).hexdigest()

    def reject_join(self, path):
        if len(self.failed_joins) < self.max_edges:
            self.failed_joins.add(self.path_key(path))

    def certify(self, path, origin):
        """Call only after root replay and terminal verification."""
        path = tuple(path)
        current = self.root
        for i, key in enumerate(path):
            edge = self.edges[key]
            if edge.source != current or (edge.success and i != len(path) - 1):
                raise ValueError("Invalid complete path")
            current = edge.target
        if not path or not self.edges[path[-1]].success:
            raise ValueError("Only successful complete paths can be certified")
        signature = tuple(self.edges[k].action for k in path)
        novel = True
        for old in list(self.solutions):
            if tuple(self.edges[k].action for k in old) == signature:
                if old == path or self.score(old) >= self.score(path):
                    return False
                del self.solutions[old]
                novel = False
                break
        if len(self.solutions) >= self.max_solutions:
            return False
        self.solutions[path] = origin
        return novel

    def choose_frontier(self, budget, eta=1.0):
        prefixes, suffixes = self.paths()
        eligible = [s for s in prefixes if s != self.root and s not in suffixes
                    and self.nodes[s].depth < budget and prefixes[s]]
        if not eligible:
            return ()
        depth = min({self.nodes[s].depth for s in eligible}, key=lambda d: (self.depth_visits.get(d, 0), d))
        state = min((s for s in eligible if self.nodes[s].depth == depth),
                    key=lambda s: (-(self.score(prefixes[s][0]) - eta * math.log1p(self.frontier_visits.get(s, 0))), s))
        self.depth_visits[depth] = self.depth_visits.get(depth, 0) + 1
        self.frontier_visits[state] = self.frontier_visits.get(state, 0) + 1
        return prefixes[state][0]

    def fragment_ids(self, path):
        return {(self.edges[k].source, self.edges[k].action) for k in path}

    def select_basis(self, size=2, coverage=True):
        remaining = sorted(self.solutions, key=lambda p: (-self.score(p), p))
        selected, covered = [], set()
        while remaining and len(selected) < size:
            if selected and coverage:
                def priority(path):
                    gain = sum(1 / math.sqrt(1 + self.rehearsals.get(f, 0))
                               for f in self.fragment_ids(path) - covered)
                    return (-gain, -self.score(path), path)
                remaining.sort(key=priority)
            path = remaining.pop(0)
            selected.append(path)
            covered.update(self.fragment_ids(path))
        for fragment in covered:
            self.rehearsals[fragment] = self.rehearsals.get(fragment, 0) + 1
        return selected

    def to_dict(self):
        return {
            "problem_id": self.problem_id, "root": self.root,
            "nodes": [asdict(n) for n in self.nodes.values()],
            "edges": [asdict(e) for e in self.edges.values()],
            "solutions": [[list(p), origin] for p, origin in self.solutions.items()],
            "rehearsals": [[list(k), v] for k, v in self.rehearsals.items()],
            "frontier_visits": self.frontier_visits, "depth_visits": self.depth_visits,
            "max_edges": self.max_edges, "max_solutions": self.max_solutions,
            "top_l": self.top_l, "rejected_edges": self.rejected_edges,
            "failed_joins": sorted(self.failed_joins),
        }

    @classmethod
    def from_dict(cls, data):
        nodes = {n["key"]: Node(**{**n, "prompt_ids": tuple(n["prompt_ids"])}) for n in data["nodes"]}
        graph = cls(data["problem_id"], nodes[data["root"]], data["max_edges"], data["max_solutions"], data["top_l"])
        graph.nodes = nodes
        for raw in data["edges"]:
            edge = Edge(**{**raw, "prompt_ids": tuple(raw["prompt_ids"]), "completion_ids": tuple(raw["completion_ids"])})
            graph.edges[edge.key] = edge
        graph.solutions = {tuple(p): origin for p, origin in data["solutions"]}
        graph.rehearsals = {tuple(k): v for k, v in data["rehearsals"]}
        graph.frontier_visits = data["frontier_visits"]
        graph.depth_visits = {int(k): v for k, v in data["depth_visits"].items()}
        graph.rejected_edges = data["rejected_edges"]
        graph.failed_joins = set(data.get("failed_joins", []))
        return graph
