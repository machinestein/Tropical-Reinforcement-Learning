"""Only root-replayed goal-reaching FrozenLake paths become training targets."""

from ragen.env.frozen_lake.tropic_env import FrozenLakeTropicEnv
from .collector import TropicCollector
from .frozen_lake_graph import FrozenLakeFragmentGraph


class FrozenLakeTropicCollector(TropicCollector):
    environment_class = FrozenLakeTropicEnv
    graph_class = FrozenLakeFragmentGraph

    def collect(self, iteration):
        before = sum(g.archive_replacements for g in self.graphs.values())
        bases, metrics = super().collect(iteration)
        lengths, revisits, walls = [], [], 0
        for problem, paths in bases.items():
            graph = self.graphs[problem]
            for path in paths:
                positions = [graph.nodes[graph.root].snapshot["position"]] + [
                    graph.nodes[graph.edges[k].target].snapshot["position"] for k in path]
                lengths.append(len(path))
                revisits.append(len(set(positions)) < len(positions))
                walls += sum(a == b for a, b in zip(positions, positions[1:]))
        metrics.update({
            "tropic/archive_replacements": sum(g.archive_replacements for g in self.graphs.values()) - before,
            "tropic/archive_full_problems": sum(len(g.solutions) == g.max_solutions for g in self.graphs.values()),
            "tropic/selected_path_length_mean": sum(lengths) / len(lengths) if lengths else 0.0,
            "tropic/selected_path_revisit_fraction": sum(revisits) / len(revisits) if revisits else 0.0,
            "tropic/selected_wall_action_fraction": walls / sum(lengths) if lengths else 0.0,
        })
        return bases, metrics

    def _valid_transition(self, info):
        return bool(info.get("action_is_valid", False))

    def _prepare_start(self, graph, path):
        while path and graph.nodes[graph.edges[path[-1]].target].snapshot["done"]:
            path = path[:-1]
            self.metrics["terminal_frontiers_trimmed"] += 1
        return super()._prepare_start(graph, path)
