"""Only root-replayed, perfect-score WebShop purchases become TROPIC targets."""

from ragen.env.webshop.tropic_env import WebShopTropicEnv
from .collector import TropicCollector


class WebShopTropicCollector(TropicCollector):
    environment_class = WebShopTropicEnv
    normalize_action = staticmethod(str)

    def collect(self, iteration):
        bases, metrics = super().collect(iteration)
        lengths, repeated_clicks, repeated_paths = [], 0, 0
        for problem, paths in bases.items():
            graph = self.graphs[problem]
            for path in paths:
                repeats = 0
                for key in path:
                    edge = graph.edges[key]
                    options = graph.nodes[edge.source].snapshot["state"]["session"]["options"]
                    repeats += edge.action.lower() in {f"click[{str(v).lower()}]" for v in options.values()}
                lengths.append(len(path))
                repeated_clicks += repeats
                repeated_paths += bool(repeats)
        metrics.update({
            "tropic/selected_path_length_mean": sum(lengths) / len(lengths) if lengths else 0.0,
            "tropic/selected_repeated_option_action_fraction": repeated_clicks / sum(lengths) if lengths else 0.0,
            "tropic/selected_repeated_option_path_fraction": repeated_paths / len(lengths) if lengths else 0.0,
            "tropic/archive_full_problems": sum(len(g.solutions) == g.max_solutions for g in self.graphs.values()),
        })
        return bases, metrics

    def _valid_transition(self, info):
        return bool(info.get("action_is_valid", False))

    def _prepare_start(self, graph, path):
        while path and graph.nodes[graph.edges[path[-1]].target].snapshot["state"]["done"]:
            path = path[:-1]
            self.metrics["terminal_frontiers_trimmed"] += 1
        if path:
            if self.verify(graph, path, require_success=False):
                return path, self.verifier.get_state().to_dict()
            self.metrics["rejected_frontiers"] += 1
        return (), graph.nodes[graph.root].snapshot
