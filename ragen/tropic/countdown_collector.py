"""Countdown fragments are exact value multisets; only executable operations enter the graph."""

from ragen.env.countdown.tropic_env import CountdownStepEnv
from .collector import TropicCollector


class CountdownTropicCollector(TropicCollector):
    environment_class = CountdownStepEnv
    normalize_action = staticmethod(str)

    def _valid_transition(self, info):
        return bool(info.get("action_is_valid", False))

    def _prepare_start(self, graph, path):
        if path:
            node = graph.nodes[graph.edges[path[-1]].target]
            if len(node.snapshot["remaining"]) < 2:
                self.metrics["terminal_restart_fallbacks"] += 1
                path = ()
        return super()._prepare_start(graph, path)
