"""Lean goal joins are proposals: certify paths and restart prefixes by root replay."""

from ragen.env.lean.tropic_env import LeanTropicEnv
from .collector import TropicCollector


class LeanTropicCollector(TropicCollector):
    environment_class = LeanTropicEnv
    normalize_action = staticmethod(str)

    def _valid_transition(self, info):
        return bool(info.get("accepted", info.get("action_is_valid", False)))

    def _prepare_start(self, graph, path):
        if path:
            if self.verify(graph, path, require_success=False):
                return path, self.verifier.get_state().to_dict()
            self.metrics["rejected_frontiers"] += 1
        return (), graph.nodes[graph.root].snapshot
