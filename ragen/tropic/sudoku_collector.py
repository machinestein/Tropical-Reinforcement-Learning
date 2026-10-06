"""Sudoku fragments are exact boards; only rule-valid placements enter the graph."""

from ragen.env.sudoku.tropic_env import SudokuTropicEnv
from .collector import TropicCollector


class SudokuTropicCollector(TropicCollector):
    environment_class = SudokuTropicEnv
    normalize_action = staticmethod(str)

    def _valid_transition(self, info):
        return bool(info.get("action_is_valid", False))
