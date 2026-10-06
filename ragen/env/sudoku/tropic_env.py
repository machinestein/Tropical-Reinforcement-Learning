"""State-conditioned Sudoku for TROPIC: exact snapshots, pure renders, invalid placements terminate."""

import numpy as np

from .env import SudokuEnv
from .state import SudokuSnapshot
from .utils import format_grid_with_conflicts, is_solved


class SudokuTropicEnv(SudokuEnv):
    def __init__(self, config=None):
        super().__init__(config)
        self.num_env_steps = 0
        self._done = False

    def reset(self, seed=None, mode=None):
        super().reset(seed=seed, mode=mode)
        self.num_env_steps, self._done = 0, False
        self.last_action_feedback = ""
        return self.render()

    def step(self, action):
        if self.current_grid is None or self._done:
            raise RuntimeError("Sudoku TROPIC cannot act before reset or after episode termination")
        _, reward, done, info = super().step(action)
        self.num_env_steps = self.num_steps
        info = dict(info)
        info["success"] = bool(info["success"])
        self._done = bool(done or not info["action_is_valid"])
        return self.render(), reward, self._done, info

    def render(self):
        """A pure function of the snapshot: no last-action feedback, no hints."""
        if self.current_grid is None:
            return "Sudoku environment not initialised."
        empty = {"row": [], "col": [], "box": []}  
        grid = format_grid_with_conflicts(self.current_grid, self.initial_grid, empty)
        remaining = int(np.count_nonzero(self.current_grid == 0))
        return (f"Sudoku {self.grid_size}x{self.grid_size} (rows and columns are 1-indexed)\n{grid}\n"
                f"Legend: [N]=given cell, N=placed cell, .=empty\nEmpty cells remaining: {remaining}")

    def get_state(self):
        if self.current_grid is None:
            raise RuntimeError("Reset Sudoku before taking a snapshot")
        return SudokuSnapshot(self.initial_grid.tolist(), self.current_grid.tolist(), self.solution_grid.tolist(),
                              int(self.num_env_steps), int(self.max_steps))

    def set_state(self, snapshot):
        snapshot = SudokuSnapshot.from_dict(snapshot) if isinstance(snapshot, dict) else snapshot
        initial = np.asarray(snapshot.initial_grid, dtype=int)
        current = np.asarray(snapshot.current_grid, dtype=int)
        solution = np.asarray(snapshot.solution_grid, dtype=int)
        shape = (self.grid_size, self.grid_size)
        if (initial.shape != shape or current.shape != shape or solution.shape != shape
                or snapshot.max_steps != self.max_steps or not 0 <= snapshot.num_env_steps <= snapshot.max_steps
                or np.any((initial != 0) & (current != initial)) or np.any((solution == 0))
                or np.any((initial != 0) & (initial != solution))):
            raise ValueError("Invalid or incompatible Sudoku snapshot")
        self.initial_grid, self.current_grid, self.solution_grid = initial, current, solution
        self.num_steps = self.num_env_steps = int(snapshot.num_env_steps)
        self._done = bool(is_solved(current) or self.num_env_steps >= self.max_steps)
        self.last_action_feedback = ""
        return self.render()
