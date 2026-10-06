"""Exact Sudoku snapshots; the join key is the visible board at a given depth."""

from dataclasses import asdict, dataclass
import hashlib
import json


@dataclass
class SudokuSnapshot:
    initial_grid: list
    current_grid: list
    solution_grid: list  
    num_env_steps: int
    max_steps: int

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**value)

    def key(self, problem, budget):
        fields = [problem, self.initial_grid, self.current_grid, self.num_env_steps, self.max_steps, budget]
        return hashlib.sha256(json.dumps(fields, separators=(",", ":")).encode()).hexdigest()
