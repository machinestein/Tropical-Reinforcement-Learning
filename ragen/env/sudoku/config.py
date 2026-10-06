from dataclasses import dataclass, field
from typing import Optional

@dataclass
class SudokuEnvConfig:
    """Configuration for Sudoku environment with enhanced feedback."""
    grid_size: int = 9  
    max_steps: int = 81  
    difficulty: str = "easy"  
    render_mode: str = "text"
    show_conflicts: bool = True  
    show_valid_numbers: bool = True  
    show_candidates: bool = False  
    render_format: str = "detailed"  

    correct_placement_score: float = 1.0
    invalid_action_score: float = -0.1  
    completion_bonus: float = 10.0  

    def __post_init__(self):
        if self.grid_size not in {4, 9, 16}:
            raise ValueError(f"Unsupported grid_size: {self.grid_size}. Must be 4, 9, or 16.")
        if self.render_format not in {"simple", "detailed", "with_feedback"}:
            raise ValueError(f"Unsupported render_format: {self.render_format}")
        if self.difficulty not in {"easy", "medium", "hard"}:
            raise ValueError(f"Unsupported difficulty: {self.difficulty}")
