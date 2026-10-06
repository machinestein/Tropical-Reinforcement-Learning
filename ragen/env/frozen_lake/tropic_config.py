"""Separate rules for deterministic, finite-horizon FrozenLake composition."""

from dataclasses import dataclass

from .config import FrozenLakeEnvConfig


@dataclass
class FrozenLakeTropicEnvConfig(FrozenLakeEnvConfig):
    success_rate: float = 1.0
    observation_format: str = "grid_coord"
    max_steps: int = 10

    def __post_init__(self):
        super().__post_init__()
        if self.success_rate != 1.0:
            raise ValueError("FrozenLake TROPIC requires deterministic movement (success_rate=1)")
        if self.render_mode != "text":
            raise ValueError("FrozenLake TROPIC requires text observations")
        if self.max_steps < 1:
            raise ValueError("FrozenLake TROPIC requires a positive max_steps")
