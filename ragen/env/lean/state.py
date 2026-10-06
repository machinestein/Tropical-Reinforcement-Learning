"""Portable proof prefixes; printed goals are candidate join keys, not Lean states."""

from dataclasses import asdict, dataclass
import hashlib
import json


@dataclass
class LeanSnapshot:
    theorem: dict
    tactics: list[str]
    goals: list[str]
    num_env_steps: int
    success: bool
    max_steps: int

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**value)

    def key(self, problem, budget):
        fields = [problem, self.theorem, self.goals, self.num_env_steps,
                  self.success, self.max_steps, budget]
        return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()
