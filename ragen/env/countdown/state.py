"""Exact multiset-of-values Countdown snapshots; commuting operation orders share a key."""

from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import json


@dataclass
class CountdownSnapshot:
    target: int
    nums: list          
    remaining: list     
    num_env_steps: int
    max_steps: int

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**value)

    def key(self, problem, budget):
        remaining = [str(v) for v in sorted(Fraction(v) for v in self.remaining)]
        fields = [problem, self.target, remaining, self.num_env_steps, self.max_steps, budget]
        return hashlib.sha256(json.dumps(fields, separators=(",", ":")).encode()).hexdigest()
