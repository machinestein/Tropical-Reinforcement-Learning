"""Exact deterministic map, position and finite-horizon state; no path history."""

from dataclasses import asdict, dataclass
import hashlib
import json


@dataclass
class FrozenLakeSnapshot:
    desc: list[str]
    position: int
    num_env_steps: int
    max_steps: int
    done: bool

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**value)

    def key(self, problem, budget):
        fields = [problem, self.desc, self.position, self.num_env_steps,
                  self.max_steps, self.done, budget]
        return hashlib.sha256(json.dumps(fields, separators=(",", ":")).encode()).hexdigest()
