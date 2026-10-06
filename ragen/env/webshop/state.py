"""Portable shopping prefixes with session-independent candidate join keys."""

from dataclasses import asdict, dataclass
import hashlib
import json
from numbers import Integral, Real


def canonical_data(value):
    """Convert simulator dictionaries, sets and NumPy scalars to portable JSON."""
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("WebShop snapshot dictionaries require string keys")
        return {key: canonical_data(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted((canonical_data(item) for item in value), key=canonical_json)
    if isinstance(value, (list, tuple)):
        return [canonical_data(item) for item in value]
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        return float(value)
    raise TypeError(f"Unsupported WebShop snapshot value: {type(value).__name__}")


def canonical_json(value):
    return json.dumps(canonical_data(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


@dataclass
class WebShopSnapshot:
    catalog_id: str
    root_seed: int
    mode: str
    actions: list[str]
    state: dict
    num_env_steps: int
    max_steps: int

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**value)

    def key(self, problem, budget):
        session = {key: value for key, value in self.state["session"].items() if key != "actions"}
        state = {**self.state, "session": session}
        return fingerprint([problem, self.catalog_id, self.mode, state,
                            self.num_env_steps, self.max_steps, budget])
