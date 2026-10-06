"""Multi-operation Countdown for TROPIC: one exact binary operation per action over live values."""

from fractions import Fraction
import re

from .env import CountdownEnv
from .state import CountdownSnapshot

OPERATORS = {"+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b,
             "/": lambda a, b: None if b == 0 else a / b}
_ACTION = re.compile(r"^\s*(-?\d+(?:/\d+)?)\s*([+\-*/x×÷])\s*(-?\d+(?:/\d+)?)\s*$")
_ALIASES = {"x": "*", "×": "*", "÷": "/"}


def fmt(value):
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


class CountdownStepEnv(CountdownEnv):
    def __init__(self, config=None):
        super().__init__(config)
        self.target, self.nums, self.remaining = None, None, []
        self.num_env_steps, self._done, self._success = 0, False, False
        if int(getattr(self.config, "max_steps", 1)) < 1:
            raise ValueError("Countdown step environment needs a positive max_steps")

    def reset(self, seed=None, mode=None):
        super().reset(seed=seed, mode=mode)
        data = self.data[self.index]
        self.target, self.nums = int(data["target"]), [int(v) for v in data["nums"]]
        self.remaining = sorted(Fraction(v) for v in self.nums)
        self.num_env_steps, self._done, self._success = 0, False, False
        return self.render()

    def parse_action(self, action):
        match = _ACTION.match(str(action))
        if match is None:
            return None
        a, op, b = match.groups()
        try:
            return Fraction(a), _ALIASES.get(op, op), Fraction(b)
        except (ValueError, ZeroDivisionError):
            return None

    def step(self, action):
        if self.target is None or self._done:
            raise RuntimeError("Countdown step environment cannot act before reset or after termination")
        parsed = self.parse_action(action)
        result = None
        if parsed is not None:
            a, op, b = parsed
            live = list(self.remaining)
            if a in live:
                live.remove(a)
                if b in live:
                    live.remove(b)
                    result = OPERATORS[op](a, b)
        self.num_env_steps += 1
        if result is None:
            self._done = True
            return self.render(), 0.0, True, {"action_is_valid": False, "action_is_effective": False, "success": False}
        self.remaining = sorted(live + [result])
        self._success = len(self.remaining) == 1 and self.remaining[0] == self.target
        self._done = self._success or len(self.remaining) == 1 or self.num_env_steps >= int(self.config.max_steps)
        reward = float(self.config.score) if self._success else 0.0
        return self.render(), reward, self._done, {"action_is_valid": True, "action_is_effective": True,
                                                   "success": self._success}

    def render(self):
        if self.target is None:
            return "Countdown environment not initialised."
        return (f"Target: {self.target}\nNumbers: [{', '.join(fmt(v) for v in sorted(self.remaining))}]\n"
                f"Operations left before every number is used: {len(self.remaining) - 1}")

    def get_state(self):
        if self.target is None:
            raise RuntimeError("Reset Countdown before taking a snapshot")
        return CountdownSnapshot(self.target, list(self.nums), [fmt(v) for v in sorted(self.remaining)],
                                 int(self.num_env_steps), int(self.config.max_steps))

    def set_state(self, snapshot):
        snapshot = CountdownSnapshot.from_dict(snapshot) if isinstance(snapshot, dict) else snapshot
        remaining = sorted(Fraction(v) for v in snapshot.remaining)
        if (snapshot.max_steps != int(self.config.max_steps) or not remaining
                or len(snapshot.nums) - len(remaining) != snapshot.num_env_steps
                or not 0 <= snapshot.num_env_steps <= snapshot.max_steps):
            raise ValueError("Invalid or incompatible Countdown snapshot")
        self.target, self.nums, self.remaining = int(snapshot.target), [int(v) for v in snapshot.nums], remaining
        self.num_env_steps = int(snapshot.num_env_steps)
        self._success = len(remaining) == 1 and remaining[0] == self.target
        self._done = self._success or len(remaining) == 1 or self.num_env_steps >= snapshot.max_steps
        self.render_cache = self.render()
        return self.render_cache
