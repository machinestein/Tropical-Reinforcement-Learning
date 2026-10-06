"""Deterministic FrozenLake adapter; the original baseline environment is unchanged."""

from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv as GymFrozenLakeEnv
import numpy as np

from .env import FrozenLakeEnv
from .state import FrozenLakeSnapshot
from .tropic_config import FrozenLakeTropicEnvConfig


class FrozenLakeTropicEnv(FrozenLakeEnv):
    def __init__(self, config=None):
        config = config if config is not None else FrozenLakeTropicEnvConfig()
        config.__post_init__()
        super().__init__(config)
        self.num_env_steps, self._done, self._ready = 0, False, False

    def reset(self, seed=None, mode=None):
        super().reset(seed=seed, mode=mode)
        self.num_env_steps, self._done, self._ready = 0, False, True
        return self.render()

    def step(self, action):
        if not self._ready or self._done:
            raise RuntimeError("FrozenLake TROPIC cannot act before reset or after termination")
        if action not in self.action_map:
            self.num_env_steps += 1
            self._done = True
            return self.render(), 0.0, True, {
                "action_is_valid": False, "action_is_effective": False, "success": False}
        observation, reward, done, info = super().step(action)
        self.num_env_steps += 1
        self._done = bool(done or self.num_env_steps >= self.config.max_steps)
        info["success"] = bool(info["success"])
        return observation, float(reward), self._done, info

    def get_state(self):
        if not self._ready:
            raise RuntimeError("Reset FrozenLake before taking a snapshot")
        desc = [b"".join(row).decode("ascii") for row in self.desc]
        return FrozenLakeSnapshot(desc, int(self.s), int(self.num_env_steps),
                                  int(self.config.max_steps), self._done)

    def set_state(self, snapshot):
        snapshot = FrozenLakeSnapshot.from_dict(snapshot) if isinstance(snapshot, dict) else snapshot
        size = self.config.size
        if (len(snapshot.desc) != size or any(len(row) != size for row in snapshot.desc)
                or any(tile not in "SFHG" for row in snapshot.desc for tile in row)
                or sum(row.count("S") for row in snapshot.desc) != 1
                or sum(row.count("G") for row in snapshot.desc) != 1
                or snapshot.max_steps != self.config.max_steps
                or not 0 <= snapshot.position < size * size
                or not 0 <= snapshot.num_env_steps <= snapshot.max_steps):
            raise ValueError("Invalid or incompatible FrozenLake snapshot")
        tile = snapshot.desc[snapshot.position // size][snapshot.position % size]
        if not snapshot.done and (tile in "HG" or snapshot.num_env_steps == snapshot.max_steps):
            raise ValueError("Terminal FrozenLake snapshot must be marked done")
        desc = np.asarray([list(row) for row in snapshot.desc], dtype="c")
        if not np.array_equal(self.desc, desc):
            GymFrozenLakeEnv.__init__(self, desc=desc, is_slippery=self.config.is_slippery,
                                      render_mode=self.config.render_mode, success_rate=1.0)
        self.s, self.lastaction = int(snapshot.position), None
        self.num_env_steps = int(snapshot.num_env_steps)
        self._done, self._ready = bool(snapshot.done), True
        return self.render()
