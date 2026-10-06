import gym
from gym_sokoban.envs.sokoban_env import SokobanEnv as GymSokobanEnv
import numpy as np
from .utils import (
    generate_room,
    collect_entity_coordinates,
    format_coordinate_render,
)
from ragen.env.base import BaseDiscreteActionEnv
from ragen.env.sokoban.config import SokobanEnvConfig
from ragen.env.sokoban.state import SokobanSnapshot
from ragen.utils import all_seed

class SokobanEnv(BaseDiscreteActionEnv, GymSokobanEnv):
    def __init__(self, config=None, **kwargs):
        self.config = config or SokobanEnvConfig()
        self.GRID_LOOKUP = self.config.grid_lookup
        self.ACTION_LOOKUP = self.config.action_lookup
        self.search_depth = self.config.search_depth
        self.ACTION_SPACE = gym.spaces.discrete.Discrete(4, start=1)
        self.render_mode = self.config.render_mode
        self.observation_format = self.config.observation_format

        BaseDiscreteActionEnv.__init__(self)
        GymSokobanEnv.__init__(
            self,
            dim_room=self.config.dim_room, 
            max_steps=self.config.max_steps,
            num_boxes=self.config.num_boxes,
            **kwargs
        )

    def reset(self, seed=None, mode=None):
        try:
            with all_seed(seed):
                self.room_fixed, self.room_state, self.box_mapping, action_sequence = generate_room(
                    dim=self.dim_room,
                    num_steps=self.num_gen_steps,
                    num_boxes=self.num_boxes,
                    search_depth=self.search_depth
                )
            self.num_env_steps, self.reward_last, self.boxes_on_target = 0, 0, 0
            self.player_position = np.argwhere(self.room_state == 5)[0]
            self.old_box_position = self.new_box_position = None
            return self.render()
        except (RuntimeError, RuntimeWarning) as e:
            next_seed = abs(hash(str(seed))) % (2 ** 32) if seed is not None else None
            return self.reset(next_seed)
        
    def step(self, action: int):
        previous_pos = self.player_position
        _, reward, done, _ = GymSokobanEnv.step(self, action) 
        next_obs = self.render()
        action_effective = not np.array_equal(previous_pos, self.player_position)
        info = {"action_is_effective": action_effective, "action_is_valid": True, "success": self.boxes_on_target == self.num_boxes}
        return next_obs, reward, done, info

    def render(self, mode=None):
        if mode in {'grid', 'coord', 'grid_coord'}:
            return self._render_text(mode)

        render_mode = mode if mode is not None else self.render_mode
        if render_mode == 'text':
            return self._render_text(self.observation_format)
        if render_mode == 'rgb_array':
            return self.get_image(mode='rgb_array', scale=1)
        raise ValueError(f"Invalid mode: {render_mode}")

    def _render_text(self, observation_format: str) -> str:
        if observation_format == 'grid':
            room = np.where((self.room_state == 5) & (self.room_fixed == 2), 6, self.room_state)
            return '\n'.join(''.join(self.GRID_LOOKUP.get(cell, "?") for cell in row) for row in room.tolist())
        if observation_format == 'coord':
            entity_coords = collect_entity_coordinates(self.room_state, self.room_fixed)
            return format_coordinate_render(entity_coords, self.dim_room)
        if observation_format == 'grid_coord':
            entity_coords = collect_entity_coordinates(self.room_state, self.room_fixed)
            return "Coordinates: \n" + format_coordinate_render(entity_coords, self.dim_room) + "\n" + "Grid Map: \n" + self._render_text('grid')
        raise ValueError(f"Invalid observation_format: {observation_format}")
    
    def get_all_actions(self):
        return list([k for k in self.ACTION_LOOKUP.keys()])
    
    def get_state(self):
        def position(value):
            return None if value is None else [int(x) for x in value]

        return SokobanSnapshot(
            room_fixed=self.room_fixed.tolist(), room_state=self.room_state.tolist(),
            player_position=self.player_position.tolist(),
            box_mapping=[[position(k), position(v)] for k, v in self.box_mapping.items()],
            num_env_steps=int(self.num_env_steps), boxes_on_target=int(self.boxes_on_target),
            reward_last=float(self.reward_last),
            old_box_position=position(getattr(self, "old_box_position", None)),
            new_box_position=position(getattr(self, "new_box_position", None)),
            max_steps=int(self.max_steps), num_boxes=int(self.num_boxes),
        )

    def set_state(self, snapshot):
        if isinstance(snapshot, dict):
            snapshot = SokobanSnapshot(**snapshot)
        fixed = np.asarray(snapshot.room_fixed, dtype=int)
        state = np.asarray(snapshot.room_state, dtype=int)
        if fixed.shape != tuple(self.dim_room) or state.shape != fixed.shape:
            raise ValueError("Snapshot board dimensions do not match the environment")
        if snapshot.max_steps != self.max_steps or snapshot.num_boxes != self.num_boxes:
            raise ValueError("Snapshot rules do not match the environment")
        self.room_fixed, self.room_state = fixed.copy(), state.copy()
        self.player_position = np.asarray(snapshot.player_position, dtype=int).copy()
        self.box_mapping = {tuple(k): tuple(v) for k, v in snapshot.box_mapping}
        self.num_env_steps = snapshot.num_env_steps
        self.boxes_on_target = snapshot.boxes_on_target
        self.reward_last = snapshot.reward_last
        self.old_box_position = None if snapshot.old_box_position is None else tuple(snapshot.old_box_position)
        self.new_box_position = None if snapshot.new_box_position is None else tuple(snapshot.new_box_position)
        return self.render()

    def close(self):
        self.render_cache = None
        super(SokobanEnv, self).close()

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    config = SokobanEnvConfig(dim_room=(6, 6), num_boxes=1, max_steps=100, search_depth=10)
    env = SokobanEnv(config)
    for i in range(10):
        print(env.reset(seed=1010 + i))
        print()
    while True:
        keyboard = input("Enter action: ")
        if keyboard == 'q':
            break
        action = int(keyboard)
        assert action in env.ACTION_LOOKUP, f"Invalid action: {action}"
        obs, reward, done, info = env.step(action)
        print(obs, reward, done, info)
    np_img = env.get_image('rgb_array')
    plt.imsave('sokoban1.png', np_img)
