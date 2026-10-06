import gymnasium as gym
import numpy as np
from typing import Tuple, Any, Dict, List
from ragen.env.base import BaseDiscreteActionEnv
from .config import RubiksCube2x2Config
from ragen.utils import all_seed

class RubiksCube2x2Env(BaseDiscreteActionEnv, gym.Env):
    """
    2x2 Pocket Cube Environment for LLM-based RL.
    State: 24 integers representing the stickers.
    Reward: +1.0 for solved, 0.0 otherwise.
    """
    def __init__(self, config: RubiksCube2x2Config | None = None):
        BaseDiscreteActionEnv.__init__(self)
        self.config = config or RubiksCube2x2Config()
        self.ACTION_LOOKUP = self.config.action_lookup
        self.ACTION_SPACE = gym.spaces.discrete.Discrete(12, start=1)
        self.render_mode = self.config.render_mode
        self.rng = np.random.default_rng()
        
        self.state = np.zeros(24, dtype=int)
        self.solved_state = np.zeros(24, dtype=int)
        self._init_solved_state()
        
        self.current_step = 0
        
    def _init_solved_state(self):
        for i in range(6):
            self.solved_state[i*4 : (i+1)*4] = i

    def reset(self, seed=None, mode=None):
        gym.Env.reset(self, seed=seed)
        with all_seed(seed):
            self.rng = np.random.default_rng(seed)
            self.state = self.solved_state.copy()
            self.current_step = 0
            
            depth = self.config.scramble_depth
            for _ in range(depth):
                action = self.rng.integers(1, 13) 
                self._apply_action(action)
                
        return self.render(done=False)

    def step(self, action: int) -> Tuple[Any, float, bool, Dict]:
        assert action in self.ACTION_LOOKUP, f"Invalid action: {action}"
        info = {"action_is_effective": True, "action_is_valid": True, "success": False}
        
        self._apply_action(action)
        self.current_step += 1
        
        is_solved = True
        for i in range(6):
            face_stickers = self.state[i*4 : (i+1)*4]
            if len(set(face_stickers)) > 1:
                is_solved = False
                break

        truncated = self.current_step >= self.config.max_steps
        
        done = is_solved or truncated
        
        if is_solved:
            reward = 1.0
            info["success"] = True
            msg = "Cube Solved!"
        elif truncated:
            reward = 0.0
            info["success"] = False
            msg = "Max steps reached."
        else:
            reward = 0.0
            msg = ""
            
        next_obs = self.render(done=done, result_msg=msg)

        return next_obs, float(reward), done, info

    def _rotate_face_clockwise(self, face_idx):

        base = face_idx * 4
        s = self.state
        tmp = s[base + 0]
        s[base + 0] = s[base + 2]
        s[base + 2] = s[base + 3]
        s[base + 3] = s[base + 1]
        s[base + 1] = tmp

    def _apply_action(self, action: int):
        if action % 2 == 0: 
            turns = 3
            base_act = action - 1
        else:
            turns = 1
            base_act = action
        
        for _ in range(turns):
            if base_act == 1: self._move_U()
            elif base_act == 3: self._move_D()
            elif base_act == 5: self._move_L()
            elif base_act == 7: self._move_R()
            elif base_act == 9: self._move_F()
            elif base_act == 11: self._move_B()
    
    def _move_U(self):
        self._rotate_face_clockwise(0)
        s = self.state
        t0, t1 = s[8], s[9]
        s[8], s[9] = s[12], s[13]
        s[12], s[13] = s[16], s[17]
        s[16], s[17] = s[4], s[5]
        s[4], s[5] = t0, t1

    def _move_D(self):
        self._rotate_face_clockwise(5) 
        s = self.state
        t0, t1 = s[10], s[11]
        s[10], s[11] = s[6], s[7]
        s[6], s[7] = s[18], s[19]
        s[18], s[19] = s[14], s[15]
        s[14], s[15] = t0, t1

    def _move_L(self):
        self._rotate_face_clockwise(1) 
        s = self.state
        t0, t2 = s[0], s[2]
        s[0], s[2] = s[19], s[17]
        s[19], s[17] = s[20], s[22]
        s[20], s[22] = s[8], s[10]
        s[8], s[10] = t0, t2

    def _move_R(self):
        self._rotate_face_clockwise(3) 
        s = self.state
        t1, t3 = s[1], s[3]
        s[1], s[3] = s[9], s[11]
        s[9], s[11] = s[21], s[23]
        s[21], s[23] = s[18], s[16]
        s[18], s[16] = t1, t3

    def _move_F(self):
        self._rotate_face_clockwise(2) 
        s = self.state
        t2, t3 = s[2], s[3]
        s[2], s[3] = s[7], s[5]
        s[7], s[5] = s[21], s[20]
        s[21], s[20] = s[12], s[14]
        s[12], s[14] = t2, t3

    def _move_B(self):
        self._rotate_face_clockwise(4) 
        s = self.state
        t0, t1 = s[0], s[1]
        s[0], s[1] = s[13], s[15]
        s[13], s[15] = s[22], s[23]
        s[22], s[23] = s[6], s[4]
        s[6], s[4] = t0, t1

    def render(self, mode: str | None = None, done: bool = False, result_msg: str = "") -> str:
        colors = {0: 'W', 1: 'O', 2: 'G', 3: 'R', 4: 'B', 5: 'Y'}
        
        def get_face_str(face_idx):
            base = face_idx * 4
            c = [colors[self.state[base+i]] for i in range(4)]
            return f"[{c[0]}, {c[1]}]\n       [{c[2]}, {c[3]}]"

        lines = []
        lines.append("=== Rubik's Cube 2x2 State ===")
        lines.append(f"Step: {self.current_step}/{self.config.max_steps}")
        lines.append("")
        lines.append(f"Up (U):    {get_face_str(0).strip().replace('       ', ' ')}")
        lines.append(f"Left (L):  {get_face_str(1).strip().replace('       ', ' ')}")
        lines.append(f"Front (F): {get_face_str(2).strip().replace('       ', ' ')}")
        lines.append(f"Right (R): {get_face_str(3).strip().replace('       ', ' ')}")
        lines.append(f"Back (B):  {get_face_str(4).strip().replace('       ', ' ')}")
        lines.append(f"Down (D):  {get_face_str(5).strip().replace('       ', ' ')}")
        
        if not done:
            lines.append("")
            lines.append("Available Actions:")
            lines.append("U, U', D, D', L, L', R, R', F, F', B, B'")
            lines.append("\nWhat is your next move?")
        else:
            lines.append("")
            lines.append(f"=== Game Over: {result_msg} ===")

        return "\n".join(lines)

    def close(self):
        pass

    def get_all_actions(self):
        return list(self.ACTION_LOOKUP.keys())

if __name__ == "__main__":
    config = RubiksCube2x2Config(scramble_depth=1, max_steps=50)
    env = RubiksCube2x2Env(config)

    print(f"Action Lookup: {env.ACTION_LOOKUP}")
    
    obs = env.reset()
    print(obs)
    
    while True:
        keyboard = input("\nEnter action (1-12) or 'q' to quit: ")
        if keyboard == 'q':
            break
        
        try:
            action = int(keyboard)
        except ValueError:
            print("Please enter a valid integer.")
            continue

        if action not in env.ACTION_LOOKUP:
            print(f"Invalid action: {action}. Please input 1-12.")
            continue
            
        obs, reward, done, info = env.step(action)
        
        print(obs) 
        print(f"Reward: {reward}, Done: {done}, Info: {info}")
        
        if done:
            print("\n=== Episode Ended. Resetting... ===")
            obs = env.reset()
            print(obs)