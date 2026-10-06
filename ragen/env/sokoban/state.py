"""Serializable simulator state, separate from the graph's finite-horizon key."""

from dataclasses import asdict, dataclass
import hashlib
import json


@dataclass
class SokobanSnapshot:
    room_fixed: list
    room_state: list
    player_position: list
    box_mapping: list
    num_env_steps: int
    boxes_on_target: int
    reward_last: float
    old_box_position: list | None
    new_box_position: list | None
    max_steps: int
    num_boxes: int

    def to_dict(self):
        return asdict(self)

    def key(self, problem_id, budget):
        fields = [problem_id, self.room_fixed, self.room_state,
                  self.player_position, self.num_env_steps, self.boxes_on_target,
                  budget - self.num_env_steps, self.max_steps, self.num_boxes]
        return hashlib.sha256(json.dumps(fields, separators=(",", ":")).encode()).hexdigest()
