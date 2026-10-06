import json
import pytest

from ragen.env.sokoban.config import SokobanEnvConfig
from ragen.env.sokoban.env import SokobanEnv
from ragen.env.sokoban.state import SokobanSnapshot


def test_snapshot_round_trip_reproduces_observations_rewards_and_termination():
    cfg = SokobanEnvConfig(dim_room=(5, 5), num_boxes=1, max_steps=8, search_depth=10)
    env = SokobanEnv(cfg)
    clone = SokobanEnv(cfg)
    try:
        env.reset(seed=1234)
        env.step(1)
        snapshot = json.loads(json.dumps(env.get_state().to_dict()))
        clone.set_state(snapshot)
        for action in [4, 2, 3, 1, 1, 4, 2]:
            assert env.step(action) == clone.step(action)
            assert env.get_state() == clone.get_state()
        clone.room_state[0, 0] = 99
        assert env.room_state[0, 0] != 99
        assert snapshot['room_state'][0][0] != 99
    finally:
        env.close()
        clone.close()


def test_key_includes_problem_depth_and_budget():
    state = SokobanSnapshot([[0]], [[0]], [0, 0], [], 0, 0, 0.0, None, None, 10, 1)
    key = state.key('a', 5)
    assert key != state.key('b', 5)
    assert key != state.key('a', 6)
    state.num_env_steps += 1
    assert key != state.key('a', 5)


def test_incompatible_snapshot_rules_are_rejected():
    env = SokobanEnv(SokobanEnvConfig(dim_room=(5, 5), num_boxes=1, search_depth=10))
    try:
        state = env.get_state()
        state.max_steps += 1
        with pytest.raises(ValueError, match='rules'):
            env.set_state(state)
    finally:
        env.close()
