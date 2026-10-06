"""Sudoku TROPIC contracts with the real environment on small 4x4 puzzles."""

from pathlib import Path

from hydra import compose, initialize_config_dir
import numpy as np
import pytest

from conftest import CharacterTokenizer
from test_collector import ScriptedProxy, score
from ragen.env.sudoku.config import SudokuEnvConfig
from ragen.env.sudoku.state import SudokuSnapshot
from ragen.env.sudoku.tropic_env import SudokuTropicEnv
from ragen.tropic.sudoku_collector import SudokuTropicCollector

SEED = 7  


def small_env():
    return SudokuTropicEnv(SudokuEnvConfig(grid_size=4, max_steps=16))


def solution_moves(env):
    """(row, col, number) for every blank, row-major, from the hidden solution."""
    return [(int(r) + 1, int(c) + 1, int(env.solution_grid[r, c])) for r, c in zip(*np.where(env.current_grid == 0))]


def place(env, move):
    r, c, n = move
    return env.step(f"place {n} at row {r} col {c}")


def test_snapshot_roundtrip_and_pure_render():
    env, peer = small_env(), small_env()
    env.reset(seed=SEED)
    moves = solution_moves(env)
    assert len(moves) == 6
    place(env, moves[0]); place(env, moves[1])
    snapshot = env.get_state()
    peer.reset(seed=SEED + 1)  
    assert peer.set_state(snapshot.to_dict()) == env.render()
    assert peer.get_state() == snapshot and peer.num_env_steps == 2 and not peer._done
    assert "Correct" not in env.render() and "✓" not in env.render()  
    _, _, _, info = place(peer, moves[2])
    assert info["action_is_valid"] and peer.get_state().current_grid != snapshot.current_grid


def test_commuting_placements_join_but_different_depths_do_not():
    a, b = small_env(), small_env()
    a.reset(seed=SEED); b.reset(seed=SEED)
    moves = solution_moves(a)
    place(a, moves[0]); place(a, moves[1])
    place(b, moves[1]); place(b, moves[0])
    assert a.get_state().key("p", 6) == b.get_state().key("p", 6)
    place(b, moves[2])
    assert a.get_state().key("p", 6) != b.get_state().key("p", 6)
    assert a.get_state().key("p", 6) != a.get_state().key("p", 8)  


@pytest.mark.parametrize("action", ["place 9 at row 1 col 1", "nonsense", "1,1,4"])
def test_invalid_placements_terminate_without_success(action):
    env = small_env()
    env.reset(seed=SEED)
    given = [(int(r) + 1, int(c) + 1) for r, c in zip(*np.where(env.initial_grid != 0))]
    if action == "1,1,4" and (1, 1) not in given:
        action = f"{given[0][0]},{given[0][1]},4"  
    _, reward, done, info = env.step(action)
    assert done and not info["action_is_valid"] and not info["success"] and reward < 0
    with pytest.raises(RuntimeError):
        env.step("1,1,1")


def test_completing_the_puzzle_is_the_only_success():
    env = small_env()
    env.reset(seed=SEED)
    outcomes = [place(env, move) for move in solution_moves(env)]
    assert [o[3]["success"] for o in outcomes] == [False] * 5 + [True]
    assert outcomes[-1][2] and outcomes[-1][1] > 10  
    with pytest.raises(ValueError):
        env.set_state(SudokuSnapshot(env.initial_grid.tolist(), env.initial_grid.tolist(),
                                     np.zeros((4, 4), dtype=int).tolist(), 0, 16))


def test_preflight_checks_budget_and_replay(tmp_path):
    from ragen.env.sudoku.preflight import run_preflight
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[2] / "config"), version_base=None):
        cfg = compose(config_name="_8_sudoku", overrides=["custom_envs.SimpleSudoku.env_config.grid_size=4",
                                                          "es_manager.val.env_groups=4", "seed.train=10000"])
    manifest = run_preflight(cfg, 8, 6, require_solvable=True)
    assert manifest["max_blanks"] <= 6 and manifest["solvable_within_budget"] and len(manifest["validation_seeds"]) == 4
    with pytest.raises(ValueError, match="cannot fill"):
        run_preflight(cfg, 8, 4, require_solvable=True)
    assert not run_preflight(cfg, 8, 4, require_solvable=False)["solvable_within_budget"]


@pytest.fixture
def sudoku_config():
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[2] / "config"), version_base=None):
        cfg = compose(config_name="_8_sudoku_tropic")
    cfg.seed.train = SEED
    cfg.custom_envs.SimpleSudoku.env_config.grid_size = 4
    cfg.custom_envs.SimpleSudoku.env_config.max_steps = 16
    cfg.custom_envs.SimpleSudoku.max_actions_per_traj = 6
    cfg.agent_proxy.max_turn = 6
    cfg.es_manager.train.env_groups = cfg.es_manager.train.seed_pool_size = 1
    cfg.es_manager.train.env_configs.n_groups = [1]
    cfg.es_manager.train.group_size = 2
    cfg.es_manager.val.env_groups = cfg.es_manager.val.group_size = 1
    cfg.es_manager.val.env_configs.n_groups = [1]
    cfg.tropic.waves_per_iteration = 1
    cfg.actor_rollout_ref.rollout.max_model_len = 10000
    return cfg


def test_collection_composes_a_solution_not_sampled_end_to_end(sudoku_config):
    probe = small_env(); probe.reset(seed=SEED)
    m = [f"place {n} at row {r} col {c}" for r, c, n in solution_moves(probe)]

    class SudokuProxy(ScriptedProxy):
        scripts = {0: [m[0], m[1], "place 9 at row 1 col 1", m[3], m[4], m[5]],
                   1: [m[1], m[0], m[2], m[3], m[4], m[5]]}

    proxy = SudokuProxy(sudoku_config, None, CharacterTokenizer())
    collector = SudokuTropicCollector(sudoku_config, proxy, score)
    try:
        bases, metrics = collector.collect(1)
        graph = next(iter(collector.graphs.values()))
        solutions = {tuple(graph.edges[k].action for k in p) for p in graph.solutions}
        assert (m[1], m[0], m[2], m[3], m[4], m[5]) in solutions  
        assert (m[0], m[1], m[2], m[3], m[4], m[5]) in solutions  
        assert metrics["tropic/root_verified"] == 1 and metrics["tropic/composition_new_solutions"] >= 1
        assert metrics.get("tropic/failed_verifications", 0) == 0 and metrics["tropic/invalid_decisions"] == 1
        assert len(bases[graph.problem_id]) == 2
        restored = collector.state_dict()
        fresh = SudokuTropicCollector(sudoku_config, proxy, score)
        fresh.load_state_dict(restored)
        assert fresh.graphs[graph.problem_id].solutions == graph.solutions
    finally:
        collector.close()
        proxy.train_es_manager.close()
        proxy.val_es_manager.close()
