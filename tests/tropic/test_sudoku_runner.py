"""Launch contracts for the Sudoku light suite: original baselines, shared budget, no side effects."""

import os
from pathlib import Path
import shlex
import subprocess

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/runs/run_sudoku_light.sh"
ALL = ("EVAL", "PPO", "GRPO", "SNR", "TROPIC")


def run_script(*args, **env):
    settings = {**os.environ, "MODEL": "Qwen/Qwen2.5-3B-Instruct", "GPU": "0", "SEED": "10000",
                "EXPERIMENT": "sudoku_test", "MICRO_BATCH_SIZE": "", "WANDB_PROJECT": "", "STEPS": "",
                "TEST_FREQ": "", "SAVE_FREQ": "", "EVAL_ATTEMPTS": "", "EVAL_PROBLEMS": "", "ACTION_BUDGET": "",
                "RUNS": "", **env}
    return subprocess.run(["bash", str(SCRIPT), *args], cwd=ROOT, env=settings,
                          text=True, capture_output=True, timeout=30)


def configs(result, methods=ALL):
    assert result.returncode == 0, result.stderr
    commands = [shlex.split(line.removeprefix("COMMAND ")) for line in result.stdout.splitlines()
                if line.startswith("COMMAND ")]
    assert len(commands) == len(methods)
    assert [c[c.index("--config-name") + 1] for c in commands] == [
        "_8_sudoku_tropic" if method == "TROPIC" else "_8_sudoku" for method in methods]
    cfgs = []
    with initialize_config_dir(config_dir=str(ROOT / "config"), version_base=None):
        for command in commands:
            index = command.index("--config-name")
            cfg = compose(config_name=command[index + 1], overrides=command[index + 2:])
            OmegaConf.resolve(cfg)
            cfgs.append(cfg)
    assert [cfg.trainer.experiment_name.rsplit("_", 1)[-1] for cfg in cfgs] == list(methods)
    return cfgs


def original():
    with initialize_config_dir(config_dir=str(ROOT / "config"), version_base=None):
        cfg = compose(config_name="_8_sudoku")
    OmegaConf.resolve(cfg)
    return cfg


def test_help_and_defaults():
    result = run_script("--help")
    assert result.returncode == 0
    assert "_8_sudoku directly" in result.stdout and "ACTION_BUDGET=40" in result.stdout
    assert "32 puzzles x 16" in result.stdout and "RAGEN_SUDOKU" in result.stdout


def test_original_budget_reproduces_the_config_exactly(tmp_path):
    from train import add_dependency_and_validate_config
    base = original()
    cfgs = configs(run_script("--dry-run", SAVE_DIR=str(tmp_path / "unused"), ACTION_BUDGET="10"))
    for cfg in cfgs[:4]:
        add_dependency_and_validate_config(cfg)
        for key in ("agent_proxy", "custom_envs", "es_manager", "collapse_detection", "ctx_manager"):
            assert cfg[key] == base[key]
        assert cfg.actor_rollout_ref.rollout.max_model_len == base.actor_rollout_ref.rollout.max_model_len
        assert cfg.actor_rollout_ref.actor.entropy_coeff == base.actor_rollout_ref.actor.entropy_coeff
        assert cfg.micro_batch_size_per_gpu == base.micro_batch_size_per_gpu == 1
        assert cfg.ppo_mini_batch_size == 32 and cfg.trainer.test_freq == 10
    assert cfgs[4].custom_envs.SimpleSudoku.max_actions_per_traj == cfgs[4].agent_proxy.max_turn == 10
    assert all(cfg.agent_proxy.max_turn == 5 and cfg.custom_envs.SimpleSudoku.max_actions_per_traj == 20 for cfg in cfgs[:4])
    assert not (tmp_path / "unused").exists()


def test_default_budget_is_the_only_shared_deviation(tmp_path):
    from train import add_dependency_and_validate_config
    from ragen.tropic.config import validate_tropic_config
    base = original()
    cfgs = configs(run_script("--dry-run", SAVE_DIR=str(tmp_path / "unused"), GPU="0,1,2,3,4,5,6,7"))
    evaluation, ppo, grpo, snr, tropic = cfgs
    for cfg in cfgs:
        add_dependency_and_validate_config(cfg)
        assert cfg.trainer.n_gpus_per_node == 8 and cfg.trainer.project_name == "RAGEN_SUDOKU"
        assert cfg.custom_envs.SimpleSudoku.max_actions_per_traj == 40
        assert cfg.es_manager.val.env_groups == 32 and cfg.es_manager.val.group_size == 16
        assert cfg.actor_rollout_ref.rollout.val_kwargs == base.actor_rollout_ref.rollout.val_kwargs
    for cfg in cfgs[:4]:
        assert cfg.agent_proxy.max_turn == 40 and cfg.agent_proxy.max_actions_per_turn == 2  
        proxy = OmegaConf.to_container(cfg.agent_proxy); proxy["max_turn"] = base.agent_proxy.max_turn
        assert proxy == OmegaConf.to_container(base.agent_proxy)
        task = OmegaConf.to_container(cfg.custom_envs.SimpleSudoku); task["max_actions_per_traj"] = 20
        assert task == OmegaConf.to_container(base.custom_envs.SimpleSudoku)
        assert cfg.es_manager.train == base.es_manager.train
    assert evaluation.trainer.val_only and not evaluation.critic.enable
    assert ppo.critic.enable and snr.critic.enable and ppo.algorithm.adv_estimator == "gae"
    assert ppo.actor_rollout_ref.actor == snr.actor_rollout_ref.actor == base.actor_rollout_ref.actor
    assert grpo.algorithm.adv_estimator == "grpo" and not grpo.critic.enable
    assert grpo.actor_rollout_ref.actor.loss_agg_mode == "seq-mean-token-mean"
    assert ppo.actor_rollout_ref.rollout.rollout_filter_value == 1.0
    assert snr.actor_rollout_ref.rollout.rollout_filter_value == .9
    assert snr.actor_rollout_ref.rollout.rollout_filter_top_p_prob_mode == "softmax"
    validate_tropic_config(tropic)
    assert tropic.trainer.method == "tropic" and tropic.custom_envs.SimpleSudoku.env_type == "sudoku_tropic"
    assert tropic.agent_proxy.max_turn == 40 and tropic.agent_proxy.max_actions_per_turn == 1
    assert tropic.es_manager.train.group_size * tropic.tropic.waves_per_iteration == base.es_manager.train.group_size
    assert tropic.es_manager.train.seed_pool_size == 128


def test_pilot_and_pass_at_k_overrides(tmp_path):
    cfgs = configs(run_script("--dry-run", SAVE_DIR=str(tmp_path), EVAL_PROBLEMS="8", EVAL_ATTEMPTS="2",
                             STEPS="2", TEST_FREQ="2", SAVE_FREQ="2", WANDB_ENTITY="my-team"))
    for cfg in cfgs:
        assert cfg.es_manager.val.env_groups == 8 and cfg.es_manager.val.group_size == 2
        assert cfg.trainer.test_freq == 2
        assert cfg.ray_kwargs.ray_init.runtime_env.env_vars.WANDB_ENTITY == "my-team"
    assert all(cfg.trainer.total_training_steps == 2 for cfg in cfgs[1:])


@pytest.mark.parametrize("env", [dict(ACTION_BUDGET="81"), dict(ACTION_BUDGET="82"), dict(ACTION_BUDGET="0"),
                                dict(EVAL_ATTEMPTS="0"), dict(GPU="0,0"), dict(GPU="0,1,2"),
                                dict(MICRO_BATCH_SIZE="3"), dict(EXPERIMENT="../bad"), dict(TEST_FREQ="0"),
                                dict(SEED="-1"), dict(RUNS="tropic"), dict(RUNS="EVAL,EVAL")])
def test_invalid_options_rejected_before_launch(tmp_path, env):
    result = run_script("--dry-run", SAVE_DIR=str(tmp_path / "unused"), **env)
    assert result.returncode != 0 and "COMMAND " not in result.stdout
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize("runs,methods", [("TROPIC", ("TROPIC",)), ("SNR,EVAL", ("EVAL", "SNR")), ("", ALL)])
def test_runs_selects_subset_in_canonical_order(tmp_path, runs, methods):
    cfgs = configs(run_script("--dry-run", RUNS=runs, SAVE_DIR=str(tmp_path / "unused")), methods)
    for method, cfg in zip(methods, cfgs):
        assert cfg.trainer.method == ("tropic" if method == "TROPIC" else "ppo")


def test_existing_selected_run_is_not_overwritten(tmp_path):
    (tmp_path / "RAGEN2_Qwen2.5-3B-Instruct_sudoku_test_GRPO").mkdir()
    result = run_script(SAVE_DIR=str(tmp_path), PYTHON="/nonexistent")
    assert result.returncode != 0 and "Refusing to overwrite" in result.stderr
    assert len(list(tmp_path.iterdir())) == 1
    result = run_script(SAVE_DIR=str(tmp_path), PYTHON="false", RUNS="TROPIC")
    assert "Refusing to overwrite" not in result.stderr and "preflight failed" in result.stderr


def test_preflight_failure_creates_no_artifacts(tmp_path):
    result = run_script(SAVE_DIR=str(tmp_path / "unused"), PYTHON="false")
    assert result.returncode != 0 and "preflight failed" in result.stderr
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize("ppo_status", [0, 7])
def test_execution_artifacts_and_failure_propagation(tmp_path, ppo_status):
    executable = tmp_path / "fake-python"
    executable.write_text("""#!/usr/bin/env bash
if [[ "$1" == "-m" ]]; then printf '{}\\n' > "$4"; exit 0; fi
if [[ "$1" == "-c" ]]; then exit 0; fi
printf 'Mock Sudoku training\\n'
if [[ "$*" == *"trainer.experiment_name=RAGEN2_Qwen2.5-3B-Instruct_sudoku_test_PPO"* ]]; then
    exit "$FAKE_PPO_STATUS"
fi
""")
    executable.chmod(0o755)
    output = tmp_path / "artifacts"
    result = run_script(SAVE_DIR=str(output), PYTHON=str(executable), FAKE_PPO_STATUS=str(ppo_status))
    assert result.returncode == ppo_status, result.stderr
    prefix = "RAGEN2_Qwen2.5-3B-Instruct_sudoku_test_"
    assert (output / (prefix + "EVAL") / "COMPLETED").exists()
    ppo = output / (prefix + "PPO")
    assert (ppo / "exit_code.txt").read_text().strip() == str(ppo_status)
    assert (ppo / "dataset_manifest.json").exists() and (ppo / "command.sh").exists()
    if ppo_status:
        assert not (ppo / "COMPLETED").exists() and not (output / (prefix + "GRPO")).exists()
    else:
        assert all((output / (prefix + m) / "COMPLETED").exists() for m in ALL)


def test_odd_budgets_are_allowed_and_bind_turns_and_placements_together(tmp_path):
    cfgs = configs(run_script("--dry-run", SAVE_DIR=str(tmp_path / "unused"), ACTION_BUDGET="33"))
    for cfg in cfgs:
        assert cfg.agent_proxy.max_turn == cfg.custom_envs.SimpleSudoku.max_actions_per_traj == 33
