"""Launcher contracts: untouched baseline interfaces, explicit eval overrides and safe retries."""

import os
from pathlib import Path
import shlex
import subprocess

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/runs/run_frozen_lake_light.sh"


def run_script(*args, **env):
    settings = {**os.environ, "MODEL": "Qwen/Qwen2.5-3B-Instruct", "GPU": "0", "SEED": "10000",
                "EXPERIMENT": "frozen_lake_test", "MICRO_BATCH_SIZE": "", "WANDB_PROJECT": "",
                "STEPS": "", "TEST_FREQ": "", "SAVE_FREQ": "", "EVAL_ATTEMPTS": "",
                "EVAL_PROBLEMS": "", "EVAL_SEED": "", "TRAIN_POOL_SIZE": "", "RUNS": "",
                "TROPIC_COVERAGE": "", **env}
    return subprocess.run(["bash", str(SCRIPT), *args], cwd=ROOT, env=settings,
                          text=True, capture_output=True, timeout=30)


def configs(result, methods=("EVAL", "PPO", "GRPO", "SNR", "TROPIC")):
    assert result.returncode == 0, result.stderr
    commands = [shlex.split(line.removeprefix("COMMAND ")) for line in result.stdout.splitlines()
                if line.startswith("COMMAND ")]
    assert len(commands) == len(methods)
    cfgs = []
    with initialize_config_dir(config_dir=str(ROOT / "config"), version_base=None):
        for method, command in zip(methods, commands):
            index = command.index("--config-name")
            assert command[index + 1] == ("_3_frozen_lake_tropic" if method == "TROPIC" else "_3_frozen_lake")
            cfg = compose(config_name=command[index + 1], overrides=command[index + 2:])
            OmegaConf.resolve(cfg)
            cfgs.append(cfg)
    assert [cfg.trainer.experiment_name.rsplit("_", 1)[-1] for cfg in cfgs] == list(methods)
    return cfgs


def test_help_and_defaults():
    result = run_script("--help")
    assert result.returncode == 0
    assert "_3_frozen_lake directly" in result.stdout
    assert "microbatch 1" in result.stdout and "RAGEN_FROZEN_LAKE" in result.stdout
    assert "RUNS=TROPIC" in result.stdout
    assert "TROPIC_COVERAGE=true" in result.stdout


def test_original_baseline_interfaces_rewards_and_batches_are_unchanged(tmp_path):
    from train import add_dependency_and_validate_config
    cfgs = configs(run_script("--dry-run", SAVE_DIR=str(tmp_path / "unused"), GPU="0,1,2,3,4,5,6,7"))
    with initialize_config_dir(config_dir=str(ROOT / "config"), version_base=None):
        original = compose(config_name="_3_frozen_lake")
        OmegaConf.resolve(original)
    for cfg in cfgs:
        add_dependency_and_validate_config(cfg)
        assert cfg.trainer.n_gpus_per_node == 8
        assert cfg.trainer.project_name == "RAGEN_FROZEN_LAKE"
        assert cfg.micro_batch_size_per_gpu == original.micro_batch_size_per_gpu == 1
        assert cfg.ppo_mini_batch_size == original.ppo_mini_batch_size == 32
        assert cfg.es_manager.val == original.es_manager.val
        assert cfg.actor_rollout_ref.rollout.val_kwargs == original.actor_rollout_ref.rollout.val_kwargs
        assert cfg.actor_rollout_ref.rollout.max_model_len == original.actor_rollout_ref.rollout.max_model_len
        assert cfg.actor_rollout_ref.rollout.response_length == original.actor_rollout_ref.rollout.response_length
        assert cfg.trainer.test_freq == original.trainer.test_freq == 10
        assert cfg.custom_envs.CoordFrozenLake.max_actions_per_traj == 10
        assert cfg.custom_envs.CoordFrozenLake.env_config.success_rate == 1
    for cfg in cfgs[:4]:
        for key in ("agent_proxy", "custom_envs", "collapse_detection", "ctx_manager"):
            assert cfg[key] == original[key]
        assert cfg.es_manager.train == original.es_manager.train
        assert cfg.actor_rollout_ref.actor.entropy_coeff == original.actor_rollout_ref.actor.entropy_coeff
        assert cfg.actor_rollout_ref.actor.optim == original.actor_rollout_ref.actor.optim
    evaluation, ppo, grpo, snr, tropic = cfgs
    assert evaluation.trainer.val_only and not evaluation.critic.enable and not evaluation.actor_rollout_ref.actor.use_ref
    assert ppo.critic.enable and snr.critic.enable
    assert ppo.actor_rollout_ref.actor == snr.actor_rollout_ref.actor == original.actor_rollout_ref.actor
    assert ppo.algorithm == snr.algorithm == original.algorithm
    assert grpo.algorithm.adv_estimator == "grpo" and not grpo.critic.enable
    assert grpo.algorithm.norm_adv_by_std_in_grpo
    assert grpo.actor_rollout_ref.actor.loss_agg_mode == "seq-mean-token-mean"
    assert ppo.actor_rollout_ref.rollout.rollout_filter_value == 1.0
    assert snr.actor_rollout_ref.rollout.rollout_filter_value == .9
    assert snr.actor_rollout_ref.rollout.rollout_filter_top_p_prob_mode == "softmax"
    assert tropic.custom_envs.CoordFrozenLake.env_type == "frozen_lake_tropic"
    assert tropic.es_manager.train.group_size * tropic.tropic.waves_per_iteration == original.es_manager.train.group_size
    assert tropic.agent_proxy.max_turn == 10 and tropic.agent_proxy.max_actions_per_turn == 1
    assert tropic.tropic.basis_size == 2 and not tropic.tropic.coverage_enabled
    assert tropic.tropic.composition_enabled and tropic.tropic.frontier_enabled
    assert tropic.es_manager.train.seed_pool_size == 128
    assert not (tmp_path / "unused").exists()


def test_pilot_and_pass_at_k_are_explicit_shared_overrides(tmp_path):
    cfgs = configs(run_script("--dry-run", SAVE_DIR=str(tmp_path), GPU="0,1,2,3,4,5,6,7",
                             EVAL_PROBLEMS="24", EVAL_ATTEMPTS="2", STEPS="2", TEST_FREQ="2", SAVE_FREQ="2",
                             WANDB_ENTITY="my-team", WANDB_PROJECT="frozen-lake-development"))
    for cfg in cfgs:
        assert cfg.es_manager.val.env_groups == 24 and cfg.es_manager.val.group_size == 2
        assert cfg.trainer.test_freq == 2
        assert cfg.trainer.project_name == "frozen-lake-development"
        assert cfg.ray_kwargs.ray_init.runtime_env.env_vars.WANDB_ENTITY == "my-team"
    assert all(cfg.trainer.total_training_steps == 2 for cfg in cfgs[1:])
    cfgs = configs(run_script("--dry-run", SAVE_DIR=str(tmp_path), EVAL_ATTEMPTS="8", TEST_FREQ="25"))
    assert all(cfg.es_manager.val.group_size == 8 and cfg.trainer.test_freq == 25 for cfg in cfgs)


@pytest.mark.parametrize("env", [dict(EVAL_PROBLEMS="0"), dict(EVAL_ATTEMPTS="0"), dict(GPU="0,0"),
                                dict(GPU="0,1,2"), dict(MICRO_BATCH_SIZE="3"), dict(EXPERIMENT="../bad"),
                                dict(TEST_FREQ="0"), dict(SEED="-1"), dict(TRAIN_POOL_SIZE="7"),
                                dict(TROPIC_COVERAGE="yes"), dict(TROPIC_COVERAGE="1")])
def test_invalid_options_rejected_before_launch(tmp_path, env):
    result = run_script("--dry-run", SAVE_DIR=str(tmp_path / "unused"), **env)
    assert result.returncode != 0 and "COMMAND " not in result.stdout
    assert not (tmp_path / "unused").exists()


def test_coverage_ablation_changes_only_the_tropic_coverage_setting():
    off = configs(run_script("--dry-run", TROPIC_COVERAGE="false"))
    on = configs(run_script("--dry-run", TROPIC_COVERAGE="true"))
    assert off[:4] == on[:4]
    assert on[-1].tropic.coverage_enabled
    assert on[-1].tropic.basis_size == off[-1].tropic.basis_size == 2
    on[-1].tropic.coverage_enabled = False
    assert on[-1] == off[-1]


def test_existing_run_is_not_overwritten(tmp_path):
    (tmp_path / "RAGEN2_Qwen2.5-3B-Instruct_frozen_lake_test_GRPO").mkdir()
    result = run_script(SAVE_DIR=str(tmp_path), PYTHON="/nonexistent")
    assert result.returncode != 0 and "Refusing to overwrite" in result.stderr
    assert len(list(tmp_path.iterdir())) == 1


def test_preflight_failure_creates_no_artifacts(tmp_path):
    result = run_script(SAVE_DIR=str(tmp_path / "unused"), PYTHON="false")
    assert result.returncode != 0 and "preflight failed" in result.stderr
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize("runs,methods", [("TROPIC", ("TROPIC",)), ("GRPO", ("GRPO",)),
                                         ("TROPIC,EVAL", ("EVAL", "TROPIC"))])
def test_runs_selects_subset_in_canonical_order(tmp_path, runs, methods):
    cfgs = configs(run_script("--dry-run", RUNS=runs, SAVE_DIR=str(tmp_path / "unused")), methods)
    for method, cfg in zip(methods, cfgs):
        assert cfg.trainer.method == ("tropic" if method == "TROPIC" else "ppo")
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize("runs", ["tropic", "UNKNOWN", "EVAL,EVAL", "EVAL,", ",EVAL", "EVAL,,TROPIC", " "])
def test_invalid_runs_rejected_before_launch(tmp_path, runs):
    result = run_script("--dry-run", RUNS=runs, SAVE_DIR=str(tmp_path / "unused"))
    assert result.returncode != 0 and "RUNS" in result.stderr
    assert "COMMAND " not in result.stdout and not (tmp_path / "unused").exists()


def mock_python(tmp_path):
    executable = tmp_path / "fake-python"
    executable.write_text('''#!/usr/bin/env bash
if [[ "$1" == "-m" ]]; then printf '{}\\n' > "$4"; exit 0; fi
if [[ "$1" == "-c" ]]; then exit 0; fi
printf 'Mock FrozenLake training\\n'
if [[ "$*" == *"trainer.experiment_name=RAGEN2_Qwen2.5-3B-Instruct_frozen_lake_test_PPO"* ]]; then
    exit "${FAKE_PPO_STATUS:-0}"
fi
''')
    executable.chmod(0o755)
    return str(executable)


def test_only_selected_run_is_checked_on_retry(tmp_path):
    output = tmp_path / "saves"
    prefix = "RAGEN2_Qwen2.5-3B-Instruct_frozen_lake_test_"
    evaluation, tropic = output / (prefix + "EVAL"), output / (prefix + "TROPIC")
    evaluation.mkdir(parents=True)
    (evaluation / "metrics.jsonl").write_text("existing evaluation\n")
    tropic.mkdir()
    result = run_script(RUNS="TROPIC", PYTHON="/nonexistent", SAVE_DIR=str(output))
    assert result.returncode != 0 and "Refusing to overwrite" in result.stderr
    tropic.rename(tmp_path / "failed-tropic")
    result = run_script(RUNS="TROPIC", PYTHON=mock_python(tmp_path), SAVE_DIR=str(output))
    assert result.returncode == 0, result.stderr
    assert {path.name for path in output.iterdir()} == {evaluation.name, tropic.name}
    assert (tropic / "COMPLETED").exists()
    assert (evaluation / "metrics.jsonl").read_text() == "existing evaluation\n"


@pytest.mark.parametrize("ppo_status", [0, 7])
def test_execution_artifacts_and_failure_propagation(tmp_path, ppo_status):
    output = tmp_path / "artifacts"
    result = run_script(SAVE_DIR=str(output), PYTHON=mock_python(tmp_path), FAKE_PPO_STATUS=str(ppo_status))
    assert result.returncode == ppo_status, result.stderr
    prefix = "RAGEN2_Qwen2.5-3B-Instruct_frozen_lake_test_"
    assert (output / (prefix + "EVAL") / "COMPLETED").exists()
    assert (output / (prefix + "PPO") / "exit_code.txt").read_text().strip() == str(ppo_status)
    assert (output / (prefix + "PPO") / "dataset_manifest.json").exists()
    assert (output / (prefix + "TROPIC") / "COMPLETED").exists() is (ppo_status == 0)
