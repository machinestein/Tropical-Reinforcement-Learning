"""Launch contracts for the Countdown light suite: original single-turn baselines, shared data, no side effects."""

import os
from pathlib import Path
import shlex
import subprocess

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/runs/run_countdown_light.sh"
ALL = ("EVAL", "PPO", "GRPO", "SNR", "TROPIC")


def run_script(*args, script=SCRIPT, **env):
    settings = {**os.environ, "MODEL": "Qwen/Qwen2.5-3B-Instruct", "GPU": "0", "SEED": "10000",
                "EXPERIMENT": "countdown_test", "MICRO_BATCH_SIZE": "", "WANDB_PROJECT": "", "STEPS": "",
                "TEST_FREQ": "", "SAVE_FREQ": "", "EVAL_ATTEMPTS": "", "EVAL_PROBLEMS": "", "DATA": "", "RUNS": "",
                "PROTOCOL": "", "WARMUP_MODEL": "", "WARMUP_STEPS": "", "WARMUP_MICRO_BATCH_SIZE": "", **env}
    return subprocess.run(["bash", str(script), *args], cwd=ROOT, env=settings,
                          text=True, capture_output=True, timeout=30)


def configs(result, methods=ALL, protocol="original"):
    assert result.returncode == 0, result.stderr
    commands = [shlex.split(line.removeprefix("COMMAND ")) for line in result.stdout.splitlines()
                if line.startswith("COMMAND ")]
    assert len(commands) == len(methods)
    assert [c[c.index("--config-name") + 1] for c in commands] == [
        "_4_countdown_atomic" if protocol == "atomic" else
        "_4_countdown_tropic" if method.startswith("TROPIC") else "_4_countdown" for method in methods]
    cfgs = []
    with initialize_config_dir(config_dir=str(ROOT / "config"), version_base=None):
        for command in commands:
            index = command.index("--config-name")
            cfg = compose(config_name=command[index + 1], overrides=command[index + 2:])
            OmegaConf.resolve(cfg)
            cfgs.append(cfg)
    assert all(cfg.trainer.experiment_name.endswith("_" + method) for cfg, method in zip(cfgs, methods))
    return cfgs


def original():
    with initialize_config_dir(config_dir=str(ROOT / "config"), version_base=None):
        cfg = compose(config_name="_4_countdown")
    OmegaConf.resolve(cfg)
    return cfg


def test_help_and_defaults():
    result = run_script("--help")
    assert result.returncode == 0
    assert "_4_countdown directly" in result.stdout and "DATA=original" in result.stdout
    assert "RAGEN_COUNTDOWN" in result.stdout and "RUNS=TROPIC" in result.stdout
    assert "PROTOCOL=atomic" in result.stdout and "WARMUP_MODEL" in result.stdout


def test_baselines_keep_the_original_single_turn_protocol_and_share_generated_data(tmp_path):
    from train import add_dependency_and_validate_config
    from ragen.tropic.config import validate_tropic_config
    base = original()
    cfgs = configs(run_script("--dry-run", SAVE_DIR=str(tmp_path / "unused"), GPU="0,1,2,3,4,5,6,7"))
    evaluation, ppo, grpo, snr, tropic = cfgs
    for cfg in cfgs:
        add_dependency_and_validate_config(cfg)
        assert cfg.trainer.n_gpus_per_node == 8 and cfg.trainer.project_name == "RAGEN_COUNTDOWN"
        assert cfg.custom_envs.Countdown.env_config.train_path == "data/countdown/tropic_train.parquet"
        assert not cfg.custom_envs.Countdown.env_config.solvable_filter
        assert cfg.es_manager.val.env_config_overrides.Countdown.train_path == "data/countdown/tropic_val.parquet"
        assert cfg.es_manager.val.env_groups == 512 and cfg.es_manager.val.group_size == 1
        assert cfg.actor_rollout_ref.rollout.val_kwargs == base.actor_rollout_ref.rollout.val_kwargs
        assert cfg.micro_batch_size_per_gpu == 1 and cfg.ppo_mini_batch_size == 32 and cfg.trainer.test_freq == 10
    for cfg in cfgs[:4]:
        assert cfg.agent_proxy == base.agent_proxy                      
        task = OmegaConf.to_container(cfg.custom_envs.Countdown); task["env_config"] = None
        assert task == OmegaConf.to_container(base.custom_envs.Countdown)
        assert cfg.es_manager.train == base.es_manager.train
        for key in ("collapse_detection", "ctx_manager"):
            assert cfg[key] == base[key]
    assert evaluation.trainer.val_only and not evaluation.critic.enable
    assert ppo.critic.enable and snr.critic.enable and ppo.algorithm.adv_estimator == "gae"
    assert ppo.actor_rollout_ref.actor == snr.actor_rollout_ref.actor == base.actor_rollout_ref.actor
    assert grpo.algorithm.adv_estimator == "grpo" and not grpo.critic.enable
    assert grpo.actor_rollout_ref.actor.loss_agg_mode == "seq-mean-token-mean"
    assert ppo.actor_rollout_ref.rollout.rollout_filter_value == 1.0
    assert snr.actor_rollout_ref.rollout.rollout_filter_value == .9
    assert snr.actor_rollout_ref.rollout.rollout_filter_top_p_prob_mode == "softmax"
    validate_tropic_config(tropic)
    assert tropic.custom_envs.Countdown.env_type == "countdown_step"
    assert tropic.custom_envs.Countdown.max_actions_per_traj == tropic.agent_proxy.max_turn == 5
    assert tropic.custom_envs.Countdown.env_config.max_steps == 5
    assert tropic.es_manager.train.group_size * tropic.tropic.waves_per_iteration == base.es_manager.train.group_size
    assert tropic.es_manager.train.seed_pool_size == 128


def test_original_data_mode_leaves_the_original_config_untouched(tmp_path):
    base = original()
    cfgs = configs(run_script("--dry-run", SAVE_DIR=str(tmp_path / "unused"), DATA="original"))
    for cfg in cfgs[:4]:
        assert cfg.custom_envs == base.custom_envs and cfg.es_manager.val == base.es_manager.val
    tropic = cfgs[4]
    assert tropic.custom_envs.Countdown.env_config.train_path == "data/countdown/train.parquet"
    assert tropic.custom_envs.Countdown.env_config.solvable_filter
    assert tropic.es_manager.val.env_config_overrides.Countdown.train_path == "data/countdown/train.parquet"


def test_pilot_and_pass_at_k_overrides(tmp_path):
    cfgs = configs(run_script("--dry-run", SAVE_DIR=str(tmp_path), EVAL_PROBLEMS="24", EVAL_ATTEMPTS="2",
                             STEPS="2", TEST_FREQ="2", SAVE_FREQ="2", WANDB_ENTITY="my-team"))
    for cfg in cfgs:
        assert cfg.es_manager.val.env_groups == 24 and cfg.es_manager.val.group_size == 2
        assert cfg.trainer.test_freq == 2
        assert cfg.ray_kwargs.ray_init.runtime_env.env_vars.WANDB_ENTITY == "my-team"
    assert all(cfg.trainer.total_training_steps == 2 for cfg in cfgs[1:])
    cfgs = configs(run_script("--dry-run", SAVE_DIR=str(tmp_path), EVAL_ATTEMPTS="8", TEST_FREQ="25"))
    assert all(cfg.es_manager.val.group_size == 8 and cfg.trainer.test_freq == 25 for cfg in cfgs)


@pytest.mark.parametrize("env", [dict(EVAL_PROBLEMS="513"), dict(DATA="paper"), dict(EVAL_ATTEMPTS="0"),
                                dict(GPU="0,0"), dict(GPU="0,1,2"), dict(MICRO_BATCH_SIZE="3"),
                                dict(EXPERIMENT="../bad"), dict(TEST_FREQ="0"), dict(SEED="-1"),
                                dict(RUNS="tropic"), dict(RUNS="EVAL,EVAL"), dict(PROTOCOL="bad"),
                                dict(PROTOCOL="atomic", DATA="original"), dict(WARMUP_MODEL="/tmp/adapter"),
                                dict(PROTOCOL="atomic", WARMUP_STEPS="0"),
                                dict(PROTOCOL="atomic", WARMUP_MICRO_BATCH_SIZE="3")])
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
    (tmp_path / "RAGEN2_Qwen2.5-3B-Instruct_countdown_test_GRPO").mkdir()
    result = run_script(SAVE_DIR=str(tmp_path), PYTHON="/nonexistent")
    assert result.returncode != 0 and "Refusing to overwrite" in result.stderr
    assert len(list(tmp_path.iterdir())) == 1


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
printf 'Mock Countdown training\\n'
if [[ "$*" == *"trainer.experiment_name=RAGEN2_Qwen2.5-3B-Instruct_countdown_test_PPO"* ]]; then
    exit "$FAKE_PPO_STATUS"
fi
""")
    executable.chmod(0o755)
    output = tmp_path / "artifacts"
    result = run_script(SAVE_DIR=str(output), PYTHON=str(executable), FAKE_PPO_STATUS=str(ppo_status))
    assert result.returncode == ppo_status, result.stderr
    prefix = "RAGEN2_Qwen2.5-3B-Instruct_countdown_test_"
    assert (output / (prefix + "EVAL") / "COMPLETED").exists()
    ppo = output / (prefix + "PPO")
    assert (ppo / "exit_code.txt").read_text().strip() == str(ppo_status)
    assert (ppo / "dataset_manifest.json").exists() and (ppo / "command.sh").exists()
    if ppo_status:
        assert not (ppo / "COMPLETED").exists() and not (output / (prefix + "GRPO")).exists()
    else:
        assert all((output / (prefix + m) / "COMPLETED").exists() for m in ALL)


def test_matched_atomic_protocol_has_shared_interface_rewards_and_initial_model(tmp_path):
    from train import add_dependency_and_validate_config
    result = run_script("--dry-run", SAVE_DIR=str(tmp_path / "unused"), PROTOCOL="atomic",
                        GPU="0,1,2,3,4,5,6,7", EVAL_ATTEMPTS="8", TEST_FREQ="25")
    assert "WARMUP_COMMAND " in result.stdout
    assert "torch.distributed.run" in result.stdout and "--nproc_per_node=8" in result.stdout
    cfgs = configs(result, protocol="atomic")
    for cfg in cfgs:
        add_dependency_and_validate_config(cfg)
        assert cfg.model_path == cfgs[0].model_path
        assert cfg.model_path.endswith("countdown_test_atomic_WARMUP/model")
        assert cfg.actor_rollout_ref.model.path == cfg.model_path
        assert cfg.critic.model.path == cfg.model_path
        assert cfg.agent_proxy == cfgs[0].agent_proxy
        assert cfg.custom_envs.Countdown == cfgs[0].custom_envs.Countdown
        assert cfg.custom_envs.Countdown.env_type == "countdown_step"
        assert not cfg.agent_proxy.enable_think
        assert cfg.es_manager.format_penalty == 0
        assert cfg.es_manager.train.seed_pool_size == 128
        assert cfg.actor_rollout_ref.rollout.response_length == 48
        assert cfg.actor_rollout_ref.rollout.val_kwargs.temperature == 1
        assert cfg.es_manager.val.group_size == 8
        assert cfg.trainer.test_freq == 25
        assert not cfg.actor_rollout_ref.actor.use_ref
        assert cfg.actor_rollout_ref.actor.entropy_coeff == 0
    assert cfgs[0].trainer.val_only
    assert cfgs[1].critic.enable and cfgs[3].critic.enable and not cfgs[2].critic.enable
    assert cfgs[2].algorithm.adv_estimator == "grpo"
    assert cfgs[-1].trainer.method == "tropic" and not cfgs[-1].critic.enable
    assert cfgs[-1].es_manager.train.group_size * cfgs[-1].tropic.waves_per_iteration == cfgs[1].es_manager.train.group_size
    assert not (tmp_path / "unused").exists()


def test_atomic_subset_reuses_a_shared_warmup_model(tmp_path):
    result = run_script("--dry-run", PROTOCOL="atomic", RUNS="TROPIC", SAVE_DIR=str(tmp_path),
                        WARMUP_MODEL="/tmp/Qwen-shared-warmup/model")
    cfg, = configs(result, methods=("TROPIC",), protocol="atomic")
    assert cfg.model_path == "/tmp/Qwen-shared-warmup/model"
    assert "WARMUP_COMMAND " not in result.stdout


@pytest.mark.parametrize("warmup_status", [0, 7])
def test_atomic_warmup_runs_once_and_failure_prevents_rl(tmp_path, warmup_status):
    executable = tmp_path / "fake-python"
    executable.write_text("""#!/usr/bin/env bash
if [[ "$1" == "-c" ]]; then exit 0; fi
if [[ "$1" == "-m" && "$2" == "ragen.env.countdown.preflight" ]]; then
    printf '{}\\n' > "$4"; exit 0
fi
if [[ "$1" == "-m" && "$2" == "torch.distributed.run" ]]; then
    if [[ "$FAKE_WARMUP_STATUS" != 0 ]]; then exit "$FAKE_WARMUP_STATUS"; fi
    while [[ "$1" != "--output" ]]; do shift; done
    mkdir -p "$2/model"
    printf '{}\\n' > "$2/warmup_manifest.json"
    touch "$2/COMPLETED"
    exit 0
fi
if [[ "$1" == "-m" && "$2" == "ragen.env.countdown.warmup" ]]; then exit 0; fi
printf 'Mock atomic training\\n'
""")
    executable.chmod(0o755)
    output = tmp_path / "artifacts"
    settings = dict(SAVE_DIR=str(output), PYTHON=str(executable), PROTOCOL="atomic",
                    FAKE_WARMUP_STATUS=str(warmup_status))
    result = run_script(RUNS="EVAL,TROPIC", **settings)
    assert result.returncode == warmup_status, result.stderr
    prefix = "RAGEN2_Qwen2.5-3B-Instruct_countdown_test_atomic_"
    if warmup_status:
        assert not (output / (prefix + "EVAL")).exists()
        assert not (output / (prefix + "TROPIC")).exists()
    else:
        for method in ("EVAL", "TROPIC"):
            directory = output / (prefix + method)
            assert (directory / "COMPLETED").exists()
            assert (directory / "warmup_manifest.json").is_file()
        result = run_script(RUNS="GRPO", **settings)
        assert result.returncode == 0, result.stderr
        assert "WARMUP_COMMAND " not in result.stdout
        assert (output / (prefix + "GRPO") / "COMPLETED").exists()
