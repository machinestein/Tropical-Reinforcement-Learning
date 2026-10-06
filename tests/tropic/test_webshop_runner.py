"""Launch contracts: original WebShop configs, optional evaluation overrides, no side effects."""

import os
from pathlib import Path
import shlex
import subprocess

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/runs/run_webshop_light.sh"


def run_script(*args, **env):
    settings = {**os.environ, "MODEL": "Qwen/Qwen2.5-3B-Instruct", "GPU": "0", "SEED": "10000",
                "EXPERIMENT": "webshop_test", "MICRO_BATCH_SIZE": "", "WANDB_PROJECT": "",
                "STEPS": "200", "TEST_FREQ": "10", "SAVE_FREQ": "100", "EVAL_ATTEMPTS": "1",
                "EVAL_PROBLEMS": "512", "RUNS": "", "TROPIC_COVERAGE": "true", **env}
    return subprocess.run(["bash", str(SCRIPT), *args], cwd=ROOT, env=settings,
                          text=True, capture_output=True, timeout=30)


def configs(result, methods=("EVAL", "PPO", "GRPO", "SNR", "TROPIC")):
    assert result.returncode == 0, result.stderr
    commands = [shlex.split(line.removeprefix("COMMAND ")) for line in result.stdout.splitlines()
                if line.startswith("COMMAND ")]
    assert len(commands) == len(methods)
    assert [c[c.index("--config-name") + 1] for c in commands] == [
        "_6_webshop_tropic" if method == "TROPIC" else "_6_webshop" for method in methods]
    cfgs = []
    with initialize_config_dir(config_dir=str(ROOT / "config"), version_base=None):
        for command in commands:
            index = command.index("--config-name")
            cfg = compose(config_name=command[index + 1], overrides=command[index + 2:])
            OmegaConf.resolve(cfg)
            cfgs.append(cfg)
    assert [cfg.trainer.experiment_name.rsplit("_", 1)[-1] for cfg in cfgs] == list(methods)
    return cfgs


def test_help_and_defaults():
    result = run_script("--help")
    assert result.returncode == 0
    assert "_6_webshop directly" in result.stdout
    assert "microbatch 1" in result.stdout and "RAGEN_WEBSHOP" in result.stdout
    assert "RUNS=TROPIC" in result.stdout


def test_original_baseline_policy_reward_and_batches_are_unchanged(tmp_path):
    from train import add_dependency_and_validate_config
    cfgs = configs(run_script("--dry-run", SAVE_DIR=str(tmp_path / "unused"), GPU="0,1,2,3,4,5,6,7"))
    with initialize_config_dir(config_dir=str(ROOT / "config"), version_base=None):
        original = compose(config_name="_6_webshop")
        OmegaConf.resolve(original)
    for cfg in cfgs:
        add_dependency_and_validate_config(cfg)
        assert cfg.trainer.n_gpus_per_node == 8
        assert cfg.trainer.project_name == "RAGEN_WEBSHOP"
        assert original.micro_batch_size_per_gpu == 4 and cfg.micro_batch_size_per_gpu == 1
        assert cfg.actor_rollout_ref.actor.entropy_from_logits_with_chunking
        assert cfg.ppo_mini_batch_size == original.ppo_mini_batch_size == 32
        assert cfg.es_manager.val == original.es_manager.val
        assert cfg.actor_rollout_ref.rollout.val_kwargs == original.actor_rollout_ref.rollout.val_kwargs
        assert cfg.actor_rollout_ref.rollout.max_model_len == 15000
        assert cfg.actor_rollout_ref.rollout.response_length == original.actor_rollout_ref.rollout.response_length
        assert cfg.trainer.test_freq == original.trainer.test_freq == 10
        assert cfg.agent_proxy.max_turn == 9
    for cfg in cfgs[:4]:
        for key in ("agent_proxy", "custom_envs", "collapse_detection", "ctx_manager"):
            assert cfg[key] == original[key]
        assert cfg.es_manager.train == original.es_manager.train
        assert cfg.actor_rollout_ref.actor.entropy_coeff == original.actor_rollout_ref.actor.entropy_coeff
        assert cfg.actor_rollout_ref.actor.optim == original.actor_rollout_ref.actor.optim
    evaluation, ppo, grpo, snr, tropic = cfgs
    assert evaluation.trainer.val_only and not evaluation.critic.enable and not evaluation.actor_rollout_ref.actor.use_ref
    from verl.trainer.ppo.utils import need_critic
    assert ppo.critic.enable == snr.critic.enable == need_critic(original)
    microbatch_keys = {"enable", "ppo_micro_batch_size_per_gpu", "forward_micro_batch_size_per_gpu"}
    assert {k: v for k, v in ppo.critic.items() if k not in microbatch_keys} == {
        k: v for k, v in original.critic.items() if k not in microbatch_keys}
    assert ppo.critic.ppo_micro_batch_size_per_gpu == ppo.critic.forward_micro_batch_size_per_gpu == 1
    expected_actor = OmegaConf.to_container(original.actor_rollout_ref.actor)
    expected_actor["entropy_from_logits_with_chunking"] = True  
    expected_actor["ppo_micro_batch_size_per_gpu"] = 1  
    assert (OmegaConf.to_container(ppo.actor_rollout_ref.actor) == OmegaConf.to_container(snr.actor_rollout_ref.actor)
            == expected_actor)
    assert ppo.algorithm == snr.algorithm == original.algorithm
    assert grpo.algorithm.adv_estimator == "grpo" and not grpo.critic.enable
    assert grpo.actor_rollout_ref.actor.loss_agg_mode == "seq-mean-token-mean"
    assert ppo.actor_rollout_ref.rollout.rollout_filter_value == 1.0
    assert snr.actor_rollout_ref.rollout.rollout_filter_value == .9
    assert snr.actor_rollout_ref.rollout.rollout_filter_top_p_prob_mode == "softmax"
    assert tropic.custom_envs.WebShop.env_type == "webshop_tropic"
    assert tropic.custom_envs.WebShop.env_config.dataset == "small"
    assert tropic.es_manager.train.group_size * tropic.tropic.waves_per_iteration == original.es_manager.train.group_size
    assert not (tmp_path / "unused").exists()


def test_pilot_and_pass_at_k_are_explicit_shared_overrides(tmp_path):
    cfgs = configs(run_script("--dry-run", SAVE_DIR=str(tmp_path), GPU="0,1,2,3,4,5,6,7",
                             EVAL_PROBLEMS="24", EVAL_ATTEMPTS="2", STEPS="2", TEST_FREQ="2", SAVE_FREQ="2",
                             WANDB_ENTITY="my-team", WANDB_PROJECT="webshop-development"))
    for cfg in cfgs:
        assert cfg.es_manager.val.env_groups == 24 and cfg.es_manager.val.group_size == 2
        assert cfg.trainer.test_freq == 2
        assert cfg.trainer.project_name == "webshop-development"
        assert cfg.ray_kwargs.ray_init.runtime_env.env_vars.WANDB_ENTITY == "my-team"
    assert all(cfg.trainer.total_training_steps == 2 for cfg in cfgs[1:])
    cfgs = configs(run_script("--dry-run", SAVE_DIR=str(tmp_path), EVAL_ATTEMPTS="8", TEST_FREQ="25"))
    assert all(cfg.es_manager.val.group_size == 8 and cfg.trainer.test_freq == 25 for cfg in cfgs)


def test_coverage_ablation_changes_only_tropic_basis_selection(tmp_path):
    default = configs(run_script("--dry-run", SAVE_DIR=str(tmp_path)))
    ablated = configs(run_script("--dry-run", SAVE_DIR=str(tmp_path), TROPIC_COVERAGE="false"))
    assert default[-1].tropic.coverage_enabled is True
    assert ablated[-1].tropic.coverage_enabled is False
    ablated[-1].tropic.coverage_enabled = True
    assert [OmegaConf.to_container(c, resolve=True) for c in default] == [
        OmegaConf.to_container(c, resolve=True) for c in ablated]
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("env", [dict(EVAL_PROBLEMS="1001"), dict(EVAL_ATTEMPTS="0"), dict(GPU="0,0"),
                                dict(GPU="0,1,2"), dict(MICRO_BATCH_SIZE="3"), dict(EXPERIMENT="../bad"),
                                dict(TEST_FREQ="0"), dict(SEED="-1"), dict(TROPIC_COVERAGE="0")])
def test_invalid_options_rejected_before_launch(tmp_path, env):
    result = run_script("--dry-run", SAVE_DIR=str(tmp_path / "unused"), **env)
    assert result.returncode != 0 and "COMMAND " not in result.stdout
    assert not (tmp_path / "unused").exists()


def test_existing_run_is_not_overwritten(tmp_path):
    (tmp_path / "RAGEN2_Qwen2.5-3B-Instruct_webshop_test_GRPO").mkdir()
    result = run_script(SAVE_DIR=str(tmp_path), PYTHON="/nonexistent")
    assert result.returncode != 0 and "Refusing to overwrite" in result.stderr
    assert len(list(tmp_path.iterdir())) == 1


def test_preflight_failure_creates_no_artifacts(tmp_path):
    result = run_script(SAVE_DIR=str(tmp_path / "unused"), PYTHON="false")
    assert result.returncode != 0 and "preflight failed" in result.stderr
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize("runs,methods", [("TROPIC", ("TROPIC",)), ("GRPO", ("GRPO",)),
                                         ("TROPIC,EVAL", ("EVAL", "TROPIC")),
                                         ("", ("EVAL", "PPO", "GRPO", "SNR", "TROPIC"))])
def test_runs_selects_subset_in_canonical_order(tmp_path, runs, methods):
    cfgs = configs(run_script("--dry-run", RUNS=runs, SAVE_DIR=str(tmp_path / "unused")), methods)
    for method, cfg in zip(methods, cfgs):
        assert cfg.trainer.method == ("tropic" if method == "TROPIC" else "ppo")
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize("runs", ["tropic", "UNKNOWN", "EVAL,EVAL", "EVAL,", ",EVAL",
                                  "EVAL,,TROPIC", "EVAL TROPIC", " "])
def test_invalid_runs_rejected_before_launch(tmp_path, runs):
    result = run_script("--dry-run", RUNS=runs, SAVE_DIR=str(tmp_path / "unused"))
    assert result.returncode != 0 and "RUNS" in result.stderr
    assert "COMMAND " not in result.stdout and not (tmp_path / "unused").exists()


def test_only_selected_run_is_checked_and_can_retry_after_archiving(tmp_path):
    executable = tmp_path / "fake-python"
    executable.write_text('''#!/usr/bin/env bash
if [[ "$1" == "-m" ]]; then printf '{}\\n' > "$4"; exit 0; fi
if [[ "$1" == "-c" ]]; then exit 0; fi
printf 'Mock WebShop training\\n'
''')
    executable.chmod(0o755)
    output = tmp_path / "saves"
    prefix = "RAGEN2_Qwen2.5-3B-Instruct_webshop_test_"
    evaluation, tropic = output / (prefix + "EVAL"), output / (prefix + "TROPIC")
    evaluation.mkdir(parents=True)
    (evaluation / "metrics.jsonl").write_text("existing evaluation\n")
    tropic.mkdir()
    (tropic / "exit_code.txt").write_text("7\n")
    result = run_script(RUNS="TROPIC", PYTHON="/nonexistent", SAVE_DIR=str(output))
    assert result.returncode != 0 and "Refusing to overwrite" in result.stderr
    assert str(tropic) in result.stderr
    archived = tmp_path / "failed-tropic"
    tropic.rename(archived)
    result = run_script(RUNS="TROPIC", PYTHON=str(executable), SAVE_DIR=str(output))
    assert result.returncode == 0, result.stderr
    assert {path.name for path in output.iterdir()} == {evaluation.name, tropic.name}
    assert (tropic / "COMPLETED").exists()
    assert (tropic / "exit_code.txt").read_text() == "0\n"
    assert (evaluation / "metrics.jsonl").read_text() == "existing evaluation\n"
    assert (archived / "exit_code.txt").read_text() == "7\n"


@pytest.mark.parametrize("ppo_status", [0, 7])
def test_execution_artifacts_and_failure_propagation(tmp_path, ppo_status):
    executable = tmp_path / "fake-python"
    executable.write_text('''#!/usr/bin/env bash
if [[ "$1" == "-m" ]]; then printf '{}\\n' > "$4"; exit 0; fi
if [[ "$1" == "-c" ]]; then exit 0; fi
printf 'Mock WebShop training\\n'
if [[ "$*" == *"trainer.experiment_name=RAGEN2_Qwen2.5-3B-Instruct_webshop_test_PPO"* ]]; then
    exit "$FAKE_PPO_STATUS"
fi
''')
    executable.chmod(0o755)
    output = tmp_path / "artifacts"
    result = run_script(SAVE_DIR=str(output), PYTHON=str(executable), FAKE_PPO_STATUS=str(ppo_status))
    assert result.returncode == ppo_status, result.stderr
    prefix = "RAGEN2_Qwen2.5-3B-Instruct_webshop_test_"
    assert (output / (prefix + "EVAL") / "COMPLETED").exists()
    assert (output / (prefix + "PPO") / "exit_code.txt").read_text().strip() == str(ppo_status)
    assert (output / (prefix + "PPO") / "dataset_manifest.json").exists()
    assert (output / (prefix + "TROPIC") / "COMPLETED").exists() is (ppo_status == 0)
