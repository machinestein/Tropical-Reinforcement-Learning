import os
from pathlib import Path
import shlex
import subprocess

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/runs/run_lean_light.sh"


def run_script(*args, **env):
    settings = {**os.environ, "MODEL": "Qwen/Qwen2.5-3B-Instruct", "GPU": "0", "SEED": "10000",
                "EXPERIMENT": "lean_test", "MICRO_BATCH_SIZE": "1", "WANDB_PROJECT": "", "RUNS": "",
                "LEAN_MAX_MODEL_LEN": "", **env}
    return subprocess.run(["bash", str(SCRIPT), *args], cwd=ROOT, env=settings,
                          text=True, capture_output=True, timeout=30)


def configs(result, count=5):
    assert result.returncode == 0, result.stderr
    commands = [shlex.split(line.removeprefix('COMMAND ')) for line in result.stdout.splitlines()
                if line.startswith('COMMAND ')]
    assert len(commands) == count
    with initialize_config_dir(config_dir=str(ROOT / 'config'), version_base=None):
        cfgs = []
        for command in commands:
            i = command.index('--config-name')
            cfg = compose(config_name=command[i + 1], overrides=command[i + 2:])
            OmegaConf.resolve(cfg)
            cfgs.append(cfg)
    return cfgs


def test_help_installs_only_lean_client_not_legacy_ragen_dependencies():
    result = run_script('--help')
    assert result.returncode == 0
    assert 'python -m pip install "kimina-client==0.2.1"' in result.stdout
    assert 'pip install -e' not in result.stdout
    assert 'project defaults to RAGEN_LEAN' in result.stdout


def test_light_configs_preserve_original_lean_policy_and_rewards():
    with initialize_config_dir(config_dir=str(ROOT / 'config'), version_base=None):
        original = compose(config_name='_7_lean')
        light = compose(config_name='_7_lean_light')
    for key in ('agent_proxy', 'actor_rollout_ref', 'critic', 'algorithm', 'collapse_detection', 'ctx_manager'):
        assert OmegaConf.to_container(original[key], resolve=True) == OmegaConf.to_container(light[key], resolve=True)
    task = dict(light.custom_envs.Lean)
    task['env_config'] = None
    assert task == dict(original.custom_envs.Lean)
    assert original.es_manager.train == light.es_manager.train
    assert original.custom_envs.Lean.env_config is None
    assert light.trainer.project_name == 'RAGEN_LEAN'


def test_five_commands_validate_and_do_not_create_files(tmp_path):
    from train import add_dependency_and_validate_config
    target = tmp_path / 'not created'
    result = run_script('--dry-run', SAVE_DIR=str(target), GPU='0,1,2,3,4,5,6,7', WANDB_ENTITY='my-team')
    cfgs = configs(result)
    assert not target.exists()
    for cfg, method in zip(cfgs, ['EVAL', 'PPO', 'GRPO', 'SNR', 'TROPIC']):
        add_dependency_and_validate_config(cfg)
        assert cfg.trainer.experiment_name == f'RAGEN2_Qwen2.5-3B-Instruct_lean_test_{method}'
        assert cfg.trainer.n_gpus_per_node == 8
        assert cfg.trainer.test_freq == 25
        assert cfg.es_manager.val.group_size == 8
        assert cfg.es_manager.val.env_groups == 244
        assert cfg.es_manager.val.env_config_overrides.Lean.sample_mode == 'sequential'
        assert cfg.custom_envs.Lean.env_config.dataset_partition == 'valid'
        assert cfg.es_manager.val.env_config_overrides.Lean.dataset_partition == 'test'
        assert cfg.actor_rollout_ref.rollout.val_kwargs.temperature == .5
        assert cfg.trainer.logger == ['console', 'file', 'wandb']
        assert cfg.trainer.project_name == 'RAGEN_LEAN'
        assert cfg.ray_kwargs.ray_init.runtime_env.env_vars.WANDB_ENTITY == 'my-team'
    evaluation, ppo, grpo, snr, tropic = cfgs
    assert evaluation.trainer.val_only and evaluation.trainer.save_freq == -1
    assert ppo.agent_proxy == grpo.agent_proxy == snr.agent_proxy == evaluation.agent_proxy
    assert ppo.agent_proxy.max_turn == 15 and ppo.agent_proxy.max_actions_per_turn == 4
    assert ppo.critic.enable and snr.critic.enable and not grpo.critic.enable
    assert ppo.actor_rollout_ref.actor.loss_agg_mode == 'token-mean'
    assert grpo.actor_rollout_ref.actor.loss_agg_mode == 'seq-mean-token-mean'
    assert snr.actor_rollout_ref.rollout.rollout_filter_value == .9
    assert snr.actor_rollout_ref.rollout.rollout_filter_top_p_prob_mode == 'softmax'
    assert tropic.trainer.method == 'tropic'
    assert tropic.custom_envs.Lean.env_type == 'lean_tropic'
    assert tropic.es_manager.train.group_size * tropic.tropic.waves_per_iteration == ppo.es_manager.train.group_size
    assert tropic.custom_envs.Lean.max_actions_per_traj == ppo.custom_envs.Lean.max_actions_per_traj == 30


def test_lean_project_can_be_overridden_without_changing_run_names(tmp_path):
    cfgs = configs(run_script('--dry-run', SAVE_DIR=str(tmp_path / 'unused'), WANDB_PROJECT='RAGEN_LEAN_DEV'))
    for cfg in cfgs:
        assert cfg.trainer.project_name == 'RAGEN_LEAN_DEV'
        assert cfg.trainer.experiment_name.startswith('RAGEN2_Qwen2.5-3B-Instruct_lean_test_')


def test_long_context_retry_fits_vllm_and_preserves_method_settings(tmp_path):
    from train import add_dependency_and_validate_config
    defaults = configs(run_script('--dry-run', SAVE_DIR=str(tmp_path / 'unused')))
    retries = configs(run_script('--dry-run', SAVE_DIR=str(tmp_path / 'unused'), LEAN_MAX_MODEL_LEN='16384'))
    for original, retry in zip(defaults, retries):
        add_dependency_and_validate_config(retry)
        rollout = retry.actor_rollout_ref.rollout
        assert rollout.max_model_len - rollout.response_length == 15872
        assert rollout.response_length == original.actor_rollout_ref.rollout.response_length == 512
        assert rollout.enable_chunked_prefill
        assert rollout.max_num_batched_tokens >= rollout.max_model_len
        assert retry.actor_rollout_ref.actor.entropy_from_logits_with_chunking
        assert original.actor_rollout_ref.rollout.max_model_len == 8096
        assert original.actor_rollout_ref.rollout.max_num_batched_tokens == 8192
        assert not original.actor_rollout_ref.actor.entropy_from_logits_with_chunking
        for key in ('agent_proxy', 'custom_envs', 'algorithm', 'es_manager'):
            assert retry[key] == original[key]
        assert retry.actor_rollout_ref.actor.entropy_coeff == original.actor_rollout_ref.actor.entropy_coeff
        assert retry.actor_rollout_ref.actor.loss_agg_mode == original.actor_rollout_ref.actor.loss_agg_mode


def test_pilot_settings_keep_original_action_limits_and_separate_project(tmp_path):
    cfgs = configs(run_script('--dry-run', SAVE_DIR=str(tmp_path / 'unused'), GPU='0,1,2,3,4,5,6,7',
                             EXPERIMENT='lean_pilot', STEPS='2', TEST_FREQ='2', SAVE_FREQ='2',
                             EVAL_PROBLEMS='24', EVAL_ATTEMPTS='8'))
    for cfg in cfgs:
        assert cfg.trainer.project_name == 'RAGEN_LEAN'
        assert cfg.es_manager.val.env_groups == 24
        assert cfg.es_manager.val.group_size == 8
        assert cfg.custom_envs.Lean.max_actions_per_traj == 30
    for cfg in cfgs[1:]:
        assert cfg.trainer.total_training_steps == 2
        assert cfg.trainer.test_freq == cfg.trainer.save_freq == 2
    for cfg in cfgs[:4]:
        assert cfg.agent_proxy.max_turn == 15
        assert cfg.agent_proxy.max_actions_per_turn == 4


@pytest.mark.parametrize('env', [dict(EVAL_ATTEMPTS='0'), dict(EVAL_ATTEMPTS='2.5'), dict(GPU='0,0'),
                                dict(GPU='0,1,2'), dict(MICRO_BATCH_SIZE='3'), dict(EXPERIMENT='../bad'),
                                dict(LEAN_TRAIN_PARTITION='test'), dict(LEAN_MAX_MODEL_LEN='0'),
                                dict(LEAN_MAX_MODEL_LEN='512'), dict(LEAN_MAX_MODEL_LEN='16k')])
def test_invalid_settings_fail_without_launching(tmp_path, env):
    result = run_script('--dry-run', SAVE_DIR=str(tmp_path / 'unused'), **env)
    assert result.returncode != 0
    assert 'COMMAND ' not in result.stdout
    assert not (tmp_path / 'unused').exists()


def test_existing_runs_rejected_before_any_preflight(tmp_path):
    (tmp_path / 'RAGEN2_Qwen2.5-3B-Instruct_lean_test_PPO').mkdir()
    result = run_script(SAVE_DIR=str(tmp_path), PYTHON='/nonexistent')
    assert result.returncode != 0
    assert 'Refusing to overwrite' in result.stderr
    assert len(list(tmp_path.iterdir())) == 1


def test_preflight_failure_does_not_create_run_directories(tmp_path):
    target = tmp_path / 'unused'
    result = run_script(SAVE_DIR=str(target), PYTHON='false')
    assert result.returncode != 0
    assert 'preflight failed' in result.stderr
    assert not target.exists()


def test_custom_microbatch_leaves_baseline_loss_definition_unchanged(tmp_path):
    cfgs = configs(run_script('--dry-run', SAVE_DIR=str(tmp_path), GPU='0,1,2,3,4,5,6,7',
                             MICRO_BATCH_SIZE='4', EVAL_ATTEMPTS='4', TEST_FREQ='50', STEPS='100'))
    for cfg in cfgs:
        assert cfg.micro_batch_size_per_gpu == 4
        assert cfg.es_manager.val.group_size == 4
        assert cfg.trainer.test_freq == 50
    assert cfgs[1].actor_rollout_ref.actor.loss_agg_mode == 'token-mean'


@pytest.mark.parametrize('ppo_status', [0, 7])
def test_execution_artifacts_and_failure_propagation(tmp_path, ppo_status):
    executable = tmp_path / 'fake-python'
    executable.write_text('''#!/usr/bin/env bash
if [[ "$1" == "-m" ]]; then printf '{}\\n' > "$4"; exit 0; fi
if [[ "$1" == "-c" ]]; then exit 0; fi
printf 'Mock Lean training\\n'
if [[ "$*" == *"trainer.experiment_name=RAGEN2_Qwen2.5-3B-Instruct_lean_test_PPO"* ]]; then
    exit "$FAKE_PPO_STATUS"
fi
''')
    executable.chmod(0o755)
    output = tmp_path / 'artifacts'
    result = run_script(SAVE_DIR=str(output), PYTHON=str(executable), FAKE_PPO_STATUS=str(ppo_status))
    assert result.returncode == ppo_status, result.stderr
    prefix = 'RAGEN2_Qwen2.5-3B-Instruct_lean_test_'
    assert (output / (prefix + 'EVAL') / 'COMPLETED').exists()
    ppo = output / (prefix + 'PPO')
    assert (ppo / 'exit_code.txt').read_text().strip() == str(ppo_status)
    assert 'Mock Lean training' in (ppo / 'train.log').read_text()
    assert (ppo / 'dataset_manifest.json').exists()
    assert (ppo / 'command.sh').exists()
    if ppo_status:
        assert not (ppo / 'COMPLETED').exists()
        assert not (output / (prefix + 'GRPO')).exists()
    else:
        assert all((output / (prefix + method) / 'COMPLETED').exists()
                   for method in ('EVAL', 'PPO', 'GRPO', 'SNR', 'TROPIC'))


def test_runs_selects_a_subset_in_canonical_order(tmp_path):
    cfgs = configs(run_script('--dry-run', RUNS='TROPIC', SAVE_DIR=str(tmp_path / 'unused')), count=1)
    assert cfgs[0].trainer.experiment_name.endswith('_TROPIC') and cfgs[0].trainer.method == 'tropic'
    cfgs = configs(run_script('--dry-run', RUNS='TROPIC,EVAL', SAVE_DIR=str(tmp_path / 'unused')), count=2)
    assert [c.trainer.experiment_name.rsplit('_', 1)[-1] for c in cfgs] == ['EVAL', 'TROPIC']
    for bad in ('tropic', 'EVAL,EVAL', 'EVAL,', ',EVAL', 'EVAL,,TROPIC', 'EVAL TROPIC'):
        result = run_script('--dry-run', RUNS=bad, SAVE_DIR=str(tmp_path / 'unused'))
        assert result.returncode != 0 and 'RUNS' in result.stderr, bad
    assert len(configs(run_script('--dry-run', RUNS='', SAVE_DIR=str(tmp_path / 'unused')))) == 5


def test_overwrite_check_covers_only_selected_runs(tmp_path):
    executable = tmp_path / 'fake-python'
    executable.write_text("""#!/usr/bin/env bash
if [[ "$1" == "-m" ]]; then printf '{}\\n' > "$4"; exit 0; fi
if [[ "$1" == "-c" ]]; then exit 0; fi
printf 'Mock Lean training\\n'
""")
    executable.chmod(0o755)
    output = tmp_path / 'saves'
    (output / 'RAGEN2_Qwen2.5-3B-Instruct_lean_test_EVAL').mkdir(parents=True)  
    result = run_script(RUNS='TROPIC', PYTHON=str(executable), SAVE_DIR=str(output))
    assert result.returncode == 0, result.stderr
    assert sorted(p.name.rsplit('_', 1)[-1] for p in output.iterdir()) == ['EVAL', 'TROPIC']
    assert (output / 'RAGEN2_Qwen2.5-3B-Instruct_lean_test_TROPIC' / 'COMPLETED').exists()
    result = run_script(RUNS='EVAL,TROPIC', PYTHON=str(executable), SAVE_DIR=str(output))
    assert result.returncode != 0 and 'Refusing to overwrite' in result.stderr
