import os
from pathlib import Path
import shlex
import subprocess

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'scripts/runs/run_sokoban_tropic_suite.sh'
LIGHT_SCRIPT = ROOT / 'scripts/runs/run_sokoban_light.sh'
ABLATION_SCRIPT = ROOT / 'scripts/runs/sokoban_ablations.sh'


def run_script(*args, env=None, cwd=ROOT):
    return subprocess.run(['bash', str(SCRIPT), *map(str, args)], cwd=cwd,
                          env=env, text=True, capture_output=True, timeout=30)


def commands(result):
    assert result.returncode == 0, result.stderr
    return [shlex.split(line.removeprefix('COMMAND ')) for line in result.stdout.splitlines()
            if line.startswith('COMMAND ')]


def configs(result):
    result_configs = {}
    with initialize_config_dir(config_dir=str(ROOT / 'config'), version_base=None):
        for command in commands(result):
            i = command.index('--config-name')
            cfg = compose(config_name=command[i + 1], overrides=command[i + 2:])
            OmegaConf.resolve(cfg)
            result_configs[cfg.trainer.experiment_name] = cfg
    return result_configs


@pytest.fixture(scope='module')
def all_configs(tmp_path_factory):
    output = tmp_path_factory.mktemp('suite') / 'not-created'
    result = run_script('--suite', 'all', '--dry-run', '--output-dir', output)
    assert not output.exists()
    return configs(result)


def test_all_commands_compose_and_pass_real_entrypoint_validation(all_configs):
    from train import add_dependency_and_validate_config
    assert len(all_configs) == 16
    for cfg in all_configs.values():
        assert add_dependency_and_validate_config(cfg).data.train_batch_size > 0
        assert cfg.seed.val == 123
        assert cfg.trainer.validation_steps == 1
        assert cfg.es_manager.val.env_groups == 512
        assert cfg.es_manager.val.env_configs.n_groups == [512]
        assert cfg.es_manager.val.group_size == 1
        assert cfg.actor_rollout_ref.rollout.val_kwargs.do_sample
        assert cfg.actor_rollout_ref.rollout.val_kwargs.temperature == .5
        assert cfg.trainer.resume_mode == 'disable'
        assert cfg.trainer.logger == ['console', 'file']
        assert cfg.ray_kwargs.ray_init.runtime_env.env_vars.VERL_FILE_LOGGER_PATH == str(
            Path(cfg.trainer.default_local_dir).parent / 'metrics.jsonl')


def test_matched_baselines_and_original_protocol_remain_distinct(all_configs):
    ppo = all_configs['ppo_seed10000']
    grpo = all_configs['grpo_seed10000']
    tropic = all_configs['tropic_seed10000']
    assert ppo.agent_proxy == grpo.agent_proxy == tropic.agent_proxy
    assert ppo.custom_envs.CoordSokoban == tropic.custom_envs.CoordSokoban
    assert ppo.es_manager.train.seed_pool_size == tropic.es_manager.train.seed_pool_size
    assert ppo.es_manager.train.group_size == tropic.es_manager.train.group_size * tropic.tropic.waves_per_iteration
    assert ppo.critic.enable
    assert not grpo.critic.enable
    assert grpo.algorithm.adv_estimator == 'grpo'
    assert grpo.actor_rollout_ref.actor.loss_agg_mode == 'seq-mean-token-mean'
    for prefix in ('ppo', 'grpo', 'original_ppo'):
        assert all_configs[f'{prefix}_seed10000'].actor_rollout_ref.rollout.rollout_filter_value == 1.0
        assert all_configs[f'{prefix}_snr_seed10000'].actor_rollout_ref.rollout.rollout_filter_value == .9
    original = all_configs['original_ppo_seed10000']
    assert original.agent_proxy.context_window_mode == 'full'
    assert original.agent_proxy.max_actions_per_turn == 2
    assert original.actor_rollout_ref.actor.entropy_coeff == .001
    assert 'seed_pool_size' not in original.es_manager.train
    assert original.actor_rollout_ref.rollout.rollout_filter_top_p_prob_mode == 'softmax'
    original_grpo = all_configs['original_grpo_seed10000']
    assert original_grpo.agent_proxy == original.agent_proxy
    assert original_grpo.algorithm.adv_estimator == 'grpo'
    assert not original_grpo.critic.enable


def test_base_evaluations_cannot_update_or_resume(all_configs):
    for name in ('base_state_eval123', 'base_original_eval123'):
        cfg = all_configs[name]
        assert cfg.trainer.val_only and cfg.trainer.val_before_train
        assert cfg.trainer.save_freq == -1
        assert not cfg.critic.enable
        assert not cfg.actor_rollout_ref.actor.use_ref


def test_ablations_change_only_the_named_tropic_mechanism(all_configs):
    full = OmegaConf.to_container(all_configs['tropic_seed10000'].tropic)
    changes = {
        'verified_replay': {'frontier_enabled': False, 'composition_enabled': False},
        'tropic_no_frontier': {'frontier_enabled': False},
        'tropic_no_composition': {'composition_enabled': False},
        'tropic_successful_fragments_only': {'successful_fragments_only': True},
        'tropic_no_coverage': {'coverage_enabled': False},
        'tropic_basis1': {'basis_size': 1},
    }
    for name, changed in changes.items():
        assert OmegaConf.to_container(all_configs[f'{name}_seed10000'].tropic) == {**full, **changed}


def test_multiseed_sweep_evaluates_each_base_only_once(tmp_path):
    cfgs = configs(run_script('--suite', 'all', '--seeds', '10000,20000,30000',
                             '--dry-run', '--output-dir', tmp_path / 'unused'))
    assert len(cfgs) == 44
    assert len([name for name in cfgs if name.startswith('base_')]) == 2
    assert {c.seed.train for name, c in cfgs.items() if not name.startswith('base_')} == {10000, 20000, 30000}


def test_paths_with_spaces_multi_gpu_and_custom_evaluation(tmp_path):
    cfgs = configs(run_script('--runs', 'base_state,tropic_pilot', '--steps', '20', '--gpus', '2,4',
                             '--model', 'Qwen/a model,with=punctuation', '--dry-run',
                             '--eval-seed', '500', '--eval-problems', '20',
                             '--output-dir', tmp_path / 'space in path', cwd=tmp_path))
    for cfg in cfgs.values():
        assert cfg.model_path == 'Qwen/a model,with=punctuation'
        assert cfg.trainer.n_gpus_per_node == 2
        assert cfg.system.CUDA_VISIBLE_DEVICES == '2,4'
        assert cfg.seed.val == 500
        assert cfg.es_manager.val.env_groups == 20
    assert cfgs['tropic_pilot_seed10000'].tropic.collection_only
    assert cfgs['tropic_pilot_seed10000'].trainer.total_training_steps == 20


@pytest.mark.parametrize('args', [
    ['--suite', 'missing'], ['--runs', 'typo'], ['--runs', 'ppo,ppo'], ['--runs', 'ppo,'],
    ['--steps'], ['--steps', '0'], ['--steps', '08'], ['--save-freq', '0'],
    ['--test-freq', '0'], ['--seeds', '10000,10000'], ['--gpus', '0,0'],
    ['--gpus', '0,'], ['--seeds', '123'], ['--unknown'],
    ['--name-prefix', 'bad/name'],
    ['--name-prefix', 'valid', '--seeds', '10000,20000'],
    ['--eval-mode'], ['--eval-mode', 'typo'],
    ['--eval-attempts'], ['--eval-attempts', '0'], ['--eval-attempts', '08'],
    ['--eval-attempts', '8', '--eval-mode', 'greedy'],
    ['--runs', 'base_state,base_original', '--name-prefix', 'same'],
    ['--runs', 'ppo,original_ppo', '--name-prefix', 'same'],
    ['--runs', 'grpo,original_grpo', '--name-prefix', 'same'],
    ['--runs', 'ppo_snr,original_ppo_snr', '--name-prefix', 'same'],
    ['--micro-batch-size'], ['--micro-batch-size', '0'], ['--micro-batch-size', '-1'],
    ['--micro-batch-size', '04'], ['--micro-batch-size', '3'],
    ['--gpus', '0,1,2,3,4,5,6,7', '--micro-batch-size', '8'],
])
def test_invalid_cli_or_overlapping_holdout_fails_before_launch(args, tmp_path):
    output = tmp_path / 'unused'
    result = run_script(*args, '--dry-run', '--output-dir', output)
    assert result.returncode != 0
    assert 'Error:' in result.stderr
    assert not output.exists()


@pytest.fixture
def fake_python(tmp_path):
    executable = tmp_path / 'fake-python'
    executable.write_text('''#!/usr/bin/env bash
if [[ "$1" == "-c" ]]; then exit "${FAKE_PREFLIGHT_STATUS:-0}"; fi
printf '%s\\n' "$*" >> "$FAKE_CALL_LOG"
printf 'Mock training invocation\\n'
if [[ "$*" == *"trainer.experiment_name=ppo_seed10000"* ]]; then
    exit "${FAKE_PPO_STATUS:-0}"
fi
exit 0
''')
    executable.chmod(0o755)
    env = {**os.environ, 'FAKE_CALL_LOG': str(tmp_path / 'calls.txt')}
    return executable, env


def test_execution_logging_skip_and_overwrite_protection(tmp_path, fake_python):
    executable, env = fake_python
    output = tmp_path / 'artifacts'
    args = ['--runs', 'base_state,ppo', '--python', executable, '--output-dir', output]
    result = run_script(*args, env=env)
    assert result.returncode == 0, result.stderr
    for name in ('base_state_eval123', 'ppo_seed10000'):
        directory = output / name
        assert (directory / 'COMPLETED').exists()
        assert (directory / 'exit_code.txt').read_text().strip() == '0'
        assert 'Mock training' in (directory / 'train.log').read_text()
        assert 'train.py' in (directory / 'command.sh').read_text()
    assert run_script(*args, env=env).returncode != 0
    assert run_script(*args, '--skip-completed', env=env).returncode == 0
    assert len(Path(env['FAKE_CALL_LOG']).read_text().splitlines()) == 2
    assert run_script(*args, '--skip-completed', '--steps', '2', env=env).returncode != 0


def test_failed_preflight_never_starts_jobs_or_creates_artifacts(tmp_path, fake_python):
    executable, env = fake_python
    env['FAKE_PREFLIGHT_STATUS'] = '1'
    output = tmp_path / 'not-created'
    result = run_script('--python', executable, '--output-dir', output, env=env)
    assert result.returncode != 0
    assert 'no experiments started' in result.stderr
    assert not output.exists()
    assert not Path(env['FAKE_CALL_LOG']).exists()


@pytest.mark.parametrize('continue_on_error', [False, True])
def test_training_failure_is_not_hidden_by_tee(tmp_path, fake_python, continue_on_error):
    executable, env = fake_python
    env['FAKE_PPO_STATUS'] = '7'
    output = tmp_path / 'artifacts'
    args = ['--runs', 'ppo,grpo', '--python', executable, '--output-dir', output]
    if continue_on_error:
        args.append('--continue-on-error')
    result = run_script(*args, env=env)
    assert result.returncode != 0
    assert (output / 'ppo_seed10000/exit_code.txt').read_text().strip() == '7'
    assert not (output / 'ppo_seed10000/COMPLETED').exists()
    assert (output / 'grpo_seed10000/COMPLETED').exists() == continue_on_error


def light_env(**overrides):
    settings = {'MODEL', 'GPU', 'MICRO_BATCH_SIZE', 'STEPS', 'TEST_FREQ', 'SAVE_FREQ', 'EVAL_ATTEMPTS', 'SEED', 'EXPERIMENT',
                'SAVE_DIR', 'PYTHON', 'WANDB_PROJECT', 'WANDB_ENTITY', 'WANDB_MODE'}
    return {**{key: value for key, value in os.environ.items() if key not in settings}, **overrides}


def run_light(*args, env=None):
    return subprocess.run(['bash', str(LIGHT_SCRIPT), *args], cwd=ROOT, env=env or light_env(),
                          text=True, capture_output=True, timeout=30)


def run_ablations(*args, env=None):
    return subprocess.run(['bash', str(ABLATION_SCRIPT), *args], cwd=ROOT, env=env or light_env(),
                          text=True, capture_output=True, timeout=30)


def test_focused_ablations_match_light_tropic_except_named_mechanism(tmp_path):
    from train import add_dependency_and_validate_config
    env = light_env(GPU='0,1,2,3,4,5,6,7', SAVE_DIR=str(tmp_path / 'unused'))
    control = list(configs(run_light('--dry-run', env=env)).values())[-1]
    cfgs = configs(run_ablations('--dry-run', env=env))
    variants = [('TROPIC_NO_COMPOSITION', {'composition_enabled': False}),
                ('TROPIC_SUCCESSFUL_FRAGMENTS_ONLY', {'successful_fragments_only': True}),
                ('TROPIC_NO_FRONTIER', {'frontier_enabled': False})]
    assert list(cfgs) == ['RAGEN2_Qwen2.5-3B-Instruct_sokoban_ablations_seed10000_' + name
                          for name, _ in variants]
    for cfg, (_, changed) in zip(cfgs.values(), variants):
        add_dependency_and_validate_config(cfg)
        assert OmegaConf.to_container(cfg.tropic) == {**OmegaConf.to_container(control.tropic), **changed}
        for key in ('model_path', 'agent_proxy', 'custom_envs', 'actor_rollout_ref', 'critic',
                    'algorithm', 'seed', 'es_manager', 'collapse_detection'):
            assert cfg[key] == control[key]
        assert cfg.micro_batch_size_per_gpu == 1
        assert cfg.trainer.n_gpus_per_node == 8
        assert cfg.trainer.total_training_steps == 200
        assert cfg.trainer.test_freq == 25 and cfg.trainer.save_freq == 100
        assert cfg.es_manager.val.env_groups == 512 and cfg.es_manager.val.group_size == 8
        assert cfg.trainer.val_before_train and not cfg.trainer.val_only
        assert cfg.trainer.resume_mode == 'disable'
        assert cfg.trainer.project_name == 'RAGEN2'
        assert cfg.trainer.logger == ['console', 'file', 'wandb']
        assert cfg.ray_kwargs.ray_init.runtime_env.env_vars.WANDB_MODE == 'disabled'
    assert not (tmp_path / 'unused').exists()


def test_focused_ablations_accept_light_settings_and_execute_exactly_three_jobs(tmp_path, fake_python):
    executable, fake_env = fake_python
    output = tmp_path / 'ablations'
    env = light_env(PYTHON=str(executable), SAVE_DIR=str(output), GPU='2,4',
                    MICRO_BATCH_SIZE='2', STEPS='3', TEST_FREQ='2', SAVE_FREQ='2',
                    EVAL_ATTEMPTS='16', SEED='20000', EXPERIMENT='ablation_trial',
                    WANDB_PROJECT='my-project', WANDB_MODE='offline',
                    FAKE_CALL_LOG=fake_env['FAKE_CALL_LOG'])
    cfgs = configs(run_ablations('--dry-run', env=env))
    for cfg in cfgs.values():
        assert cfg.micro_batch_size_per_gpu == 2
        assert cfg.trainer.n_gpus_per_node == 2
        assert cfg.trainer.total_training_steps == 3
        assert cfg.trainer.test_freq == cfg.trainer.save_freq == 2
        assert cfg.es_manager.val.group_size == 16
        assert cfg.seed.train == 20000
        assert cfg.trainer.project_name == 'my-project'
        assert cfg.ray_kwargs.ray_init.runtime_env.env_vars.WANDB_MODE == 'offline'
    result = run_ablations(env=env)
    assert result.returncode == 0, result.stderr
    assert len(Path(fake_env['FAKE_CALL_LOG']).read_text().splitlines()) == 3
    assert {p.name for p in output.iterdir()} == set(cfgs)
    assert all((p / 'COMPLETED').exists() for p in output.iterdir())
    assert run_ablations(env=env).returncode != 0  
    assert len(Path(fake_env['FAKE_CALL_LOG']).read_text().splitlines()) == 3


def test_focused_ablations_reject_expansion_and_invalid_settings(tmp_path):
    assert run_ablations('--help').returncode == 0
    assert run_ablations('--runs', 'tropic').returncode != 0
    assert run_ablations('--dry-run', '--suite', 'all').returncode != 0
    for settings in ({'EVAL_ATTEMPTS': '0'}, {'SEED': '123'}, {'GPU': '0,0'}):
        result = run_ablations('--dry-run', env=light_env(SAVE_DIR=str(tmp_path / 'unused'), **settings))
        assert result.returncode != 0
        assert 'COMMAND ' not in result.stdout
    assert not (tmp_path / 'unused').exists()


def test_light_defaults_are_exactly_five_local_runs_under_saves():
    from train import add_dependency_and_validate_config
    cfgs = configs(run_light('--dry-run'))
    prefix = 'RAGEN2_Qwen2.5-3B-Instruct_sokoban_seed10000'
    assert list(cfgs) == [f'{prefix}_{method}' for method in ('EVAL', 'PPO', 'GRPO', 'SNR', 'TROPIC')]
    for name, cfg in cfgs.items():
        add_dependency_and_validate_config(cfg)
        folder = ROOT / 'saves' / name
        assert cfg.trainer.project_name == 'RAGEN2'
        assert cfg.trainer.logger == ['console', 'file', 'wandb']
        assert cfg.model_path == 'Qwen/Qwen2.5-3B-Instruct'
        assert cfg.trainer.default_local_dir == str(folder / 'checkpoints')
        assert cfg.trainer.validation_data_dir == str(folder / 'validation')
        runtime_env = cfg.ray_kwargs.ray_init.runtime_env.env_vars
        assert runtime_env.WANDB_DIR == str(folder)
        assert runtime_env.WANDB_MODE == 'disabled'
        assert cfg.trainer.val_before_train
        assert cfg.micro_batch_size_per_gpu == 1
        assert cfg.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu == 1
        assert cfg.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu == 1
        assert cfg.critic.ppo_micro_batch_size_per_gpu == 1
        assert cfg.critic.forward_micro_batch_size_per_gpu == 1
        assert cfg.ppo_mini_batch_size == 32
        assert cfg.trainer.test_freq == 25
        assert cfg.es_manager.val.env_groups == 512
        assert cfg.es_manager.val.group_size == 8
        assert cfg.actor_rollout_ref.rollout.val_kwargs.do_sample
        if not name.endswith('_EVAL'):
            assert cfg.trainer.total_training_steps == 200
            assert cfg.trainer.save_freq == 100
        else:
            assert cfg.trainer.val_only and cfg.trainer.save_freq == -1
    assert cfgs[prefix + '_SNR'].algorithm.adv_estimator == 'gae'
    assert cfgs[prefix + '_SNR'].actor_rollout_ref.rollout.rollout_filter_value == .9
    assert cfgs[prefix + '_TROPIC'].trainer.method == 'tropic'


@pytest.mark.parametrize('microbatch', [1, 4])
def test_light_baselines_inherit_original_protocol_without_tropic_overrides(microbatch):
    result = run_light('--dry-run', env=light_env(
        GPU='0,1,2,3,4,5,6,7', MICRO_BATCH_SIZE=str(microbatch)))
    cfgs = configs(result)
    with initialize_config_dir(config_dir=str(ROOT / 'config'), version_base=None):
        original = compose(config_name='_2_sokoban', overrides=[
            f'micro_batch_size_per_gpu={microbatch}', 'trainer.n_gpus_per_node=8'])
        OmegaConf.resolve(original)
    for command, cfg in zip(commands(result), cfgs.values()):
        method = cfg.trainer.experiment_name.rsplit('_', 1)[-1]
        assert cfg.actor_rollout_ref.rollout.val_kwargs == original.actor_rollout_ref.rollout.val_kwargs
        assert not any(arg.startswith('actor_rollout_ref.rollout.val_kwargs.') for arg in command)
        if method == 'TROPIC':
            assert command[command.index('--config-name') + 1] == '_2_sokoban_tropic'
            assert cfg.trainer.method == 'tropic'
            assert cfg.agent_proxy.context_window_mode == 'single_turn'
            assert cfg.agent_proxy.terminate_on_invalid_action
            assert cfg.es_manager.train.seed_pool_size == 128
            continue
        assert command[command.index('--config-name') + 1] == '_2_sokoban'
        assert 'tropic' not in cfg
        assert cfg.trainer.method == 'ppo'
        assert cfg.agent_proxy == original.agent_proxy
        assert cfg.es_manager.train == original.es_manager.train
        assert cfg.es_manager.format_penalty == original.es_manager.format_penalty
        assert cfg.custom_envs.CoordSokoban == original.custom_envs.CoordSokoban
        assert cfg.collapse_detection == original.collapse_detection
        if method != 'GRPO':
            assert cfg.critic.loss_agg_mode == original.critic.loss_agg_mode == 'token-mean'
        expected_actor = OmegaConf.to_container(original.actor_rollout_ref.actor)
        expected_algorithm = OmegaConf.to_container(original.algorithm)
        if method == 'EVAL':
            expected_actor['use_ref'] = False
            assert cfg.trainer.val_only and not cfg.critic.enable
        elif method == 'GRPO':
            expected_actor['loss_agg_mode'] = 'seq-mean-token-mean'
            expected_algorithm['adv_estimator'] = 'grpo'
            assert not cfg.critic.enable
        else:
            assert cfg.critic.enable
        assert OmegaConf.to_container(cfg.actor_rollout_ref.actor) == expected_actor
        assert OmegaConf.to_container(cfg.algorithm) == expected_algorithm
        expected_rollout = OmegaConf.to_container(original.actor_rollout_ref.rollout)
        expected_rollout['rollout_filter_top_p_prob_mode'] = 'softmax'
        expected_rollout['rollout_filter_value'] = .9 if method == 'SNR' else 1.0
        assert OmegaConf.to_container(cfg.actor_rollout_ref.rollout) == expected_rollout


def test_greedy_evaluation_requires_explicit_opt_in(tmp_path):
    cfgs = configs(run_script('--runs', 'base_original,original_grpo,tropic',
                             '--eval-mode', 'greedy', '--dry-run',
                             '--output-dir', tmp_path / 'unused'))
    for cfg in cfgs.values():
        assert not cfg.actor_rollout_ref.rollout.val_kwargs.do_sample
        assert cfg.actor_rollout_ref.rollout.val_kwargs.temperature == 0


def test_light_validation_and_checkpoint_intervals_can_be_overridden():
    cfgs = configs(run_light('--dry-run', env=light_env(TEST_FREQ='5', SAVE_FREQ='50', EVAL_ATTEMPTS='1')))
    for name, cfg in cfgs.items():
        assert cfg.trainer.test_freq == 5
        assert cfg.trainer.save_freq == (-1 if name.endswith('_EVAL') else 50)
        assert cfg.es_manager.val.group_size == 1


def test_suite_eval_attempts_default_to_one_and_apply_to_every_run(tmp_path):
    result = run_script('--runs', 'base_original,original_ppo,grpo_snr,tropic', '--eval-attempts', '4',
                        '--dry-run', '--output-dir', tmp_path / 'unused')
    assert '4 attempt(s) per board (pass@1..4)' in result.stdout
    cfgs = configs(result)
    assert len(cfgs) == 4
    for cfg in cfgs.values():
        assert cfg.es_manager.val.env_groups == 512
        assert cfg.es_manager.val.group_size == 4
        assert cfg.es_manager.train.group_size in (8, 16)  
    default = configs(run_script('--runs', 'tropic', '--dry-run', '--output-dir', tmp_path / 'unused2'))
    assert default['tropic_seed10000'].es_manager.val.group_size == 1


def test_light_custom_account_and_names_do_not_expose_api_keys(tmp_path):
    output = tmp_path / 'not-created'
    result = run_light('--dry-run', env=light_env(
        GPU='1', STEPS='30', SEED='20000', EXPERIMENT='sokoban_trial02', SAVE_DIR=str(output),
        WANDB_PROJECT='my-project', WANDB_ENTITY='my-team', WANDB_MODE='offline',
        WANDB_API_KEY='not-a-real-key'))
    assert 'not-a-real-key' not in result.stdout + result.stderr
    cfgs = configs(result)
    assert len(cfgs) == 5
    for name, cfg in cfgs.items():
        assert name.startswith('RAGEN2_Qwen2.5-3B-Instruct_sokoban_trial02_')
        assert cfg.trainer.project_name == 'my-project'
        assert cfg.ray_kwargs.ray_init.runtime_env.env_vars.WANDB_ENTITY == 'my-team'
        assert cfg.ray_kwargs.ray_init.runtime_env.env_vars.WANDB_MODE == 'offline'
        assert cfg.system.CUDA_VISIBLE_DEVICES == '1'
        assert cfg.seed.train == 20000
        assert Path(cfg.trainer.default_local_dir).parent.parent == output
    assert not output.exists()


@pytest.mark.parametrize('microbatch', [1, 2, 4])
def test_eight_gpu_microbatches_preserve_global_batches_and_budgets(tmp_path, microbatch):
    from train import add_dependency_and_validate_config
    cfgs = configs(run_script('--runs', 'ppo,grpo,ppo_snr,tropic',
                             '--gpus', '0,1,2,3,4,5,6,7', '--micro-batch-size', microbatch,
                             '--dry-run', '--output-dir', tmp_path / 'unused'))
    for name, cfg in cfgs.items():
        add_dependency_and_validate_config(cfg)
        assert cfg.actor_rollout_ref.actor.ppo_mini_batch_size == 32
        assert cfg.critic.ppo_mini_batch_size == 32
        assert cfg.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu == microbatch
        assert cfg.critic.ppo_micro_batch_size_per_gpu == microbatch
        assert cfg.critic.forward_micro_batch_size_per_gpu == microbatch
        assert cfg.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu == microbatch
        assert cfg.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu == microbatch
        assert cfg.actor_rollout_ref.actor.fsdp_config.fsdp_size == -1
        assert cfg.trainer.n_gpus_per_node == 8
        assert cfg.es_manager.train.env_groups == 8
        assert cfg.es_manager.val.env_groups == 512
        if name.startswith('tropic'):
            assert cfg.es_manager.train.group_size * cfg.tropic.waves_per_iteration == 16
            assert cfg.actor_rollout_ref.actor.ppo_epochs == 2
        else:
            assert cfg.es_manager.train.group_size == 16
            assert cfg.actor_rollout_ref.actor.ppo_epochs == 1
        if name.startswith('ppo'):
            mode = 'token-mean' if microbatch == 1 else 'seq-mean-token-mean'
            assert cfg.actor_rollout_ref.actor.loss_agg_mode == mode
            assert cfg.critic.loss_agg_mode == mode
        elif name.startswith('grpo'):
            assert cfg.actor_rollout_ref.actor.loss_agg_mode == 'seq-mean-token-mean'


def test_light_microbatch_can_be_overridden(tmp_path):
    cfgs = configs(run_light('--dry-run', env=light_env(
        GPU='0,1,2,3,4,5,6,7', MICRO_BATCH_SIZE='2', SAVE_DIR=str(tmp_path / 'unused'))))
    assert len(cfgs) == 5
    assert all(cfg.micro_batch_size_per_gpu == 2 for cfg in cfgs.values())


def test_invalid_microbatch_fails_before_preflight_or_artifacts(tmp_path, fake_python):
    executable, env = fake_python
    output = tmp_path / 'not-created'
    result = run_script('--runs', 'ppo', '--gpus', '0,1,2,3,4,5,6,7', '--micro-batch-size', '8',
                        '--python', executable, '--output-dir', output, env=env)
    assert result.returncode != 0
    assert 'must be divisible' in result.stderr
    assert not output.exists()
    assert not Path(env['FAKE_CALL_LOG']).exists()


def test_light_executes_five_jobs_with_matching_folder_names(tmp_path, fake_python):
    executable, env = fake_python
    output = tmp_path / 'saves'
    result = run_light(env=light_env(PYTHON=str(executable), SAVE_DIR=str(output),
                                    FAKE_CALL_LOG=env['FAKE_CALL_LOG']))
    assert result.returncode == 0, result.stderr
    folders = list(output.iterdir())
    assert len(folders) == 5
    for folder in folders:
        assert (folder / 'COMPLETED').is_file()
        command = (folder / 'command.sh').read_text()
        assert f'trainer.experiment_name={folder.name}' in command
    assert len(Path(env['FAKE_CALL_LOG']).read_text().splitlines()) == 5


def test_light_rejects_suite_expansion_and_supports_help():
    assert run_light('--suite', 'all').returncode != 0
    assert run_light('--dry-run', '--runs', 'tropic').returncode != 0
    result = run_light('--help')
    assert result.returncode == 0
    assert 'wandb login' in result.stdout
