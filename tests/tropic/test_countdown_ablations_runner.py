"""Validate the actual ablation commands and execution lifecycle without launching GPUs."""

import json

from omegaconf import OmegaConf
import pytest

from test_countdown_runner import ROOT, configs, run_script

SCRIPT = ROOT / 'scripts/runs/countdown_ablations.sh'
METHODS = ('TROPIC', 'TROPIC_NO_COMPOSITION', 'TROPIC_SUCCESSFUL_FRAGMENTS_ONLY', 'TROPIC_NO_FRONTIER')
CHANGES = ({}, {'composition_enabled': False}, {'successful_fragments_only': True}, {'frontier_enabled': False})


@pytest.mark.parametrize('protocol', ['original', 'atomic'])
def test_ablation_commands_differ_only_in_the_intended_mechanism(tmp_path, protocol):
    from train import add_dependency_and_validate_config
    result = run_script('--dry-run', script=SCRIPT, SAVE_DIR=str(tmp_path / 'space in path'),
                        GPU='0,1,2,3,4,5,6,7', PROTOCOL=protocol, EXPERIMENT='')
    cfgs = configs(result, METHODS, protocol=protocol)
    control = cfgs[0]
    full_options = OmegaConf.to_container(control.tropic)
    normalized = []
    for cfg, change in zip(cfgs, CHANGES):
        add_dependency_and_validate_config(cfg)
        assert OmegaConf.to_container(cfg.tropic) == {**full_options, **change}
        assert cfg.trainer.method == 'tropic' and not cfg.critic.enable
        assert cfg.trainer.total_training_steps == 200 and cfg.trainer.test_freq == 25
        assert cfg.trainer.save_freq == 100 and cfg.trainer.resume_mode == 'disable'
        assert cfg.trainer.val_before_train and cfg.trainer.validation_steps == 1
        assert cfg.seed.train == 10000 and cfg.seed.val == 123
        assert cfg.es_manager.val.env_groups == 512 and cfg.es_manager.val.group_size == 8
        assert cfg.es_manager.train.seed_pool_size == 128
        assert cfg.es_manager.train.group_size * cfg.tropic.waves_per_iteration == 16
        assert cfg.trainer.n_gpus_per_node == 8 and cfg.micro_batch_size_per_gpu == 1
        assert cfg.trainer.project_name == 'RAGEN_COUNTDOWN'
        assert cfg.trainer.logger == ['console', 'file', 'wandb']
        assert 'countdown_ablations_seed10000' in cfg.trainer.experiment_name
        assert cfg.actor_rollout_ref.rollout.val_kwargs.temperature == (1 if protocol == 'atomic' else .5)
        raw = OmegaConf.to_container(cfg)
        raw['tropic'] = full_options
        normalized.append(json.dumps(raw, sort_keys=True).replace(cfg.trainer.experiment_name, 'RUN'))
    assert len(set(normalized)) == 1  
    if protocol == 'original':
        assert all(cfg.model_path == 'Qwen/Qwen2.5-3B-Instruct' for cfg in cfgs)
        assert 'WARMUP_COMMAND ' not in result.stdout
        previous, = configs(run_script('--dry-run', RUNS='TROPIC', TEST_FREQ='25', EVAL_ATTEMPTS='8',
                                       SAVE_DIR=str(tmp_path / 'space in path'), GPU='0,1,2,3,4,5,6,7',
                                       EXPERIMENT='countdown_ablations_seed10000'), methods=('TROPIC',))
        add_dependency_and_validate_config(previous)
        assert previous == control
    else:
        assert result.stdout.count('WARMUP_COMMAND ') == 1
        assert all(cfg.model_path == control.model_path for cfg in cfgs)
    assert not (tmp_path / 'space in path').exists()


def test_subset_retry_uses_only_requested_runs_and_respects_overrides(tmp_path):
    requested = (METHODS[1], METHODS[3])
    result = run_script('--dry-run', script=SCRIPT, RUNS=','.join(reversed(requested)),
                        SAVE_DIR=str(tmp_path), SEED='20000', EXPERIMENT='', GPU='2,3', STEPS='4',
                        EVAL_PROBLEMS='24', EVAL_ATTEMPTS='2', TEST_FREQ='2', SAVE_FREQ='4')
    cfgs = configs(result, requested)
    assert all(cfg.trainer.total_training_steps == 4 and cfg.seed.train == 20000 for cfg in cfgs)
    assert all(cfg.es_manager.val.env_groups == 24 and cfg.es_manager.val.group_size == 2 for cfg in cfgs)
    assert all('countdown_ablations_seed20000' in cfg.trainer.experiment_name for cfg in cfgs)


@pytest.mark.parametrize('settings', [
    {'RUNS': 'PPO'}, {'RUNS': 'TROPIC_TYPO'}, {'RUNS': 'TROPIC,TROPIC'},
    {'RUNS': 'TROPIC,'}, {'DATA': 'original'}, {'EVAL_ATTEMPTS': '0'},
])
def test_invalid_ablation_selection_fails_before_any_launch(tmp_path, settings):
    result = run_script('--dry-run', script=SCRIPT, SAVE_DIR=str(tmp_path / 'unused'), **settings)
    assert result.returncode != 0 and 'COMMAND ' not in result.stdout
    assert not (tmp_path / 'unused').exists()


@pytest.mark.parametrize('failure', [0, 7])
def test_ablation_artifacts_stop_on_failure_and_refuse_overwrite(tmp_path, failure):
    executable = tmp_path / 'fake-python'
    executable.write_text('''#!/usr/bin/env bash
if [[ "$1" == "-m" ]]; then printf '{}\\n' > "$4"; exit 0; fi
if [[ "$1" == "-c" ]]; then exit 0; fi
printf 'Mock Countdown ablation\\n'
if [[ "$*" == *"countdown_test_TROPIC_NO_COMPOSITION"* ]]; then exit "$FAKE_STATUS"; fi
exit 0
''')
    executable.chmod(0o755)
    output = tmp_path / 'artifacts'
    settings = dict(script=SCRIPT, SAVE_DIR=str(output), PYTHON=str(executable), FAKE_STATUS=str(failure))
    result = run_script(**settings)
    assert result.returncode == failure, result.stderr
    prefix = 'RAGEN2_Qwen2.5-3B-Instruct_countdown_test_'
    assert (output / (prefix + 'TROPIC') / 'COMPLETED').is_file()
    ablation = output / (prefix + 'TROPIC_NO_COMPOSITION')
    assert (ablation / 'exit_code.txt').read_text().strip() == str(failure)
    if failure:
        assert not (ablation / 'COMPLETED').exists()
        assert not (output / (prefix + 'TROPIC_SUCCESSFUL_FRAGMENTS_ONLY')).exists()
        assert not (output / (prefix + 'TROPIC_NO_FRONTIER')).exists()
    else:
        for method in METHODS:
            directory = output / (prefix + method)
            assert (directory / 'COMPLETED').is_file()
            assert (directory / 'dataset_manifest.json').is_file()
            assert (directory / 'command.sh').is_file()
    retry = run_script(**settings)
    assert retry.returncode != 0 and 'Refusing to overwrite' in retry.stderr
