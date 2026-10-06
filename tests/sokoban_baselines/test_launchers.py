import os
from pathlib import Path
import shlex
import subprocess

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest

from ragen.sokoban_baselines.config import validate_config

ROOT = Path(__file__).resolve().parents[2]


def launch(method, *args, **env):
    return subprocess.run(['bash', str(ROOT / f'scripts/runs/run_sokoban_{method}.sh'), *args],
                          env={**os.environ, **env}, cwd='/tmp', text=True, capture_output=True, timeout=30)


def parsed_config(result):
    assert result.returncode == 0, result.stderr
    commands = [shlex.split(line[8:]) for line in result.stdout.splitlines() if line.startswith('COMMAND ')]
    assert len(commands) == 1
    command = commands[0]
    pos = command.index('--config-name')
    with initialize_config_dir(config_dir=str(ROOT / 'config'), version_base=None):
        config = compose(config_name=command[pos + 1], overrides=command[pos + 2:])
    OmegaConf.resolve(config)
    return config


@pytest.mark.parametrize('method', ['maxrl', 'tstar'])
@pytest.mark.parametrize('gpu', ['0', '0,1,2,3,4,5,6,7'])
def test_launchers_compose_validate_and_preserve_light_task(tmp_path, method, gpu):
    cfg = parsed_config(launch(method, '--dry-run', GPU=gpu, SAVE_DIR=str(tmp_path / 'unused folder')))
    validate_config(cfg)
    from train import add_dependency_and_validate_config
    assert add_dependency_and_validate_config(cfg).data.train_batch_size == 128
    with initialize_config_dir(config_dir=str(ROOT / 'config'), version_base=None):
        light = compose(config_name='_2_sokoban')
        OmegaConf.resolve(light)
    assert cfg.agent_proxy == light.agent_proxy
    assert cfg.custom_envs.CoordSokoban == light.custom_envs.CoordSokoban
    assert cfg.es_manager.train == light.es_manager.train
    assert cfg.actor_rollout_ref.rollout.val_kwargs == light.actor_rollout_ref.rollout.val_kwargs
    assert cfg.es_manager.val.group_size == 8
    assert cfg.trainer.total_training_steps == 200
    assert cfg.trainer.experiment_name.endswith('_' + method.upper())
    assert cfg.trainer.logger == ['console', 'file', 'wandb']
    assert not (tmp_path / 'unused folder').exists()


@pytest.mark.parametrize('method', ['maxrl', 'tstar'])
def test_existing_folders_rejected_before_python_invocation(tmp_path, method):
    folder = tmp_path / f'RAGEN2_Qwen2.5-3B-Instruct_sokoban_seed10000_{method.upper()}'
    folder.mkdir()
    sentinel = folder / 'metrics.jsonl'
    sentinel.write_text('preserve this')
    result = launch(method, SAVE_DIR=str(tmp_path), PYTHON='/does/not/exist')
    assert result.returncode == 2
    assert 'Refusing to overwrite' in result.stderr
    assert sentinel.read_text() == 'preserve this'


@pytest.mark.parametrize('env', [{'GPU': '0,0'}, {'STEPS': '0'}, {'SEED': '-1'}, {'EXPERIMENT': '../old'},
                                  {'TSTAR_KL_SAMPLES': '0'}, {'TSTAR_GRAFT_BATCH_SIZE': '-2'}])
def test_bad_arguments_fail_without_side_effects(env):
    assert launch('tstar', '--dry-run', **env).returncode == 2


@pytest.mark.parametrize('override', ['actor_rollout_ref.actor.ppo_epochs=2', 'trainer.resume_mode=auto',
                                     'actor_rollout_ref.rollout.temperature=0.5', 'critic.enable=true',
                                     'actor_rollout_ref.actor.use_ref=true', 'tstar.gamma=nan'])
def test_unsupported_settings_fail_early(override):
    with initialize_config_dir(config_dir=str(ROOT / 'config'), version_base=None):
        cfg = compose(config_name='_2_sokoban_tstar', overrides=[override])
    with pytest.raises(ValueError):
        validate_config(cfg)
