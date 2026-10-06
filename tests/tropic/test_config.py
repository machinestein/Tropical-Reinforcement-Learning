from pathlib import Path
from hydra import compose, initialize_config_dir
import pytest
from omegaconf import OmegaConf

from ragen.tropic.config import validate_tropic_config


def test_tropic_configuration_passes_validation(config):
    validate_tropic_config(config)


def test_successful_fragments_only_rejects_nonboolean_or_unsupported_task(config):
    OmegaConf.update(config, 'tropic.successful_fragments_only', 'false', force_add=True)
    with pytest.raises(ValueError, match='boolean'):
        validate_tropic_config(config)
    config.tropic.successful_fragments_only = True
    validate_tropic_config(config)
    config.custom_envs.CoordSokoban.env_type = 'webshop_tropic'
    with pytest.raises(ValueError, match='supports Sokoban'):
        validate_tropic_config(config)


@pytest.mark.parametrize('field,value', [('context_window_mode', 'full'), ('max_actions_per_turn', 2),
                                      ('terminate_on_invalid_action', False)])
def test_invalid_interfaces_fail_early(config, field, value):
    config.agent_proxy[field] = value
    with pytest.raises(ValueError):
        validate_tropic_config(config)


def test_baselines_and_tropic_share_task_and_scheduled_attempt_budget():
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[2] / 'config'), version_base=None):
        baseline = compose(config_name='_2_sokoban_state')
        tropic = compose(config_name='_2_sokoban_tropic')
    assert baseline.custom_envs.CoordSokoban == tropic.custom_envs.CoordSokoban
    assert baseline.agent_proxy == tropic.agent_proxy
    assert baseline.es_manager.train.seed_pool_size == tropic.es_manager.train.seed_pool_size
    assert baseline.es_manager.train.group_size == tropic.es_manager.train.group_size * tropic.tropic.waves_per_iteration


@pytest.mark.parametrize('name,overrides', [
    ('_2_sokoban_state', []),
    ('_2_sokoban_state', ['algorithm.adv_estimator=grpo', 'critic.enable=false']),
    ('_2_sokoban_state', ['actor_rollout_ref.rollout.rollout_filter_strategy=top_p',
                        'actor_rollout_ref.rollout.rollout_filter_value=0.9']),
    ('_2_sokoban_tropic', ['tropic.collection_only=true', 'trainer.total_training_steps=20']),
    ('_2_sokoban_tropic', ['tropic.frontier_enabled=false', 'tropic.composition_enabled=false']),
    ('_2_sokoban_tropic', ['+tropic.successful_fragments_only=true']),
])
def test_documented_launch_overrides_compose_and_validate(name, overrides):
    from train import add_dependency_and_validate_config
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[2] / 'config'), version_base=None):
        cfg = compose(config_name=name, overrides=overrides)
    assert add_dependency_and_validate_config(cfg).data.train_batch_size > 0
