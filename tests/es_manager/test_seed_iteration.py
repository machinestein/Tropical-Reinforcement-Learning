import pytest
from omegaconf import OmegaConf
from ragen.llm_agent.es_manager import EnvStateManager


def make_cfg():
    return OmegaConf.create({
        'seed': {'train': 7},
        'es_manager': {
            'train': {
                'env_groups': 1,
                'group_size': 1,
                'env_configs': {'tags': ['Bandit'], 'n_groups': [1]},
            }
        },
        'custom_envs': {
            'Bandit': {
                'env_type': 'bandit',
                'max_actions_per_traj': 1,
                'env_config': None
            }
        }
    })


def test_seed_iteration():
    cfg = make_cfg()
    es = EnvStateManager(cfg, mode='train')
    es.reset()
    first_seed = es.envs[0]['status'].seed
    es.reset()
    second_seed = es.envs[0]['status'].seed
    assert first_seed == 7
    assert second_seed == 8
