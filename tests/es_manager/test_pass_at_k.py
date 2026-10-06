from math import comb

import pytest

from ragen.llm_agent.es_manager import EnvStateManager, EnvStatus, pass_at_k, pass_at_k_levels


@pytest.mark.parametrize(('attempts', 'correct', 'k', 'expected'), [
    (1, 0, 1, 0.0), (1, 1, 1, 1.0),
    (8, 0, 4, 0.0), (8, 8, 4, 1.0), (8, 5, 4, 1.0),
    (8, 1, 1, 1 / 8), (8, 1, 8, 1.0),
    (8, 2, 4, 1 - comb(6, 4) / comb(8, 4)),
    (16, 3, 8, 1 - comb(13, 8) / comb(16, 8)),
])
def test_pass_at_k_matches_combinatorial_estimator(attempts, correct, k, expected):
    assert pass_at_k(attempts, correct, k) == pytest.approx(expected)


def test_pass_at_k_rejects_invalid_k():
    with pytest.raises(ValueError):
        pass_at_k(4, 1, 0)
    with pytest.raises(ValueError):
        pass_at_k(4, 1, 5)


@pytest.mark.parametrize(('attempts', 'levels'), [
    (1, [1]), (2, [1, 2]), (4, [1, 2, 4]), (8, [1, 2, 4, 8]), (16, [1, 2, 4, 8, 16]),
    (6, [1, 2, 4, 6]), (12, [1, 2, 4, 8, 12]),
])
def test_pass_at_k_levels_are_powers_of_two_plus_group_size(attempts, levels):
    assert pass_at_k_levels(attempts) == levels


def fake_manager(successes_per_group):
    """Build a manager with finished episodes; group g has the given per-attempt outcomes."""
    manager = EnvStateManager.__new__(EnvStateManager)
    manager.group_size = len(successes_per_group[0])
    manager.envs, manager.rollout_cache = [], []
    for group_id, outcomes in enumerate(successes_per_group):
        for outcome in outcomes:
            status = EnvStatus(terminated=bool(outcome), truncated=not outcome, num_actions=3)
            manager.envs.append({'tag': 'Sokoban', 'group_id': group_id, 'env': object(), 'status': status})
            manager.rollout_cache.append({'tag': 'Sokoban', 'history': [{'info': {}}, {}]})
    return manager


def test_rollout_states_log_nested_pass_at_k_per_group():
    manager = fake_manager([[1, 0, 0, 0], [0, 0, 0, 0], [1, 1, 0, 1]])
    caches = manager.get_rollout_states()
    assert len(caches) == 12
    for cache in caches[:4]:
        assert cache['metrics']['Sokoban/pass@1'] == pytest.approx(1 / 4)
        assert cache['metrics']['Sokoban/pass@2'] == pytest.approx(1 - comb(3, 2) / comb(4, 2))
        assert cache['metrics']['Sokoban/pass@4'] == 1.0
    for cache in caches[4:8]:
        assert all(cache['metrics'][f'Sokoban/pass@{k}'] == 0.0 for k in (1, 2, 4))
    for cache in caches[8:]:
        assert cache['metrics']['Sokoban/pass@1'] == pytest.approx(3 / 4)
        assert cache['metrics']['Sokoban/pass@4'] == 1.0
    successes = [cache['metrics']['Sokoban/success'] for cache in caches]
    pass1 = [cache['metrics']['Sokoban/pass@1'] for cache in caches]
    assert sum(successes) / len(successes) == pytest.approx(sum(pass1) / len(pass1))
    assert not any('pass@8' in cache['metrics'] for cache in caches)


def test_single_attempt_groups_only_log_pass_at_1():
    manager = fake_manager([[1], [0], [1]])
    caches = manager.get_rollout_states()
    for cache in caches:
        assert set(k for k in cache['metrics'] if 'pass@' in k) == {'Sokoban/pass@1'}
        assert cache['metrics']['Sokoban/pass@1'] == cache['metrics']['Sokoban/success']
