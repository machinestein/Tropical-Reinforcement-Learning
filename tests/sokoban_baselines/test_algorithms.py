import copy
import itertools

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from ragen.sokoban_baselines.batch import token_batch, training_batch
from ragen.sokoban_baselines.core import Decision, build_tree, maxrl_advantages, policy_loss, surgical_loss, thought_prefix


def test_maxrl_weights_hard_groups_and_zeroes_uniform_groups():
    rewards = [1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1]
    groups = [0] * 4 + [1] * 4 + [2] * 2 + [3] * 2
    np.testing.assert_allclose(maxrl_advantages(rewards, groups), [3, -1, -1, -1, 1, 1, -1, -1, 0, 0, 0, 0], atol=2e-7)
    with pytest.raises(ValueError):
        maxrl_advantages([.2, 1], [0, 0])


@pytest.mark.parametrize('n', [2, 3, 5])
def test_maxrl_expected_gradient_matches_n_minus_one_truncated_objective(n):
    p = .23
    expected_gradient = 0.0
    for outcomes in itertools.product([0., 1.], repeat=n):
        r = np.asarray(outcomes)
        probability = p ** r.sum() * (1 - p) ** (n - r.sum())
        expected_gradient += probability * np.mean(maxrl_advantages(r, [0] * n, 1e-12) * (r - p))
    assert expected_gradient == pytest.approx((1 - p) * (1 - (1 - p) ** (n - 1)), abs=1e-10)


def branching_rows():
    return [
        Decision((1,), (2,), 0, 0, 0, (1,), (10,), 1),
        Decision((10,), (3, 4), 0, 0, 1, (1, 2), None, 1),
        Decision((1,), (5, 6), 1, 0, 0, (1,), (11,), 0),
        Decision((11,), (7,), 1, 0, 1, (1, 3), None, 0),
    ]


def test_tree_merges_policy_equivalent_history_and_backs_up_frequencies():
    rows = branching_rows()
    calls = []
    tree = build_tree(rows, lambda a, b: calls.append((a, b)) or .1)
    node = tree.components[0]
    assert node == tree.components[2]
    assert len(calls) == 1
    assert tree.q[node] == pytest.approx(.99 / 2)
    assert tree.advantages[0] == pytest.approx(tree.advantages[2])
    assert tree.advantages[1] > 0 > tree.advantages[3]
    assert tree.divergences == [(1, 3, 1.0)]
    rows += [Decision((1,), (2,), 2, 0, 0, (1,), (10,), 1),
             Decision((10,), (3,), 2, 0, 1, (1, 2), None, 1)]
    weighted = build_tree(rows, lambda a, b: 0)
    assert weighted.q[weighted.components[0]] == pytest.approx(.99 * 2 / 3)


def test_tree_never_merges_different_problems_actions_depths_or_terminal_outcomes():
    rows = branching_rows()
    rows += [Decision((1,), (2,), 2, 1, 0, (1,), None, 1)]
    tree = build_tree(rows, lambda a, b: 2)
    assert len(tree.members) == len(rows)
    rows[2].action_history = (9,)
    assert len(build_tree(rows, lambda *args: pytest.fail('incompatible histories queried')).members) == len(rows)


def test_tree_connected_components_are_transitive_but_levels_remain_acyclic():
    rows = [Decision((1,), (2,), ep, 0, 0, (1,), (10 + ep,), ep % 2) for ep in range(3)]
    rows += [Decision((10 + ep,), (3,), ep, 0, 1, (1, ep), None, ep % 2) for ep in range(3)]
    tree = build_tree(rows, lambda a, b: .1 if abs(a.episode - b.episode) == 1 else 1)
    assert len({tree.components[i] for i in range(3)}) == 1
    assert set(tree.components[i] for i in range(3)).isdisjoint(tree.components[i] for i in range(3, 6))


def test_masks_preserve_exact_ids_and_exclude_context():
    rows = [Decision((1, 2, 3), (4, 5)), Decision((6,), (7,))]
    batch = token_batch(rows, 0).batch
    assert batch['response_mask'].tolist() == [[0, 0, 1, 1], [0, 0, 0, 1]]
    assert batch['responses'][batch['response_mask'].bool()].tolist() == [4, 5, 7]
    logp = torch.randn(2, 4, requires_grad=True)
    policy_loss(logp, batch['response_mask'], torch.ones(2), torch.ones(2)).backward()
    assert (logp.grad[~batch['response_mask'].bool()] == 0).all()


@pytest.mark.parametrize('method', ['maxrl', 'tstar'])
@pytest.mark.parametrize('world,micro', [(1, 1), (2, 2), (8, 1)])
def test_global_normalization_and_padding_are_independent_of_gpu_count(method, world, micro):
    rows = branching_rows()
    pairs = [(rows[0], rows[2])]
    batch = training_batch(rows, [1, 1, -1, -1], pairs, 0, world, micro, method)
    normalized = (batch.batch['response_mask'].sum(-1) * batch.batch['baseline_weights']).sum()
    assert normalized.item() == pytest.approx(1)
    assert batch.batch['pair_weights'].sum().item() == pytest.approx(2)
    for shard in batch.chunk(world):
        n_rl = batch.meta_info['baseline_rl_rows_per_rank']
        assert len(shard) == n_rl + 2 * batch.meta_info['baseline_pairs_per_rank']
        assert shard.batch['baseline_weights'][n_rl:].sum() == 0


class CharTokenizer:
    @staticmethod
    def decode(ids, **kwargs):
        return ''.join(map(chr, ids))


def test_thought_only_targets_exclude_actions_and_incomplete_thoughts():
    text = '<think>push right</think><answer>Right</answer>'
    assert CharTokenizer.decode(thought_prefix(CharTokenizer, tuple(map(ord, text)))) == '<think>push right</think>'
    assert thought_prefix(CharTokenizer, tuple(map(ord, '<answer>Right</answer>'))) == ()
    assert thought_prefix(CharTokenizer, tuple(map(ord, '<think>unfinished'))) == ()


def make_actor(module, method):
    from ragen.sokoban_baselines.actor import DataParallelSokobanBaselineActor

    class CPUActor(DataParallelSokobanBaselineActor):
        def __init__(self):
            self.config = OmegaConf.create({'ppo_epochs': 1, 'ppo_micro_batch_size_per_gpu': 1,
                                           'clip_ratio_low': .2, 'grad_clip': 1000})
            self.actor_module = module
            self.actor_optimizer = torch.optim.SGD(module.parameters(), lr=.1)

        def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False):
            logits = self.actor_module(micro_batch['input_ids'][:, :-1])
            return None, logits.log_softmax(-1).gather(-1, micro_batch['responses'].unsqueeze(-1)).squeeze(-1)

    return CPUActor()


@pytest.mark.parametrize('method', ['maxrl', 'tstar'])
def test_actual_actor_update_matches_full_autograd_objective(method):
    from ragen.sokoban_baselines.actor import ShardEMA
    torch.manual_seed(7)
    module = torch.nn.Embedding(20, 20)
    actor = make_actor(module, method)
    rows = branching_rows()
    pairs = [(rows[0], rows[2])] if method == 'tstar' else []
    batch = training_batch(rows, [1, 1, -1, -1], pairs, 0, 1, 1, method)
    batch.meta_info.update(tstar_beta=.1, tstar_surgical_weight=.15, tstar_ema_alpha=.95)
    _, logp = actor._forward_micro_batch(batch.batch, 1)
    batch.batch['old_log_probs'] = logp.detach().clone()
    if method == 'tstar':
        actor.ema = ShardEMA(module)
        for value in actor.ema.values.values():
            value.mul_(.5)
        with actor.ema.applied(module), torch.no_grad():
            _, reference = actor._forward_micro_batch(batch.batch, 1)
        _, logp = actor._forward_micro_batch(batch.batch, 1)
    loss = policy_loss(logp, batch.batch['response_mask'], batch.batch['baseline_advantages'],
                       batch.batch['baseline_weights'], batch.batch['old_log_probs'] if method == 'tstar' else None)
    if pairs:
        sums = (logp * batch.batch['response_mask']).sum(-1)
        ref = (reference * batch.batch['response_mask']).sum(-1)
        loss = loss + .15 * surgical_loss(sums[-2], sums[-1], ref[-2], ref[-1], .1)
    expected_grad, = torch.autograd.grad(loss, tuple(module.parameters()))
    before = module.weight.detach().clone()
    metrics = actor.update_policy(batch)
    torch.testing.assert_close(module.weight, before - .1 * expected_grad, atol=2e-7, rtol=1e-6)
    assert metrics['actor/optimizer_steps'] == 1
    if method == 'tstar':
        torch.testing.assert_close(actor.ema.values['weight'], .95 * before * .5 + .05 * module.weight.detach())


def test_ema_restores_parameters_after_exception():
    from ragen.sokoban_baselines.actor import ShardEMA
    module = torch.nn.Linear(2, 1)
    ema = ShardEMA(module)
    with torch.no_grad():
        module.weight.add_(1)
    original = copy.deepcopy(module.state_dict())
    with pytest.raises(RuntimeError), ema.applied(module):
        torch.testing.assert_close(module.weight, original['weight'] - 1)
        raise RuntimeError('test')
    torch.testing.assert_close(module.weight, original['weight'])


@pytest.mark.parametrize('method', ['maxrl', 'tstar'])
def test_averaged_rank_gradients_match_single_rank_with_zero_weight_padding(method):
    torch.manual_seed(19)
    initial = torch.nn.Embedding(20, 20)
    rows = branching_rows()
    pairs = [(rows[0], rows[2])] if method == 'tstar' else []
    results = []
    for world in (1, 3):
        batch = training_batch(rows, [1, 1, -1, -1], pairs, 0, world, 1, method)
        batch.meta_info.update(tstar_beta=.1, tstar_surgical_weight=.15, tstar_ema_alpha=.95)
        values = []
        for shard in batch.chunk(world):
            actor = make_actor(copy.deepcopy(initial), method)
            with torch.no_grad():
                _, old = actor._forward_micro_batch(shard.batch, 1)
            shard.batch['old_log_probs'] = old
            actor.update_policy(shard)
            values.append(actor.actor_module.weight.detach())
        results.append(torch.stack(values).mean(0))
    torch.testing.assert_close(results[0], results[1], atol=3e-7, rtol=1e-6)


def test_zero_signal_skips_optimizer_and_preserves_weights():
    actor = make_actor(torch.nn.Embedding(20, 20), 'maxrl')
    rows = branching_rows()
    batch = training_batch(rows, [0] * len(rows), [], 0, 1, 1, 'maxrl')
    batch.meta_info['baseline_skip_update'] = True
    before = actor.actor_module.weight.detach().clone()
    assert actor.update_policy(batch)['actor/optimizer_steps'] == 0
    torch.testing.assert_close(actor.actor_module.weight, before, atol=0, rtol=0)
