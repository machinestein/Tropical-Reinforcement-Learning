from types import SimpleNamespace
import pytest
import torch

from ragen.tropic.batch import basis_rows, make_batch, verified_path_loss


def test_loss_normalizes_problems_paths_and_tokens_not_turns():
    edges = [SimpleNamespace(prompt_ids=(1, 4, 5), completion_ids=(6,) * n) for n in (2, 3, 1, 2)]
    graphs = {'a': SimpleNamespace(edges=dict(enumerate(edges))), 'b': SimpleNamespace(edges={3: edges[3]})}
    rows, weights = basis_rows(graphs, {'a': [(0, 1), (2,)], 'b': [(3,)], 'empty': []})
    batch = make_batch(rows, 0, weights, divisor=6)
    values = torch.tensor([-2., -4., -8., -6., -100., -100.], requires_grad=True)
    log_probs = values[:, None].expand_as(batch.batch['response_mask'])
    loss = verified_path_loss(log_probs, batch.batch['response_mask'], batch.batch['tropic_weights'])
    assert float(loss.detach()) == pytest.approx(5.8)
    loss.backward()
    assert values.grad.tolist() == pytest.approx([-.1, -.15, -.25, -.5, 0., 0.])


def test_padding_eos_and_prompt_masks_preserve_exact_tokens():
    rows = [SimpleNamespace(prompt_ids=(3, 4), completion_ids=(5, 2)),
            SimpleNamespace(prompt_ids=(3,), completion_ids=(2,))]
    batch = make_batch(rows, 0, divisor=3).batch
    assert batch['responses'][0][batch['response_mask'][0].bool()].tolist() == [5, 2]
    assert batch['responses'][1][batch['response_mask'][1].bool()].tolist() == [2]
    assert batch['tropic_weights'].tolist() == [1, 1, 0]
    assert batch['position_ids'][1].tolist() == [0, 0, 0, 1]


def test_microbatch_and_simulated_distributed_gradients_match_full_batch():
    torch.manual_seed(5)
    log_probs = torch.randn(8, 7, requires_grad=True)
    mask = torch.randint(0, 2, (8, 7)).float()
    weights = torch.rand(8)
    expected = torch.autograd.grad(verified_path_loss(log_probs, mask, weights), log_probs)[0]
    rank_losses = []
    for rank in range(2):
        loss = sum(2 * verified_path_loss(log_probs[i:i+2], mask[i:i+2], weights[i:i+2])
                   for i in range(rank * 4, rank * 4 + 4, 2))
        rank_losses.append(loss)
    actual = torch.autograd.grad(sum(rank_losses) / 2, log_probs)[0]
    torch.testing.assert_close(actual, expected)
