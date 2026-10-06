"""Entropy values, training gradients, and saved activation storage must agree."""

import pytest
import torch

from ragen.workers.actor.entropy import entropy_from_logits_recomputed


def reference(logits):
    values = logits if logits.dtype == torch.float64 else logits.float()
    return values.logsumexp(-1) - (values.softmax(-1) * values).sum(-1)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("chunk_size", [1, 7, 256])
def test_entropy_and_weighted_gradient_match(dtype, chunk_size):
    torch.manual_seed(12)
    source = (torch.randn(19, 2, 61) * 3).to(dtype)
    actual_logits = source[:, 0].detach().requires_grad_()
    expected_logits = actual_logits.detach().clone().requires_grad_()
    weights = torch.randn(19)
    weights[::3] = 0
    actual = entropy_from_logits_recomputed(actual_logits, chunk_size)
    expected = reference(expected_logits)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    tolerance = 5e-3 if dtype == torch.bfloat16 else 1e-3 if dtype == torch.float16 else 1e-5
    torch.testing.assert_close(actual_logits.grad, expected_logits.grad, rtol=tolerance, atol=1e-5)
    assert torch.count_nonzero(actual_logits.grad[::3]) == 0


def test_numerical_gradient():
    torch.manual_seed(4)
    logits = torch.randn(5, 7, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: entropy_from_logits_recomputed(x, 2), (logits,))


def test_combined_policy_entropy_loss_through_softcap_and_temperature():
    torch.manual_seed(5)
    initial = torch.randn(2, 21, 43) * 9
    weights = torch.randn(2, 13)
    labels = torch.randint(43, (2, 13))
    gradients = []
    losses = []
    for recompute in (False, True):
        raw = initial.clone().requires_grad_()
        logits = (raw / 30).tanh() * 30
        logits.div_(0.5)
        logits = logits[:, -14:-1]
        log_probs = logits.log_softmax(-1).gather(-1, labels[..., None]).squeeze(-1)
        entropy = (entropy_from_logits_recomputed(logits.reshape(-1, 43), 7).reshape(2, 13)
                   if recompute else reference(logits))
        loss = -(weights * log_probs).mean() - 0.001 * entropy.mean()
        loss.backward()
        gradients.append(raw.grad)
        losses.append(loss)
    torch.testing.assert_close(losses[0], losses[1])
    torch.testing.assert_close(gradients[0], gradients[1], rtol=1e-5, atol=1e-6)


def test_backward_saves_only_original_logits_storage():
    logits = torch.randn(513, 67, dtype=torch.bfloat16, requires_grad=True)
    saved = []

    def pack(tensor):
        saved.append(tensor)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        result = entropy_from_logits_recomputed(logits)
    assert len(saved) == 1
    assert saved[0].data_ptr() == logits.data_ptr()
    assert saved[0].dtype == torch.bfloat16
    result.mean().backward()
    assert torch.isfinite(logits.grad).all()


def test_saved_logits_are_protected_from_inplace_changes():
    logits = torch.randn(5, 7, requires_grad=True)
    result = entropy_from_logits_recomputed(logits)
    with torch.no_grad():
        logits.add_(1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        result.sum().backward()
