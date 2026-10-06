"""Token entropy with bounded softmax workspace in both forward and backward."""

import torch
from torch.autograd.function import once_differentiable


class _RecomputedEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, chunk_size):
        ctx.save_for_backward(logits)
        ctx.chunk_size = chunk_size
        compute_dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
        entropy = torch.empty(logits.shape[0], dtype=compute_dtype, device=logits.device)
        for start in range(0, logits.shape[0], chunk_size):
            values = logits[start : start + chunk_size].to(compute_dtype)
            probabilities = values.softmax(dim=-1)
            entropy[start : start + chunk_size] = values.logsumexp(dim=-1) - (
                probabilities.mul_(values).sum(dim=-1)
            )
            del values, probabilities
        return entropy

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        (logits,) = ctx.saved_tensors
        grad_logits = torch.empty_like(logits)
        for start in range(0, logits.shape[0], ctx.chunk_size):
            values = logits[start : start + ctx.chunk_size].to(grad_output.dtype)
            probabilities = values.softmax(dim=-1)
            mean = (probabilities * values).sum(dim=-1, keepdim=True)
            probabilities.mul_(mean - values)
            probabilities.mul_(grad_output[start : start + ctx.chunk_size, None])
            grad_logits[start : start + ctx.chunk_size].copy_(probabilities)
            del values, probabilities, mean
        return grad_logits, None


def entropy_from_logits_recomputed(logits: torch.Tensor, chunk_size: int = 256):
    """Return float32 entropy for (tokens, vocab) logits with first-order gradients.

    Only the original logits are saved for backward; no float32 vocabulary-sized
    intermediates survive forward. The output gradient still has logits' shape.
    At Gemma's 262,144-token vocabulary, each float32 chunk is at most 256 MiB.
    """
    if logits.ndim != 2:
        raise ValueError("Expected logits with shape (tokens, vocab)")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    return _RecomputedEntropy.apply(logits, chunk_size)
