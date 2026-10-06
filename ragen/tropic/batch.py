"""Exact-token batches and problem/path/token-normalized supervision."""

import math
import torch
from verl import DataProto


def basis_rows(graphs, bases):
    positive = [problem for problem, paths in bases.items() if paths]
    rows, weights = [], []
    for problem in positive:
        graph, paths = graphs[problem], bases[problem]
        for path in paths:
            tokens = sum(len(graph.edges[k].completion_ids) for k in path)
            if tokens == 0:
                raise ValueError("Verified paths must contain completion tokens")
            weight = 1.0 / (len(positive) * len(paths) * tokens)
            for key in path:
                rows.append(graph.edges[key])
                weights.append(weight)
    return rows, weights


def make_batch(rows, pad_token_id, weights=None, divisor=1):
    if not rows:
        raise ValueError("Cannot batch an empty verified path basis")
    if divisor < 1:
        raise ValueError("Batch divisor must be positive")
    weights = [1.0] * len(rows) if weights is None else list(weights)
    if len(weights) != len(rows):
        raise ValueError("One weight is required per row")
    n = math.ceil(len(rows) / divisor) * divisor
    width = max(len(r.prompt_ids) + len(r.completion_ids) for r in rows)
    ids = torch.full((n, width), pad_token_id, dtype=torch.long)
    attention = torch.zeros_like(ids)
    mask = torch.zeros((n, width - 1), dtype=torch.float32)
    row_weights = torch.zeros(n, dtype=torch.float32)
    for i in range(n):
        row = rows[i % len(rows)]
        sequence = tuple(row.prompt_ids) + tuple(row.completion_ids)
        offset = width - len(sequence)
        ids[i, offset:] = torch.tensor(sequence)
        attention[i, offset:] = 1
        mask[i, offset + len(row.prompt_ids) - 1:] = 1
        if i < len(rows):
            row_weights[i] = weights[i]
    positions = (attention.cumsum(-1) - 1).clamp(min=0)
    return DataProto.from_dict(tensors={
        "input_ids": ids, "responses": ids[:, 1:], "attention_mask": attention,
        "position_ids": positions, "response_mask": mask, "tropic_weights": row_weights,
    }, meta_info={"temperature": 1.0, "global_token_num": attention.sum(-1).tolist(),
                  "tropic_real_rows": len(rows)})


def verified_path_loss(log_probs, response_mask, weights):
    return -(log_probs.masked_fill(~response_mask.bool(), 0.0).sum(-1) * weights).sum()
