"""Exact-token batches with zero-weight padding and rank-local preference pairs."""

import math
from collections import Counter

import torch
from verl import DataProto


def token_batch(rows, pad_token_id):
    if not rows or any(not r.prompt_ids or not r.completion_ids for r in rows):
        raise ValueError("Nonempty prompts and completions are required")
    width = max(len(r.prompt_ids) + len(r.completion_ids) for r in rows)
    ids = torch.full((len(rows), width), pad_token_id, dtype=torch.long)
    attention = torch.zeros_like(ids)
    mask = torch.zeros((len(rows), width - 1), dtype=torch.float32)
    for i, row in enumerate(rows):
        seq = row.prompt_ids + row.completion_ids
        start = width - len(seq)
        ids[i, start:] = torch.tensor(seq)
        attention[i, start:] = 1
        mask[i, start + len(row.prompt_ids) - 1:] = 1
    return DataProto.from_dict(tensors={
        'input_ids': ids, 'responses': ids[:, 1:], 'attention_mask': attention,
        'position_ids': (attention.cumsum(-1) - 1).clamp(min=0), 'response_mask': mask,
    }, meta_info={'temperature': 1.0, 'global_token_num': attention.sum(-1).tolist()})


def training_batch(rows, advantages, pairs, pad_token_id, world, micro, method):
    """Each dispatch shard contains RL rows followed by complete (chosen,rejected) pairs.

    All ranks execute equally many forwards/backwards, including zero-weight slots.
    Padding never changes either the global RL or preference denominator.
    """
    if not rows or len(rows) != len(advantages) or world < 1 or micro < 1:
        raise ValueError("Invalid baseline training batch")
    n_rl = math.ceil(len(rows) / (world * micro)) * micro
    n_pairs = math.ceil(len(pairs) / world)
    total_tokens = sum(len(r.completion_ids) for r in rows)
    episode_tokens = Counter()
    for row in rows:
        episode_tokens[row.episode] += len(row.completion_ids)
    packed, adv, weights, pair_weights = [], [], [], []
    for rank in range(world):
        for local in range(n_rl):
            idx = rank * n_rl + local
            real = idx < len(rows)
            row = rows[idx] if real else rows[0]
            weight = (1 / total_tokens if method == 'maxrl' else
                      1 / (len(episode_tokens) * episode_tokens[row.episode]))
            packed.append(row)
            adv.append(advantages[idx] if real else 0.0)
            weights.append(weight if real else 0.0)
            pair_weights.append(0.0)
        for local in range(n_pairs):
            idx = rank * n_pairs + local
            real = idx < len(pairs)
            pair = pairs[idx] if real else pairs[0]
            for row in pair:
                packed.append(row)
                adv.append(0.0)
                weights.append(0.0)
                pair_weights.append(1 / len(pairs) if real else 0.0)
    batch = token_batch(packed, pad_token_id)
    batch.batch['baseline_advantages'] = torch.tensor(adv, dtype=torch.float32)
    batch.batch['baseline_weights'] = torch.tensor(weights, dtype=torch.float32)
    batch.batch['pair_weights'] = torch.tensor(pair_weights, dtype=torch.float32)
    batch.meta_info.update(baseline_method=method, baseline_dp_size=world,
                           baseline_rl_rows_per_rank=n_rl, baseline_pairs_per_rank=n_pairs)
    return batch
