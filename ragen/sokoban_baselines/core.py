"""Algorithmic pieces independent of Ray, vLLM and environment execution."""

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class Decision:
    prompt_ids: tuple
    completion_ids: tuple
    episode: int = 0
    group: int = 0
    depth: int = 0
    action_history: tuple = ()
    next_prompt_ids: Optional[tuple] = None
    reward: float = 0.0
    observation: str = ""


def maxrl_advantages(rewards, groups, eps=1e-8):
    """Practical MaxRL estimator, (r - mean(r)) / (mean(r) + eps).

    One entry per *trajectory*, including failures. Uniform-outcome groups have
    exactly zero gradient; neither dense environment rewards nor std scaling enter.
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    groups = np.asarray(groups)
    if rewards.shape != groups.shape or not np.isin(rewards, [0, 1]).all():
        raise ValueError("MaxRL needs aligned binary trajectory rewards and groups")
    result = np.zeros_like(rewards)
    for group in np.unique(groups):
        selected = groups == group
        mean = rewards[selected].mean()
        result[selected] = (rewards[selected] - mean) / (mean + eps)
    return result


def policy_loss(log_probs, mask, advantages, weights, old_log_probs=None, clip=.2):
    """Global weighted sum; caller compensates FSDP's gradient averaging."""
    mask = mask.bool()
    log_probs = log_probs.masked_fill(~mask, 0.0)
    if old_log_probs is None:  
        surrogate = log_probs * advantages[:, None]
    else:
        ratio = (log_probs - old_log_probs.masked_fill(~mask, 0.0)).exp()
        a = advantages[:, None]
        surrogate = torch.minimum(ratio * a, ratio.clamp(1 - clip, 1 + clip) * a)
    return -(surrogate.masked_fill(~mask, 0.0).sum(-1) * weights).sum()


def surgical_loss(chosen, rejected, ref_chosen, ref_rejected, beta):
    """Sequence-summed log probabilities of only the divergent step's thought."""
    return -F.logsigmoid(beta * ((chosen - rejected) - (ref_chosen - ref_rejected)))


def thought_prefix(tokenizer, completion, opening_in_prompt=False):
    """Keep exact generated tokens through </think>; never retokenize targets.

    A missing/unfinished thought supplies no preference pair. Tokens containing
    both the closing tag and a following action are excluded rather than training
    an action with the supposedly thought-only loss.
    """
    for end in range(1, len(completion) + 1):
        text = tokenizer.decode(completion[:end], skip_special_tokens=False)
        if '</think>' in text:
            if (opening_in_prompt or '<think>' in text) and text.rstrip().endswith('</think>'):
                return tuple(completion[:end])
            return ()
    return ()


@dataclass
class CognitiveTree:
    components: dict
    members: dict
    children: dict
    q: dict
    advantages: np.ndarray
    divergences: list


def build_tree(rows, kl, kl_threshold=.25, gamma=.99, divergence_threshold=.3):
    """Eq. 2--5/8 of T-STAR; depth-indexed components form an acyclic graph.

    A node is a completed decision/observation; its policy context is the next
    decision's prompt. Roots are shared within each problem. Absorbing leaves only
    merge with equal-outcome leaves, so a cutoff never inherits a live continuation.
    ``kl(left, right)`` returns a Monte Carlo estimate under the current policy.
    """
    buckets = defaultdict(list)
    episodes = defaultdict(list)
    for i, row in enumerate(rows):
        leaf = row.next_prompt_ids is None
        buckets[(row.group, row.depth, row.action_history, leaf,
                 row.reward if leaf else None)].append(i)
        episodes[row.episode].append(i)
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for indices in buckets.values():
        for offset, i in enumerate(indices):
            for j in indices[offset + 1:]:
                if find(i) == find(j):
                    continue
                value = 0.0 if rows[i].next_prompt_ids == rows[j].next_prompt_ids else kl(rows[i], rows[j])
                if not np.isfinite(value):
                    raise ValueError("Non-finite policy KL estimate")
                if value < kl_threshold:
                    parent[find(j)] = find(i)
    components = {i: find(i) for i in range(len(rows))}
    members = defaultdict(list)
    children = defaultdict(Counter)
    edge_rows = defaultdict(list)
    rewards = defaultdict(list)
    for i, node in components.items():
        members[node].append(i)
    for path in episodes.values():
        path.sort(key=lambda i: rows[i].depth)
        first = rows[path[0]]
        root = ('root', first.group)
        rewards[first.group].append(rows[path[-1]].reward)
        prev = root
        for i in path:
            node = components[i]
            children[prev][node] += 1
            edge_rows[prev, node].append(i)
            prev = node
    q = {}
    for node in sorted(members, key=lambda n: rows[members[n][0]].depth, reverse=True):
        edges = children[node]
        if edges:
            q[node] = gamma * sum(count * q[child] for child, count in edges.items()) / sum(edges.values())
        else:
            q[node] = float(np.mean([rows[i].reward for i in members[node]]))
    advantages = np.zeros(len(rows))
    for i, row in enumerate(rows):
        values = rewards[row.group]
        std = np.std(values, ddof=1) if len(values) > 1 else 0.0
        if std > 0:
            advantages[i] = (q[components[i]] - np.mean(values)) / (std + 1e-8)
    divergences = []
    for node, edges in children.items():
        if len(edges) < 2:
            continue
        best, worst = max(edges, key=q.get), min(edges, key=q.get)
        if q[best] - q[worst] > divergence_threshold:
            divergences.append((edge_rows[node, best][0], edge_rows[node, worst][0], q[best] - q[worst]))
    return CognitiveTree(components, dict(members), dict(children), q, advantages, divergences)
