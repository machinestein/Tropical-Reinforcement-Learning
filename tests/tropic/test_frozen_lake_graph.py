"""Archive capacity must not freeze the early, inefficient FrozenLake solutions."""

from dataclasses import replace
import json

import pytest

from ragen.tropic.frozen_lake_graph import FrozenLakeFragmentGraph
from ragen.tropic.graph import Edge, FragmentGraph, Node


def node(position, depth):
    key = f"{position}:{depth}"
    return Node(key, depth, {"position": position}, key)


def graph(capacity=2, cls=FrozenLakeFragmentGraph):
    return cls("lake", node(0, 0), max_solutions=capacity)


def add_path(g, actions, total_score=-10):
    """Real deterministic moves on an empty 4x4 board, from (0,0) to (0,1)."""
    position, path = 0, []
    for depth, action in enumerate(actions):
        source = node(position, depth)
        row, col = divmod(position, 4)
        dr, dc = {1: (0, -1), 2: (1, 0), 3: (0, 1), 4: (-1, 0)}[action]
        position = 4 * min(3, max(0, row + dr)) + min(3, max(0, col + dc))
        target = node(position, depth + 1)
        success = position == 1
        assert success == (depth == len(actions) - 1)
        edge = Edge(source.key, target.key, action, tuple(source.key.encode()),
                    tuple(actions) + (depth,), "", "", float(success), success,
                    log_prob=total_score / len(actions))
        path.append(g.add_edge(source, target, edge))
    return tuple(path)


def filled(cls=FrozenLakeFragmentGraph):
    g = graph(cls=cls)
    short = add_path(g, [1, 3], -20)
    longer = add_path(g, [2, 3, 4], -6)
    assert g.certify(short, "root")
    assert g.certify(longer, "composition")
    return g, short, longer


def test_shorter_verified_route_enters_full_archive_even_at_lower_likelihood():
    g, short, longer = filled()
    direct = add_path(g, [3], -100)
    assert g.certify(direct, "root")
    assert set(g.solutions) == {direct, longer}
    assert g.solutions[direct] == "root"
    assert short not in g.solutions
    assert g.max_solutions == len(g.solutions) == 2
    assert g.archive_replacements == 1
    assert all(k in g.edges for k in short)  


def test_better_likelihood_enters_full_archive_without_evicting_shortest():
    g, short, longer = filled()
    better = add_path(g, [2, 4, 3], -3)
    assert g.certify(better, "composition")
    assert set(g.solutions) == {short, better}
    assert longer not in g.solutions


def test_worse_candidate_does_not_displace_retained_paths_or_count_as_new():
    g, short, longer = filled()
    worse = add_path(g, [2, 2, 4, 4, 3], -100)
    before = g.to_dict()
    assert not g.certify(worse, "root")
    assert g.to_dict() == before
    assert set(g.solutions) == {short, longer}


def test_single_slot_retains_shortest_verified_path():
    g = graph(capacity=1)
    longer = add_path(g, [2, 3, 4], -1)
    direct = add_path(g, [3], -100)
    assert g.certify(longer, "root") and g.certify(direct, "root")
    assert set(g.solutions) == {direct}
    assert not g.certify(longer, "root")


def test_equal_length_prefers_likelihood_and_ties_are_deterministic():
    g = graph(capacity=1)
    a, b = add_path(g, [1, 3], -5), add_path(g, [4, 3], -2)
    g.certify(a, "root")
    g.certify(b, "root")
    assert set(g.solutions) == {b}
    for path in (a, b):
        for k in path:
            g.edges[k].log_prob = -1
    g.certify(a, "root")
    assert set(g.solutions) == {min(a, b)}


def test_readmission_after_rescoring_is_not_a_new_solution_even_after_resume():
    g, short, original = filled()
    new = add_path(g, [2, 4, 3], -3)
    assert g.certify(new, "composition")
    restored = FrozenLakeFragmentGraph.from_dict(json.loads(json.dumps(g.to_dict())))
    assert restored.to_dict() == g.to_dict()
    for k in original:
        restored.edges[k].log_prob = -.1
    assert not restored.certify(original, "root")
    assert set(restored.solutions) == {short, original}
    assert restored.archive_replacements == 2


def test_better_token_realization_preserves_action_deduplication_and_novelty():
    g, short, original = filled()
    old = g.edges[original[0]]
    new = replace(old, completion_ids=(999,), log_prob=0.)
    key = g.add_edge(g.nodes[old.source], g.nodes[old.target], new)
    replacement = (key,) + original[1:]
    assert not g.certify(replacement, "root")
    assert set(g.solutions) == {short, replacement}
    assert g.archive_replacements == 0
    assert not g.certify(original, "root")


@pytest.mark.parametrize("kind", ["incomplete", "disconnected", "empty"])
def test_invalid_path_cannot_evict_a_verified_solution(kind):
    g, short, longer = filled()
    before = g.to_dict()
    bad = {"incomplete": longer[:-1], "disconnected": longer[1:], "empty": ()}[kind]
    with pytest.raises(ValueError):
        g.certify(bad, "root")
    assert g.to_dict() == before


def test_old_archive_is_not_silently_resumed_under_new_policy():
    old, _, _ = filled(cls=FragmentGraph)
    with pytest.raises(ValueError, match="restart from the base model"):
        FrozenLakeFragmentGraph.from_dict(old.to_dict())


def test_other_environments_keep_the_original_graph_and_retention_policy():
    from ragen.tropic.collector import TropicCollector
    assert TropicCollector.graph_class is FragmentGraph
    g, short, longer = filled(cls=FragmentGraph)
    direct = add_path(g, [3], -1)
    assert not g.certify(direct, "root")
    assert set(g.solutions) == {short, longer}
