from dataclasses import replace
import itertools
import json
import pytest

from ragen.tropic.graph import Edge, FragmentGraph, Node


def node(key, depth):
    return Node(key, depth, {}, key)


def add(graph, source, target, action, score, success=False):
    edge = Edge(source.key, target.key, action, (ord(source.key),), (action, 2), '', '', 0., success,
                [str(action)], [score], score, 1)
    return graph.add_edge(source, target, edge)


def diamond():
    root, a, b, join, goal = [node(k, d) for k, d in [('r', 0), ('a', 1), ('b', 1), ('j', 2), ('g', 3)]]
    graph = FragmentGraph('problem', root)
    ra = add(graph, root, a, 1, -3)
    rb = add(graph, root, b, 2, -1)
    aj = add(graph, a, join, 3, -2)
    bj = add(graph, b, join, 4, -1)
    jg = add(graph, join, goal, 4, -1, True)
    graph.certify((ra, aj, jg), 'root')
    return graph, (ra, aj, jg), (rb, bj, jg)


def test_max_plus_composes_best_prefix_with_certified_suffix():
    graph, sampled, composed = diamond()
    assert composed in graph.candidates()
    prefixes, suffixes = graph.paths()
    assert prefixes[graph.edges[composed[-1]].target][0] == composed
    assert graph.score(composed) == -3
    assert sampled not in graph.candidates()
    graph.certify(composed, 'composition')
    assert graph.select_basis(2)[0] == composed


def test_top_l_paths_match_exhaustive_enumeration():
    graph = FragmentGraph('problem', node('r', 0), top_l=3)
    layers = [[graph.nodes['r']], [node('a', 1), node('b', 1)], [node('c', 2), node('d', 2)], [node('g', 3)]]
    for depth, (sources, targets) in enumerate(zip(layers, layers[1:])):
        for i, (source, target) in enumerate(itertools.product(sources, targets)):
            add(graph, source, target, i + 1, -float((i + 1) * (depth + 1)), depth == 2)
    all_paths = []
    def walk(state, path):
        if state == 'g':
            all_paths.append(path)
        for key, edge in graph.edges.items():
            if edge.source == state:
                walk(edge.target, path + (key,))
    walk('r', ())
    expected = sorted(all_paths, key=lambda p: (-graph.score(p), p))[:3]
    assert graph.paths()[0]['g'] == expected


def test_graph_roundtrip_retains_provenance_and_selection_state():
    graph, _, composed = diamond()
    graph.certify(composed, 'composition')
    graph.select_basis(2)
    restored = FragmentGraph.from_dict(json.loads(json.dumps(graph.to_dict())))
    assert restored.to_dict() == graph.to_dict()
    assert restored.select_basis(2) == graph.select_basis(2)


def test_prompt_aliases_cycles_and_unverified_paths_rejected():
    graph, sampled, _ = diamond()
    edge = graph.edges[sampled[0]]
    with pytest.raises(ValueError, match='different policy prompts'):
        graph.add_edge(graph.nodes[edge.source], graph.nodes[edge.target], replace(edge, prompt_ids=(999,)))
    with pytest.raises(ValueError, match='advance'):
        graph.add_edge(graph.nodes[edge.source], graph.nodes[edge.source], replace(edge, target=edge.source))
    with pytest.raises(ValueError, match='successful'):
        graph.certify(sampled[:-1], 'root')


def test_duplicate_edges_are_idempotent_and_caps_do_not_break_old_paths():
    graph, sampled, _ = diamond()
    edge = graph.edges[sampled[0]]
    count = len(graph.edges)
    assert graph.add_edge(graph.nodes[edge.source], graph.nodes[edge.target], edge) == sampled[0]
    assert len(graph.edges) == count
    graph.max_edges = count
    assert graph.add_edge(graph.nodes[edge.source], graph.nodes[edge.target], replace(edge, completion_ids=(88,))) is None
    assert graph.solutions[sampled] == 'root'


def test_better_token_realization_does_not_count_as_a_new_solution():
    graph, sampled, _ = diamond()
    old = graph.edges[sampled[0]]
    new = replace(old, completion_ids=(99,), log_prob=-1.)
    key = graph.add_edge(graph.nodes[old.source], graph.nodes[old.target], new)
    replacement = (key,) + sampled[1:]
    assert not graph.certify(replacement, 'root')
    assert replacement in graph.solutions
    assert sampled not in graph.solutions
    assert len(graph.solutions) == 1


def test_failed_joins_are_cached_across_resume_without_removing_fragments():
    graph, _, composed = diamond()
    graph.reject_join(composed)
    restored = FragmentGraph.from_dict(json.loads(json.dumps(graph.to_dict())))
    assert composed not in restored.candidates()
    assert all(key in restored.edges for key in composed)
