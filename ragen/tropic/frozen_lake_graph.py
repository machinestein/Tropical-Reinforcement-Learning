"""Bounded FrozenLake archives that admit improvements after reaching capacity."""

from .graph import FragmentGraph


class FrozenLakeFragmentGraph(FragmentGraph):
    archive_policy = "shortest_and_likelihood_v1"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.archive_replacements = 0
        self.seen_solutions = set()

    def certify(self, path, origin):
        path = tuple(path)
        capacity = self.max_solutions
        try:
            self.max_solutions = capacity + 1
            super().certify(path, origin)
        finally:
            self.max_solutions = capacity
        if len(self.solutions) > capacity:
            shortest = min(self.solutions, key=lambda p: (len(p), -self.score(p), p))
            ranked = sorted((p for p in self.solutions if p != shortest),
                            key=lambda p: (-self.score(p), p))
            keep = {shortest, *ranked[:capacity - 1]}
            self.solutions = {p: source for p, source in self.solutions.items() if p in keep}
            self.archive_replacements += int(path in keep)
        if path not in self.solutions:
            return False
        signature = tuple(self.edges[k].action for k in path)
        novel = signature not in self.seen_solutions
        self.seen_solutions.add(signature)
        return novel

    def to_dict(self):
        return {**super().to_dict(), "archive_policy": self.archive_policy,
                "archive_replacements": self.archive_replacements,
                "seen_solutions": [list(p) for p in sorted(self.seen_solutions)]}

    @classmethod
    def from_dict(cls, data):
        if data.get("archive_policy") != cls.archive_policy:
            raise ValueError("Incompatible FrozenLake archive policy; restart from the base model")
        graph = super().from_dict(data)
        graph.archive_replacements = data["archive_replacements"]
        graph.seen_solutions = {tuple(p) for p in data["seen_solutions"]}
        return graph
