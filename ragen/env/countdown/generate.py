"""Deterministic multi-operation Countdown problems in the tropical paper's regime.

Each problem has n in {3, 4, 5, 6} inputs from [1, 20] and a target in [1, 100] reachable by
combining every input exactly once with + - * / through integer intermediates. A witness
operation sequence is stored so the preflight can verify both policy interfaces on it.
"""

import argparse
from fractions import Fraction
import json
from pathlib import Path
import random

import pandas as pd

INPUT_RANGE = (1, 20)
TARGET_RANGE = (1, 100)
SIZES = (3, 4, 5, 6)


def _random_witness(rng, n):
    values = [rng.randint(*INPUT_RANGE) for _ in range(n)]
    live = [Fraction(v) for v in values]
    steps = []
    while len(live) > 1:
        i, j = rng.sample(range(len(live)), 2)
        a, b = live[i], live[j]
        options = [("+", a + b), ("-", a - b), ("*", a * b)]
        if b != 0 and (a / b).denominator == 1:
            options.append(("/", a / b))
        op, result = rng.choice(options)
        if result.denominator != 1 or abs(result) > 10_000:
            return None
        steps.append(f"{a} {op} {b}")
        live = [v for k, v in enumerate(live) if k not in (i, j)] + [result]
    target = live[0]
    if not TARGET_RANGE[0] <= target <= TARGET_RANGE[1]:
        return None
    return {"target": int(target), "nums": values, "solution": steps}


def generate(seed, per_size, sizes=SIZES, exclude=()):
    rng = random.Random(seed)
    seen = {(tuple(sorted(row["nums"])), row["target"]) for row in exclude}
    rows = []
    for n in sizes:
        count = 0
        while count < per_size:
            row = _random_witness(rng, n)
            if row is None:
                continue
            key = (tuple(sorted(row["nums"])), row["target"])
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
            count += 1
    rng.shuffle(rows)  
    return rows


def write(rows, path):
    frame = pd.DataFrame({"target": [r["target"] for r in rows], "nums": [list(r["nums"]) for r in rows],
                          "solution": [json.dumps(r["solution"]) for r in rows]})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def build(directory, seed=2026, val_per_size=128, train_per_size=512):
    directory = Path(directory)
    val = generate(seed, val_per_size)
    train = generate(seed + 1, train_per_size, exclude=val)
    write(val, directory / "tropic_val.parquet")
    write(train, directory / "tropic_train.parquet")
    return train, val


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default="data/countdown")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    train, val = build(args.directory, args.seed)
    print(f"wrote {len(train)} training and {len(val)} validation problems to {args.directory}")


if __name__ == "__main__":
    main()
