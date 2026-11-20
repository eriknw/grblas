# ruff: noqa: INP001, T201
"""Why-provenance of reachability: which sets of edges witness that v is reachable.

Values are frozensets of witnesses, each witness a frozenset of edge ids. The semiring
is (union, pairwise union): a path's witness is the union of its edges, and the
witnesses of v are the union over all paths. A layered DAG keeps the path count
finite. Reference: the same frontier iteration in pure Python with dicts, compared
exactly.

Run: PYTHONPATH=<worktree> python provenance.py [--large] [--reps N] [--csv PATH]
Success: each printed row has 'correct': True, and the last line starts with "OK:".
"""

import numpy as np
from _common import best_of, loc, run

import graphblas as gb

SIZES = ((10, 8, 2), (14, 10, 2))
EMPTY = frozenset()


def union(a, b):
    return a | b


def times(a, b):
    return frozenset(x | y for x in a for y in b)


def provenance_graphblas(rows, cols, n, source):
    """{node: frozenset of witness edge sets} for nodes reachable from ``source``."""
    w = [frozenset({frozenset({e})}) for e in range(len(rows))]
    A = gb.Matrix.from_coo(rows, cols, w, dtype=object, nrows=n, ncols=n)
    unite = gb.binary.register_anonymous(union, is_udt=True)
    semi = gb.semiring.register_anonymous(
        gb.monoid.register_anonymous(unite, EMPTY),
        gb.binary.register_anonymous(times, is_udt=True),
    )
    frontier = gb.Vector.from_coo([source], [frozenset({EMPTY})], dtype=object, size=n)
    result = frontier
    while frontier.nvals:
        frontier = frontier.vxm(A, semi).new()
        result = result.ewise_add(frontier, unite).new()
    idx, vals = result.to_coo()
    return dict(zip(idx.tolist(), vals, strict=True))


def provenance_python(rows, cols, n, source):
    """The same iteration with dicts of out-edges."""
    out = {}
    for e, (u, v) in enumerate(zip(rows.tolist(), cols.tolist(), strict=True)):
        out.setdefault(u, []).append((v, frozenset({frozenset({e})})))
    frontier = {source: frozenset({EMPTY})}
    result = dict(frontier)
    while frontier:
        nxt = {}
        for u, pu in frontier.items():
            for v, we in out.get(u, ()):
                t = times(pu, we)
                nxt[v] = nxt[v] | t if v in nxt else t
        for v, t in nxt.items():
            result[v] = result[v] | t if v in result else t
        frontier = nxt
    return result


def layered_dag(layers, width, deg, seed):
    rng = np.random.default_rng(seed)
    rows, cols = [], []
    for layer in range(layers - 1):
        for i in range(width):
            u = layer * width + i
            for j in rng.choice(width, size=deg, replace=False):
                rows.append(u)
                cols.append((layer + 1) * width + int(j))
    # a source feeding the first layer
    n = layers * width + 1
    src = n - 1
    rows = [src] * width + rows
    cols = list(range(width)) + cols
    order = np.lexsort((cols, rows))
    return np.array(rows)[order], np.array(cols)[order], n, src


def main(sizes=SIZES, reps=3):
    out = []
    for layers, width, deg in sizes:
        r, c, n, src = layered_dag(layers, width, deg, seed=6)
        t_gb, got = best_of(provenance_graphblas, r, c, n, src, reps=reps)
        t_py, ref = best_of(provenance_python, r, c, n, src, reps=reps)
        ok = got == ref
        nwit = sum(len(v) for v in got.values())
        out.append(
            {
                "demo": "why-provenance (frozensets)",
                "size": f"layers={layers} width={width} deg={deg} edges={len(r)}",
                "graphblas_s": t_gb,
                "baseline": "pure Python frontier with dicts",
                "baseline_s": t_py,
                "correct": ok,
                "note": f"{len(got)} nodes, {nwit} witness sets in total",
                "user_loc": loc(provenance_graphblas) + loc(union) + loc(times),
                "baseline_loc": loc(provenance_python) + loc(times),
            }
        )
        print(out[-1], flush=True)
    return out


if __name__ == "__main__":
    run(main, SIZES, __doc__)
