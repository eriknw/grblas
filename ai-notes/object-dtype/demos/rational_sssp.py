# ruff: noqa: INP001, T201
"""Exact shortest paths with Fraction weights (Bellman-Ford over min_plus).

Rational weights would be rounded in FP64. With Fractions as values, repeated
``vxm`` on a Python min_plus semiring gives exact distances. ``min`` returns one of its
inputs, so an unchanged distance is the same object, and a Python ``is not`` finds the
changed nodes (the next frontier).
Reference: networkx Bellman-Ford and Dijkstra, compared exactly. Without networkx the
demo still runs, and its row is unchecked.

Run: PYTHONPATH=<worktree> python rational_sssp.py [--large] [--reps N] [--csv PATH]
Success: each printed row has 'correct': True, and the last line starts with "OK:".
"""

import math
import operator
from fractions import Fraction

import numpy as np
from _common import best_of, loc, random_digraph, run

import graphblas as gb

try:
    import networkx as nx
except ModuleNotFoundError:
    nx = None

SIZES = ((2000, 10000), (20000, 100000))


def sssp_graphblas(rows, cols, weights, n, source):
    """Bellman-Ford distances from ``source`` as {node: Fraction}."""
    A = gb.Matrix.from_coo(rows, cols, weights, dtype=object, nrows=n, ncols=n)
    mn = gb.monoid.register_anonymous(gb.binary.register_anonymous(min, is_udt=True), math.inf)
    add = gb.binary.register_anonymous(operator.add, is_udt=True)
    semi = gb.semiring.register_anonymous(mn, add)
    is_not = gb.binary.register_anonymous(operator.is_not, is_udt=True)
    truthy = gb.select.register_anonymous(lambda x, i, j, t: x, is_udt=True)
    d = gb.Vector.from_coo([source], [Fraction(0)], dtype=object, size=n)
    frontier = d
    for _ in range(n):
        d2 = d.ewise_add(frontier.vxm(A, semi).new(), mn).new()
        # min returns one of its inputs, so an unchanged distance is the same object
        changed = d2.ewise_mult(d, is_not).new().select(truthy).new()
        frontier = d2.dup(mask=~d.S)
        frontier(changed.S) << d2
        d = d2
        if frontier.nvals == 0:
            break
    idx, vals = d.to_coo()
    return dict(zip(idx.tolist(), vals, strict=True))


def make_graph(rows, cols, weights):
    G = nx.DiGraph()
    G.add_weighted_edges_from(zip(rows.tolist(), cols.tolist(), weights, strict=True))
    return G


def sssp_networkx_bf(G, source):
    """Networkx Bellman-Ford (works on Fractions as is)."""
    return nx.single_source_bellman_ford_path_length(G, source)


def sssp_networkx_dijkstra(G, source):
    return nx.single_source_dijkstra_path_length(G, source)


def main(sizes=SIZES, reps=3):
    if nx is None:
        print("networkx is not installed: no networkx baseline, and the rows are unchecked")
    out = []
    for n, m in sizes:
        r, c = random_digraph(n, m, seed=2)
        rng = np.random.default_rng(3)
        w = [
            Fraction(int(a), int(b))
            for a, b in zip(rng.integers(1, 100, len(r)), rng.integers(1, 100, len(r)), strict=True)
        ]
        t_gb, got = best_of(sssp_graphblas, r, c, w, n, 0, reps=reps)
        if nx is None:
            t_bf = ok = None
            nx_note = "networkx not installed"
        else:
            G = make_graph(r, c, w)
            t_bf, ref = best_of(sssp_networkx_bf, G, 0, reps=reps)
            t_dj, ref2 = best_of(sssp_networkx_dijkstra, G, 0, reps=reps)
            t_build, _ = best_of(make_graph, r, c, w, reps=reps)
            ok = got == dict(ref) == dict(ref2)
            nx_note = f"networkx Dijkstra {t_dj:.3f}s; nx graph build {t_build:.3f}s not counted"
        f64 = gb.Matrix.from_coo(r, c, [float(x) for x in w], nrows=n, ncols=n)
        d = gb.Vector.from_coo([0], [0.0], size=n)
        for _ in range(n):
            d2 = d.ewise_add(d.vxm(f64, gb.semiring.min_plus).new(), gb.binary.min).new()
            if d2.isequal(d):
                break
            d = d2
        fi, fv = d.to_coo()
        f_wrong = sum(Fraction(x) != got[i] for i, x in zip(fi.tolist(), fv.tolist(), strict=True))
        out.append(
            {
                "demo": "rational SSSP (Fraction)",
                "size": f"n={n} m={m}",
                "graphblas_s": t_gb,
                "baseline": "networkx Bellman-Ford [Dijkstra]",
                "baseline_s": t_bf,
                "correct": ok,
                "note": (
                    f"{nx_note}; reached {len(got)}; FP64 distances differ from exact in {f_wrong}"
                ),
                "user_loc": loc(sssp_graphblas),
                "baseline_loc": loc(sssp_networkx_bf),
            }
        )
        print(out[-1], flush=True)
    return out


if __name__ == "__main__":
    run(main, SIZES, __doc__)
