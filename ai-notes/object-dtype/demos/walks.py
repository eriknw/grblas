# ruff: noqa: INP001, T201
"""Exact walk counts with big Python ints.

The number of walks of length k from each node of a graph is ``A^k 1``. It grows like
``degree^k`` and passes INT64 (and FP64's 53-bit integers) after a few dozen steps.
With Python ints as values, ``mxv`` on the plus_second semiring stays exact.
Reference: the same iteration with pure Python adjacency lists, compared exactly.

Run: PYTHONPATH=<worktree> python walks.py [--large] [--reps N] [--csv PATH]
Success: each printed row has 'correct': True, and the last line starts with "OK:".
"""

import operator

import numpy as np
from _common import best_of, loc, random_digraph, run

import graphblas as gb

SIZES = ((2000, 10000, 40), (20000, 100000, 40))


def walks_graphblas(rows, cols, n, k):
    """Walk counts of length k from every node, as {node: int}."""
    A = gb.Matrix.from_coo(rows, cols, [1] * len(rows), dtype=object, nrows=n, ncols=n)
    # plus_second: Python + as the monoid, the compiled structural `second` as multiply
    plus = gb.monoid.register_anonymous(gb.binary.register_anonymous(operator.add, is_udt=True), 0)
    semi = gb.semiring.register_anonymous(plus, gb.binary.second)
    v = gb.Vector.from_coo(np.arange(n), [1] * n, dtype=object, size=n)
    for _ in range(k):
        v = A.mxv(v, semi).new()
    idx, vals = v.to_coo()
    return dict(zip(idx.tolist(), vals, strict=True))


def walks_python(rows, cols, n, k):
    """The same with a dict of adjacency lists (strongest pure Python)."""
    adj = [[] for _ in range(n)]
    for i, j in zip(rows.tolist(), cols.tolist(), strict=True):
        adj[i].append(j)
    v = [1] * n
    for _ in range(k):
        v = [sum(v[j] for j in nbrs) for nbrs in adj]
    return {i: x for i, x in enumerate(v) if x}  # GraphBLAS stores no zeros here


def walks_native(rows, cols, n, k, dtype):
    A = gb.Matrix.from_coo(rows, cols, 1, dtype=dtype, nrows=n, ncols=n)
    v = gb.Vector.from_coo(np.arange(n), 1, dtype=dtype, size=n)
    for _ in range(k):
        v = A.mxv(v, gb.semiring.plus_second).new()
    idx, vals = v.to_coo()
    return dict(zip(idx.tolist(), vals.tolist(), strict=True))


def main(sizes=SIZES, reps=3):
    rows_out = []
    for n, m, k in sizes:
        r, c = random_digraph(n, m, seed=1)
        t_gb, got = best_of(walks_graphblas, r, c, n, k, reps=reps)
        t_py, ref = best_of(walks_python, r, c, n, k, reps=reps)
        exact = got == ref
        t_f, f64 = best_of(walks_native, r, c, n, k, "FP64", reps=reps)
        t_i, i64 = best_of(walks_native, r, c, n, k, "INT64", reps=reps)
        big = max(ref.values())
        f_wrong = sum(int(f64.get(i, 0)) != x for i, x in ref.items())
        i_wrong = sum(i64.get(i, 0) != x for i, x in ref.items())
        rows_out.append(
            {
                "demo": "walk counts (big ints)",
                "size": f"n={n} m={m} k={k}",
                "graphblas_s": t_gb,
                "baseline": "pure Python adjacency lists",
                "baseline_s": t_py,
                "correct": exact,
                "note": (
                    f"max count ~1e{len(str(big)) - 1}; "
                    f"FP64 {t_f:.3f}s wrong in {f_wrong}/{len(ref)}; "
                    f"INT64 {t_i:.3f}s wrong in {i_wrong}/{len(ref)}"
                ),
                "user_loc": loc(walks_graphblas),
                "baseline_loc": loc(walks_python),
            }
        )
        print(rows_out[-1], flush=True)
    return rows_out


if __name__ == "__main__":
    run(main, SIZES, __doc__)
