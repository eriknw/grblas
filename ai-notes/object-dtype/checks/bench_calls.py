# ruff: noqa: INP001, T201, B905, PLW0108
"""Per-call cost of Python ops on OBJECT values, against pure Python and numpy baselines.

For int, float, Fraction and tuple values, times apply, ewise_mult (an add), reduce_rowwise
(rows of 10 entries, a Python + monoid) and mxm on a Python plus_times semiring (M by M
with about 10 entries per row, not for tuples), plus a heavier apply on ints. Each line is
the best of --reps runs divided by the number of Python-op calls (for mxm, by the number
of multiplies). With --baselines it times the same work without GraphBLAS calling Python:
list comprehensions, np.frompyfunc, numpy object arrays, dict loops, and "eager" (to_coo,
a Python loop, from_coo). Nothing here checks results; the soundness checks are fuzz.py,
thread_stress.py and safety_probe.py.

Run: PYTHONPATH=<worktree> python bench_calls.py [--baselines] [--jit off|pause|run|load|on]
     [--nthreads N] [--n N] [--mxm-m M] [--big-n N] [--reps R]
Output: one JSON line per (type, op) with ns_per_call, then a last line {"done": true, ...}
that names the trampolines used ("paths"). Set GRAPHBLAS_CSHIM=0 for the Numba
trampolines, and also GRAPHBLAS_NUMBA_CAPI=0 for cffi callbacks. Run it alone on the
machine: other load skews the times.
"""

import argparse
import json
import sys
import time
from fractions import Fraction

import numpy as np

import graphblas as gb
from graphblas import Matrix, Vector, binary, monoid, semiring, unary

try:
    from graphblas.core import objects
    from graphblas.core.operator import pyobject
except ImportError:
    sys.exit(f"{gb.__file__} has no OBJECT dtype: put objdtype-core on PYTHONPATH")

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument(
    "--baselines", action="store_true", help="time the baselines instead of the GraphBLAS ops"
)
parser.add_argument(
    "--jit",
    default="on",
    choices=["off", "pause", "run", "load", "on"],
    help="jit_c_control (default on)",
)
parser.add_argument(
    "--nthreads", type=int, help="pass .new(nthreads=N) (default: python-graphblas decides)"
)
parser.add_argument("--n", type=int, default=50_000, help="values per op (default 50000)")
parser.add_argument(
    "--mxm-m", type=int, default=1000, help="mxm on M by M with 10 M entries (default 1000)"
)
parser.add_argument(
    "--big-n", type=int, default=0, help="also time apply on ints at this size (default: skip)"
)
parser.add_argument("--reps", type=int, default=5, help="best of this many runs (default 5)")
args = parser.parse_args()
print(f"graphblas {gb.__file__}; Python {sys.version.split()[0]}", file=sys.stderr, flush=True)

O = gb.dtypes.OBJECT  # noqa: E741
MODE = "baselines" if args.baselines else "ops"
NT = args.nthreads
JIT = args.jit
gb.ss.config["jit_c_control"] = JIT
if NT is not None and NT > 1:
    gb.ss.config["chunk"] = 4096  # small chunks, so that --n values split across threads
objects.threshold = 1 << 62  # no automatic collections inside the timings
REPS = args.reps
GIL = getattr(sys, "_is_gil_enabled", lambda: True)()
rng = np.random.default_rng(0)
kw = {} if NT is None else {"nthreads": NT}


def python_op_paths():
    """The trampolines that run the Python ops made so far: cshim (C), numba, or cffi."""
    c_name = {"_CHeap": "cshim", "_NumbaHeap": "numba"}.get(type(objects._heap.c).__name__, "c")
    kinds = {c_name if kind == "c" else kind for kind, _ in pyobject._callbacks.values()}
    return ",".join(sorted(kinds))


def best(f):
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t0)
    return min(ts)


def emit(vtype, op, ns, **extra):
    print(
        json.dumps(
            {
                "mode": MODE,
                "build": "GIL" if GIL else "FT",
                "nthreads": NT,
                "jit": JIT,
                "type": vtype,
                "op": op,
                "ns_per_call": round(ns, 1),
                **extra,
            }
        ),
        flush=True,
    )


def objarr(values):
    a = np.empty(len(values), dtype=object)
    a[:] = values
    return a


MAKERS = {
    "int": (lambda a: int(a), 0),
    "float": (lambda a: float(a) / 7, 0.0),
    "Fraction": (lambda a: Fraction(int(a), 7), Fraction(0)),
    "tuple": (lambda a: (int(a), int(a) + 1), ()),
}


def twice(x):
    return x * 2


def add(x, y):
    return x + y


def mul(x, y):
    return x * y


def heavy(x):
    s = 0
    for i in range(120):
        s += i * i
    return x + s


py_twice = unary.register_anonymous(twice, is_udt=True)
py_add = binary.register_anonymous(add, is_udt=True)
py_mul = binary.register_anonymous(mul, is_udt=True)
py_heavy = unary.register_anonymous(heavy, is_udt=True)


def run_type(vtype, n, m, apply_only=False):
    make, ident = MAKERS[vtype]
    xs = objarr([make(a) for a in rng.integers(1, 100, n)])
    ys = objarr([make(a) for a in rng.integers(1, 100, n)])
    idx = np.arange(n)
    plus = monoid.register_anonymous(py_add, ident)
    # reduce_rowwise input: rows of 10 entries
    nrows = n // 10
    R = np.repeat(np.arange(nrows), 10)
    Cc = (np.tile(np.arange(10), nrows) + R) % 1000
    if MODE != "baselines":
        v = Vector.from_coo(idx, xs, dtype=O, size=n)
        w = Vector.from_coo(idx, ys, dtype=O, size=n)
        M = Matrix.from_coo(R, Cc, xs, dtype=O, nrows=nrows, ncols=1000)
        emit(vtype, "apply", best(lambda: v.apply(py_twice).new(**kw)) / n * 1e9, n=n)
        if apply_only:
            return
        emit(vtype, "ewise_add", best(lambda: v.ewise_mult(w, py_add).new(**kw)) / n * 1e9, n=n)
        emit(
            vtype,
            "reduce_rowwise",
            best(lambda: M.reduce_rowwise(plus).new(**kw)) / n * 1e9,
            n=n,
        )
        if vtype != "tuple":
            nnz = 10 * m
            A = Matrix.from_coo(
                rng.integers(m, size=nnz),
                rng.integers(m, size=nnz),
                [make(a) for a in rng.integers(1, 100, nnz)],
                dtype=O,
                nrows=m,
                ncols=m,
                dup_op=binary.first,
            )
            S = A.apply(unary.one).new()
            products = int(S.mxm(S, semiring.plus_pair).new().reduce_scalar().value)
            pt = semiring.register_anonymous(plus, py_mul)
            emit(
                vtype,
                "mxm_plus_times",
                best(lambda: A.mxm(A, pt).new(**kw)) / products * 1e9,
                products=products,
            )
        if vtype == "int":
            emit(vtype, "apply_heavy", best(lambda: v.apply(py_heavy).new(**kw)) / n * 1e9, n=n)
        return
    # Baselines (no GraphBLAS op calls Python)
    emit(vtype, "apply:listcomp", best(lambda: [twice(x) for x in xs]) / n * 1e9, n=n)
    emit(vtype, "apply:frompyfunc", best(lambda: np.frompyfunc(twice, 1, 1)(xs)) / n * 1e9, n=n)
    emit(vtype, "apply:numpy_object_op", best(lambda: xs * 2) / n * 1e9, n=n)
    v = Vector.from_coo(idx, xs, dtype=O, size=n)
    w = Vector.from_coo(idx, ys, dtype=O, size=n)

    def eager_apply():
        i, vals = v.to_coo()
        return Vector.from_coo(i, objarr([twice(x) for x in vals]), dtype=O, size=n)

    emit(vtype, "apply:eager", best(eager_apply) / n * 1e9, n=n)
    if apply_only:
        return
    emit(vtype, "ewise_add:listcomp", best(lambda: [add(a, b) for a, b in zip(xs, ys)]) / n * 1e9)
    emit(vtype, "ewise_add:frompyfunc", best(lambda: np.frompyfunc(add, 2, 1)(xs, ys)) / n * 1e9)
    emit(vtype, "ewise_add:numpy_object_op", best(lambda: xs + ys) / n * 1e9)

    def eager_ewise():
        i, a = v.to_coo()
        _, b = w.to_coo()
        return Vector.from_coo(i, objarr([add(p, q) for p, q in zip(a, b)]), dtype=O, size=n)

    emit(vtype, "ewise_add:eager", best(eager_ewise) / n * 1e9)
    rows_l = R.tolist()

    def dict_loop():
        acc = {}
        for r, x in zip(rows_l, xs):
            acc[r] = add(acc[r], x) if r in acc else x
        return acc

    emit(vtype, "reduce_rowwise:dict_loop", best(dict_loop) / n * 1e9)
    starts = np.arange(0, n, 10)
    emit(
        vtype,
        "reduce_rowwise:frompyfunc_reduceat",
        best(lambda: np.frompyfunc(add, 2, 1).reduceat(xs, starts)) / n * 1e9,
    )
    emit(
        vtype,
        "reduce_rowwise:numpy_add_reduceat",
        best(lambda: np.add.reduceat(xs, starts)) / n * 1e9,
    )
    M = Matrix.from_coo(R, Cc, xs, dtype=O, nrows=nrows, ncols=1000)

    def eager_reduce():
        r, _, vals = M.to_coo()
        acc = {}
        for i, x in zip(r.tolist(), vals):
            acc[i] = add(acc[i], x) if i in acc else x
        keys = list(acc)
        return Vector.from_coo(keys, objarr([acc[k] for k in keys]), dtype=O, size=nrows)

    emit(vtype, "reduce_rowwise:eager", best(eager_reduce) / n * 1e9)
    if vtype != "tuple":
        nnz = 10 * m
        ar, ac = rng.integers(m, size=nnz), rng.integers(m, size=nnz)
        av = [make(a) for a in rng.integers(1, 100, nnz)]
        A = {}
        for i, j, x in zip(ar.tolist(), ac.tolist(), av):
            A.setdefault(i, {}).setdefault(j, x)
        products = sum(len(A.get(k, ())) for row in A.values() for k in row)

        def dict_mxm():
            out = {}
            for i, row in A.items():
                acc = {}
                for k, a in row.items():
                    for j, b in A.get(k, {}).items():
                        p = mul(a, b)
                        acc[j] = add(acc[j], p) if j in acc else p
                out[i] = acc
            return out

        emit(
            vtype,
            "mxm_plus_times:dict_of_dicts",
            best(dict_mxm) / products * 1e9,
            products=products,
        )
    if vtype == "int":
        emit(vtype, "apply_heavy:listcomp", best(lambda: [heavy(x) for x in xs]) / n * 1e9)


t_start = time.perf_counter()
for vtype in MAKERS:
    run_type(vtype, args.n, args.mxm_m)
if args.big_n:
    run_type("int", args.big_n, args.mxm_m, apply_only=True)
done = {"done": True, "mode": MODE, "seconds": round(time.perf_counter() - t_start, 1)}
if MODE != "baselines":
    done["paths"] = python_op_paths()
print(json.dumps(done))
