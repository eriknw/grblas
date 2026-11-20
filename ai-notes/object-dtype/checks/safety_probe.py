# ruff: noqa: INP001, T201, S101, C408, PTH123
"""Safety probe: SuiteSparse calls Python ops only on live, stored objects of the right role.

Python ops are registered with is_udt=True on OBJECT. Every value is a V (an int
subclass with a tag: A, B, C, ident, extra, fresh) kept alive in LIVE, so each argument a
callback receives can be checked: it must be the very object stored (identity), with a
tag allowed for that role. An absent entry, a zeroed slot or a stale handle would arrive
as garbage or with the wrong tag. Results are compared with the same expression on INT64
with built-in ops. Scenarios: ewise_mult, ewise_add, ewise_union, apply (plain, masked,
bound Scalar), reduce_rowwise and reduce_scalar, mxm (default, dot, value, structural
and complemented masks, A.T, accum with mask and replace), mxv, and ewise_mult with
accum. Each runs on A and B in every pair of sparse, bitmap and full formats, iso and
not iso; then ewise ops on empty iso inputs. Collections run between scenarios, and
objects.threshold is low so that automatic collections happen between calls too.

Run: PYTHONPATH=<worktree> python safety_probe.py [--threads N] [--seed N] [--n N] [--csv PATH]
Success: exit status 0 and a line starting "PASS" with bad=0 wrong=0 (327 rows).
Set GRAPHBLAS_CSHIM=0 for the Numba trampolines, and also GRAPHBLAS_NUMBA_CAPI=0 for
cffi callbacks; the PASS line names the one used (paths=...). On a free-threaded build,
--threads 8 --n 40 lets SuiteSparse run the ops on several threads.
"""

import argparse
import csv
import itertools
import operator
import sys
import time

import numpy as np

import graphblas as gb
from graphblas import Matrix, Scalar, Vector, binary, dtypes, monoid, semiring

try:
    from graphblas.core import objects
    from graphblas.core.operator import pyobject
except ImportError:
    sys.exit(f"{gb.__file__} has no OBJECT dtype: put objdtype-core on PYTHONPATH")

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument(
    "--threads", type=int, default=1, help="SuiteSparse's nthreads setting (default 1)"
)
parser.add_argument("--seed", type=int, default=0, help="random seed (default 0)")
parser.add_argument("--n", type=int, default=10, help="matrices are n by n (default 10)")
parser.add_argument("--csv", metavar="PATH", help="write one row per scenario to PATH")
args = parser.parse_args()
print(f"graphblas {gb.__file__}; Python {sys.version.split()[0]}", flush=True)

OBJECT = dtypes.OBJECT
OUT = args.csv
NTH = args.threads
SEED = args.seed
N = args.n
objects.threshold = 16  # collect often


def python_op_paths():
    """The trampolines that run the Python ops made so far: cshim (C), numba, or cffi."""
    c_name = {"_CHeap": "cshim", "_NumbaHeap": "numba"}.get(type(objects._heap.c).__name__, "c")
    kinds = {c_name if kind == "c" else kind for kind, _ in pyobject._callbacks.values()}
    return ",".join(sorted(kinds))


class V(int):
    pass


LIVE = {}
BAD = []
CALLS = {}


def make(val, tag):
    v = V(val)
    v.tag = tag
    LIVE[id(v)] = v
    return v


def wrap(role, f, allowed):
    def g(*args):
        CALLS[role] = CALLS.get(role, 0) + 1
        objs = [a for a in args if isinstance(a, V) or not isinstance(a, int)]
        if role == "iu":
            objs = [args[0], args[3]]
        for a in objs:
            if not isinstance(a, V) or LIVE.get(id(a)) is not a:
                BAD.append((role, "garbage", type(a).__name__, repr(a)[:60]))
            elif a.tag not in allowed:
                BAD.append((role, "tag", a.tag))
        r = f(*args)
        for a in args:
            if r is a:
                return r
        return make(int(r), "fresh")

    g.__name__ = f"probe_{role}"
    return g


ADD = binary.register_anonymous(wrap("ewise", operator.add, {"A", "B", "extra"}), is_udt=True)
MUL = binary.register_anonymous(wrap("ewise", operator.mul, {"A", "B", "extra"}), is_udt=True)
ACC = binary.register_anonymous(
    wrap("accum", operator.add, {"C", "fresh", "A", "B", "extra"}), is_udt=True
)
NEG = gb.unary.register_anonymous(wrap("apply", operator.neg, {"A"}), is_udt=True)
SMUL = binary.register_anonymous(wrap("mul", operator.mul, {"A", "B", "V"}), is_udt=True)
SADD = binary.register_anonymous(
    wrap("add", operator.add, {"fresh", "ident", "A", "B", "V"}), is_udt=True
)
RADD = binary.register_anonymous(wrap("reduce", operator.add, {"A", "fresh", "ident"}), is_udt=True)
SMIN = binary.register_anonymous(wrap("reduce", min, {"A", "fresh", "ident"}), is_udt=True)
IDENT = make(0, "ident")
PLUS = monoid.register_anonymous(SADD, IDENT)
RPLUS = monoid.register_anonymous(RADD, make(0, "ident"))
BIG = make(10**30, "ident")
RMIN = monoid.register_anonymous(SMIN, BIG)
PLUS_TIMES = semiring.register_anonymous(PLUS, SMUL)


def build(rng, n, fmt_a, fmt_b, iso):
    dense = "full" in (fmt_a, fmt_b)
    out = {}
    for name, base, fmt in (("A", 1, fmt_a), ("B", 1000, fmt_b), ("C", 100000, "sparse")):
        if dense:
            r, c = np.nonzero(np.ones((n, n), bool))
        else:
            r, c = np.nonzero(rng.random((n, n)) < 0.3)
        if iso:
            o = make(7, name)
            # Assign the object under the structure, which makes the matrix iso
            X = Matrix.from_coo(r, c, [o] * len(r), dtype=OBJECT, nrows=n, ncols=n)
            X(X.S) << Scalar.from_value(o, dtype=OBJECT)
            assert X.ss.is_iso
            S = Matrix.from_coo(r, c, 7, dtype="INT64", nrows=n, ncols=n)
        else:
            vals = [make(base + k, name) for k in range(len(r))]
            X = Matrix.from_coo(r, c, vals, dtype=OBJECT, nrows=n, ncols=n)
            S = Matrix.from_coo(r, c, [int(v) for v in vals], dtype="INT64", nrows=n, ncols=n)
        X.ss.config["sparsity_control"] = fmt
        S.ss.config["sparsity_control"] = fmt
        out[name] = (X, S)
    mr, mc = np.nonzero(rng.random((n, n)) < 0.5)
    M = Matrix.from_coo(mr, mc, rng.random(len(mr)) < 0.6, dtype="BOOL", nrows=n, ncols=n)
    vr = np.nonzero(rng.random(n) < 0.5)[0]
    vv = [make(5000 + k, "V") for k in range(len(vr))]
    out["v"] = (
        Vector.from_coo(vr, vv, dtype=OBJECT, size=n),
        Vector.from_coo(vr, [int(x) for x in vv], dtype="INT64", size=n),
    )
    return out, M


def scenarios():
    S = {}

    def pick(w, objmode):
        return w if objmode else None

    def ops(o):
        if o:
            return dict(add=ADD, mul=MUL, acc=ACC, neg=NEG, plus=RPLUS, min=RMIN, pt=PLUS_TIMES)
        return dict(
            add=binary.plus,
            mul=binary.times,
            acc=binary.plus,
            neg=gb.unary.ainv,
            plus=monoid.plus,
            min=monoid.min,
            pt=semiring.plus_times,
        )

    def fill(o, val):
        if o:
            return Scalar.from_value(make(val, "extra"), dtype=OBJECT)
        return Scalar.from_value(val, dtype="INT64")

    S["ewise_mult"] = lambda d, M, o, p: d["A"].ewise_mult(d["B"], p["mul"]).new()
    S["ewise_add"] = lambda d, M, o, p: d["A"].ewise_add(d["B"], p["add"]).new()
    S["ewise_union"] = (
        lambda d, M, o, p: d["A"].ewise_union(d["B"], p["add"], fill(o, -1), fill(o, -2)).new()
    )
    S["apply"] = lambda d, M, o, p: d["A"].apply(p["neg"]).new()
    S["apply_mask"] = lambda d, M, o, p: d["A"].apply(p["neg"]).new(mask=M.S)
    S["apply_bind2nd"] = lambda d, M, o, p: d["A"].apply(p["add"], right=fill(o, 5)).new()
    S["reduce_row"] = lambda d, M, o, p: d["A"].reduce_rowwise(p["plus"]).new()
    S["reduce_scalar"] = lambda d, M, o, p: d["A"].reduce_scalar(p["plus"]).new()
    S["reduce_min"] = lambda d, M, o, p: d["A"].reduce_rowwise(p["min"]).new()
    S["mxm"] = lambda d, M, o, p: d["A"].mxm(d["B"], p["pt"]).new()
    S["mxm_dot"] = lambda d, M, o, p: d["A"].mxm(d["B"], p["pt"]).new(axb_method="dot")
    S["mxm_maskV"] = lambda d, M, o, p: d["A"].mxm(d["B"], p["pt"]).new(mask=M.V)
    S["mxm_maskS"] = lambda d, M, o, p: d["A"].mxm(d["B"], p["pt"]).new(mask=M.S)
    S["mxm_maskC"] = lambda d, M, o, p: d["A"].mxm(d["B"], p["pt"]).new(mask=~M.S)
    S["mxm_AT"] = lambda d, M, o, p: d["A"].T.mxm(d["B"], p["pt"]).new()
    S["mxv"] = lambda d, M, o, p: d["A"].mxv(d["v"], p["pt"]).new()

    def acc_mxm(d, M, o, p):
        out = d["C"].dup()
        out(M.V, accum=p["acc"], replace=True) << d["A"].mxm(d["B"], p["pt"])
        return out

    def acc_ewise(d, M, o, p):
        out = d["C"].dup()
        out(accum=p["acc"]) << d["A"].ewise_mult(d["B"], p["mul"])
        return out

    S["mxm_accum_mask_replace"] = acc_mxm
    S["ewise_accum"] = acc_ewise
    return S, ops


def to_ints(X):
    if X is None:
        return None
    if isinstance(X, Scalar):
        return None if X.is_empty else int(X.value)
    idx = X.to_coo()
    return [i.tolist() for i in idx[:-1]] + [[int(v) for v in idx[-1]]]


def main():
    rows = []
    rng = np.random.default_rng(SEED)
    gb.ss.config["nthreads"] = NTH
    gb.ss.config["chunk"] = 64 if NTH > 1 else 65536
    S, opsof = scenarios()
    fmts = ("sparse", "bitmap", "full")
    t0 = time.time()
    for fa, fb, iso in itertools.product(fmts, fmts, (False, True)):
        data, M = build(rng, N, fa, fb, iso)
        dobj = {k: v[0] for k, v in data.items()}
        dsh = {k: v[1] for k, v in data.items()}
        for name, fn in S.items():
            BAD.clear()
            CALLS.clear()
            exp = to_ints(fn(dsh, M, False, opsof(False)))
            status = "ok"
            try:
                got = to_ints(fn(dobj, M, True, opsof(True)))
                correct = got == exp
            except Exception as exc:
                status = f"{type(exc).__name__}: {exc}"[:160]
                correct = False
            # collect, then check the inputs still resolve to their own objects
            objects.collect()
            for k in ("A", "B", "C"):
                vals = dobj[k].to_coo()[2]
                if not all(isinstance(x, V) and x.tag == k for x in vals):
                    BAD.append(("input", "lost", k))
            rows.append(
                {
                    "nthreads": NTH,
                    "fmt": f"{fa}/{fb}",
                    "iso": iso,
                    "scenario": name,
                    "calls": sum(CALLS.values()),
                    "bad": len(BAD),
                    "bad_first": repr(BAD[:2])[:200],
                    "correct": correct,
                    "status": status,
                }
            )
    # Empty iso inputs: SuiteSparse calls ops on their iso values
    for _ in range(3):
        BAD.clear()
        e1 = Matrix(OBJECT, 1, 0)
        e1 << Scalar.from_value(make(3, "A"), dtype=OBJECT)
        e2 = Matrix(OBJECT, 1, 0)
        e2 << Scalar.from_value(make(4, "B"), dtype=OBJECT)
        objects.collect()
        status = "ok"
        try:
            r = e1.ewise_mult(e2, MUL).new()
            r2 = e1.ewise_union(
                e2,
                ADD,
                Scalar.from_value(make(1, "extra"), dtype=OBJECT),
                Scalar.from_value(make(2, "extra"), dtype=OBJECT),
            ).new()
            correct = r.nvals == 0 and r2.nvals == 0
        except Exception as exc:
            status = f"{type(exc).__name__}: {exc}"[:160]
            correct = False
        rows.append(
            {
                "nthreads": NTH,
                "fmt": "empty-iso",
                "iso": True,
                "scenario": "empty_iso",
                "calls": 0,
                "bad": len(BAD),
                "bad_first": repr(BAD[:2])[:200],
                "correct": correct,
                "status": status,
            }
        )
    if OUT:
        with open(OUT, "w", newline="") as f:
            w = csv.DictWriter(f, list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    nbad = sum(r["bad"] for r in rows)
    nwrong = sum(not r["correct"] for r in rows)
    ok = nbad == 0 and nwrong == 0
    print(
        f"{'PASS' if ok else 'FAIL'} paths={python_op_paths()} nthreads={NTH} "
        f"gil={getattr(sys, '_is_gil_enabled', lambda: True)()} n={N} seed={SEED} "
        f"rows={len(rows)} bad={nbad} wrong={nwrong} "
        f"collections={objects.stats().get('collections')} time={time.time() - t0:.1f}s"
    )
    for r in rows:
        if r["bad"] or not r["correct"]:
            print("  ", r["fmt"], r["iso"], r["scenario"], r["status"], r["bad_first"])
    return 0 if ok else 1


sys.exit(main())
