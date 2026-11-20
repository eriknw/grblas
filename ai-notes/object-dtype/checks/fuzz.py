# ruff: noqa: INP001, T201, S101, S311, B905, PT018
"""Soundness fuzz for the OBJECT dtype and its mark-and-sweep collector.

Random operations on OBJECT matrices, vectors and scalars, interleaved with forced and
automatic collections, checked after every step against a pure Python model (per
container, a dict from coordinate to object, compared by identity). The operations
include from_coo, build, setitem (also while holding the collector's lock, as a
callback would), delitem, extract, assign, iso fills, masked assign, ewise ops, mxm,
transpose, select, concat and split, resize, and Python ops through apply and
ewise_add. Every 10 steps and after each explicit collect, a forced collection must
leave alive exactly the objects the model holds (counted with weakref.finalize), and
the heap's live count must cover the model.

Run: PYTHONPATH=<worktree> python fuzz.py [--seed N] [--steps N] [--threshold N]
For many seeds, loop in the shell: each run is one seed in a fresh process.
Success: exit status 0 and a line "PASS seed=... {stats}". A failure prints the seed,
the step and the operation, then raises.
Set GRAPHBLAS_CSHIM=0 for the Numba trampolines, and also GRAPHBLAS_NUMBA_CAPI=0 for
cffi callbacks; the PASS line names the one used (paths=...).
"""

import argparse
import gc
import random
import sys
import time
import weakref

import numpy as np

import graphblas as gb
from graphblas import Matrix, Scalar, Vector, binary, semiring, unary

try:
    from graphblas.core import objects
    from graphblas.core.operator import pyobject
except ImportError:
    sys.exit(f"{gb.__file__} has no OBJECT dtype: put objdtype-core on PYTHONPATH")

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument("--seed", type=int, default=41, help="random seed (default 41)")
parser.add_argument("--steps", type=int, default=1000, help="operations to run (default 1000)")
parser.add_argument(
    "--threshold",
    type=int,
    default=8,
    help="objects.threshold: allocations between automatic collections (default 8)",
)
args = parser.parse_args()
print(f"graphblas {gb.__file__}; Python {sys.version.split()[0]}", flush=True)

O = gb.dtypes.OBJECT  # noqa: E741
seed = args.seed
nsteps = args.steps
objects.threshold = args.threshold
rng = random.Random(seed)


def python_op_paths():
    """The trampolines that run the Python ops made so far: cshim (C), numba, or cffi."""
    c_name = {"_CHeap": "cshim", "_NumbaHeap": "numba"}.get(type(objects._heap.c).__name__, "c")
    kinds = {c_name if kind == "c" else kind for kind, _ in pyobject._callbacks.values()}
    return ",".join(sorted(kinds))


class Tok:
    __slots__ = ("n", "__weakref__")

    def __init__(self, n):
        self.n = n

    def __repr__(self):
        return f"Tok({self.n})"


ntoks = 0
finalized = set()


def fresh():
    global ntoks
    r = rng.random()
    if r < 0.03:
        return None
    if r < 0.06:
        return rng.choice([7, "s", (1, 2)])  # immortal-ish / shared objects, not counted
    t = Tok(ntoks)
    weakref.finalize(t, finalized.add, ntoks)
    ntoks += 1
    return t


class Var:
    __slots__ = ("kind", "c", "m", "shape")

    def __init__(self, kind, c, m, shape):
        self.kind, self.c, self.m, self.shape = kind, c, m, shape


env = {}
counts = {}
names = [f"x{i}" for i in range(8)]


def live_model_objects():
    seen = {}
    for var in env.values():
        vals = var.m.values() if var.kind != "S" else ([var.m["v"]] if "v" in var.m else [])
        for o in vals:
            seen[id(o)] = o
    return seen


def pick_obj():
    # reuse an object already stored half the time
    if rng.random() < 0.5:
        seen = list(live_model_objects().values())
        if seen:
            return rng.choice(seen)
    return fresh()


def rand_shape(kind):
    if kind == "M":
        return (rng.randint(0, 6), rng.randint(0, 6))
    return (rng.randint(0, 8),)


def all_coords(shape):
    if len(shape) == 2:
        return [(i, j) for i in range(shape[0]) for j in range(shape[1])]
    return [(i,) for i in range(shape[0])]


def key(coord):
    return coord if len(coord) == 2 else coord[0]


def new_from_coo(kind, shape=None):
    shape = shape or rand_shape(kind)
    coords = all_coords(shape)
    k = rng.randint(0, len(coords))
    chosen = rng.sample(coords, k)
    vals = [pick_obj() for _ in chosen]
    if rng.random() < 0.2 and vals:  # iso-ish: all the same object
        o = pick_obj()
        vals = [o] * len(vals)
    if kind == "M":
        c = Matrix.from_coo(
            [x[0] for x in chosen],
            [x[1] for x in chosen],
            vals,
            dtype=O,
            nrows=shape[0],
            ncols=shape[1],
        )
    else:
        c = Vector.from_coo([x[0] for x in chosen], vals, dtype=O, size=shape[0])
    return Var(kind, c, {key(x): v for x, v in zip(chosen, vals)}, shape)


def choose(kind=None, nonempty_shape=False):
    cands = [n for n, v in env.items() if (kind is None or v.kind == kind)]
    if nonempty_shape:
        cands = [n for n in cands if all(s > 0 for s in env[n].shape)]
    return rng.choice(cands) if cands else None


def op_new():
    env[rng.choice(names)] = new_from_coo(rng.choice("MMV"))


def op_scalar():
    o = pick_obj()
    s = Scalar.from_value(o, dtype=O, is_cscalar=rng.random() < 0.5)
    env[rng.choice(names)] = Var("S", s, {} if o is None else {"v": o}, ())


def op_extract_element():
    n = choose(nonempty_shape=True)
    if n is None or env[n].kind == "S":
        return
    var = env[n]
    coord = tuple(rng.randrange(s) for s in var.shape)
    s = var.c[coord if len(coord) == 2 else coord[0]].new()
    m = {"v": var.m[key(coord)]} if key(coord) in var.m else {}
    env[rng.choice(names)] = Var("S", s, m, ())


def op_build():
    n = choose()
    if n is None or env[n].kind == "S":
        return
    var = env[n]
    clear = rng.random() < 0.5 or bool(var.m)  # GrB build needs an empty output
    if not clear:
        free = [x for x in all_coords(var.shape) if key(x) not in var.m]
    else:
        free = all_coords(var.shape)
    chosen = rng.sample(free, rng.randint(0, len(free)))
    vals = [pick_obj() for _ in chosen]
    if var.kind == "M":
        var.c.build([x[0] for x in chosen], [x[1] for x in chosen], vals, clear=clear)
    else:
        var.c.build([x[0] for x in chosen], vals, clear=clear)
    if clear:
        var.m.clear()
    var.m.update({key(x): v for x, v in zip(chosen, vals)})


def op_setitem():
    n = choose(nonempty_shape=True)
    if n is None or env[n].kind == "S":
        return
    var = env[n]
    coord = tuple(rng.randrange(s) for s in var.shape)
    o = pick_obj()
    if o is None:
        o = fresh() or Tok(-1)
    var.c[coord if len(coord) == 2 else coord[0]] = o
    var.m[key(coord)] = o


def op_setitem_locked():
    """Several setitems while holding the collector's lock shared, as a callback would.

    Automatic collections requested inside are deferred to the release.
    """
    n = choose(nonempty_shape=True)
    if n is None or env[n].kind == "S":
        return
    var = env[n]
    with objects._lock:
        for _ in range(rng.randint(1, 4)):
            coord = tuple(rng.randrange(s) for s in var.shape)
            o = fresh() or Tok(-2)
            var.c[coord if len(coord) == 2 else coord[0]] = o
            var.m[key(coord)] = o


def op_delitem():
    n = choose()
    if n is None or env[n].kind == "S" or not env[n].m:
        return
    var = env[n]
    k = rng.choice(list(var.m))
    del var.c[k]
    del var.m[k]


def op_dup():
    n = choose()
    if n is None:
        return
    var = env[n]
    env[rng.choice(names)] = Var(var.kind, var.c.dup(), dict(var.m), var.shape)


def op_extract():
    n = choose(nonempty_shape=True)
    if n is None or env[n].kind == "S":
        return
    var = env[n]
    idx = [
        [rng.randrange(s) for _ in range(rng.randint(0, s + 1))] for s in var.shape
    ]  # repeats ok
    if var.kind == "M":
        c = var.c[np.array(idx[0], np.int64), np.array(idx[1], np.int64)].new()
        m = {
            (a, b): var.m[(i, j)]
            for a, i in enumerate(idx[0])
            for b, j in enumerate(idx[1])
            if (i, j) in var.m
        }
    else:
        c = var.c[np.array(idx[0], np.int64)].new()
        m = {a: var.m[i] for a, i in enumerate(idx[0]) if i in var.m}
    env[rng.choice(names)] = Var(var.kind, c, m, tuple(len(x) for x in idx))


def op_assign():
    n = choose(nonempty_shape=True)
    if n is None or env[n].kind == "S":
        return
    var = env[n]
    idx = [rng.sample(range(s), rng.randint(0, s)) for s in var.shape]  # unique
    src = new_from_coo(var.kind, tuple(len(x) for x in idx)) if rng.random() < 0.5 else None
    if src is None:
        cands = [
            v for v in env.values() if v.kind == var.kind and v.shape == tuple(len(x) for x in idx)
        ]
        src = rng.choice(cands) if cands else new_from_coo(var.kind, tuple(len(x) for x in idx))
    srcm = dict(src.m)  # src may be var itself
    if var.kind == "M":
        var.c[np.array(idx[0], np.int64), np.array(idx[1], np.int64)] = src.c
        for a, i in enumerate(idx[0]):
            for b, j in enumerate(idx[1]):
                if (a, b) in srcm:
                    var.m[(i, j)] = srcm[(a, b)]
                else:
                    var.m.pop((i, j), None)
    else:
        var.c[np.array(idx[0], np.int64)] = src.c
        for a, i in enumerate(idx[0]):
            if a in srcm:
                var.m[i] = srcm[a]
            else:
                var.m.pop(i, None)


def op_iso_fill():
    n = choose()
    if n is None or env[n].kind == "S":
        return
    var = env[n]
    o = fresh() or Tok(-3)
    if var.kind == "M":
        var.c[:, :] = o
    else:
        var.c[:] = o
    var.m.clear()
    var.m.update({key(x): o for x in all_coords(var.shape)})


def op_masked():
    """C(B.S) << A, with A and B the same shape as C."""
    n = choose()
    if n is None or env[n].kind == "S":
        return
    var = env[n]
    a = new_from_coo(var.kind, var.shape)
    b = new_from_coo(var.kind, var.shape)
    var.c(b.c.S) << a.c
    for k in b.m:
        if k in a.m:
            var.m[k] = a.m[k]
        else:
            var.m.pop(k, None)


def op_ewise():
    n = choose()
    if n is None or env[n].kind == "S":
        return
    var = env[n]
    other = [v for v in env.values() if v.kind == var.kind and v.shape == var.shape]
    o = rng.choice(other)
    opname = rng.choice(["first", "second", "any"])
    if rng.random() < 0.5:
        c = var.c.ewise_add(o.c, getattr(binary, opname)).new()
        ks = set(var.m) | set(o.m)
    else:
        if opname == "any":
            opname = "first"
        c = var.c.ewise_mult(o.c, getattr(binary, opname)).new()
        ks = set(var.m) & set(o.m)
    m = {}
    for k in ks:
        if k in var.m and k in o.m:
            if opname == "first":
                m[k] = var.m[k]
            elif opname == "second":
                m[k] = o.m[k]
            else:
                got = c[k].new().value
                assert got is var.m[k] or got is o.m[k], ("any", got)
                m[k] = got
        else:
            m[k] = var.m[k] if k in var.m else o.m[k]
    env[rng.choice(names)] = Var(var.kind, c, m, var.shape)


def op_transpose():
    n = choose("M")
    if n is None:
        return
    var = env[n]
    c = var.c.T.new() if rng.random() < 0.5 else var.c.T.dup()
    env[rng.choice(names)] = Var(
        "M", c, {(j, i): v for (i, j), v in var.m.items()}, var.shape[::-1]
    )


def op_select():
    n = choose("M")
    if n is None:
        return
    var = env[n]
    which = rng.choice(["tril", "triu", "diag", "offdiag"])
    c = var.c.select(which).new()
    keep = {
        "tril": lambda i, j: j <= i,
        "triu": lambda i, j: j >= i,
        "diag": lambda i, j: i == j,
        "offdiag": lambda i, j: i != j,
    }[which]
    env[rng.choice(names)] = Var("M", c, {k: v for k, v in var.m.items() if keep(*k)}, var.shape)


def op_mxm():
    n = choose("M")
    if n is None:
        return
    a = env[n]
    if a.shape[1] == 0:
        # SuiteSparse (8.2.1 through 10.5.2): C = A*B with iso operands and an inner
        # dimension of 0 returns entries. Not an OBJECT question, so skip it.
        return
    cands = [v for v in env.values() if v.kind == "M" and v.shape[0] == a.shape[1]]
    b = rng.choice(cands) if cands else new_from_coo("M", (a.shape[1], rng.randint(0, 5)))
    which = rng.choice(["any_first", "any_second"])
    c = a.c.mxm(b.c, getattr(semiring, which)).new()
    m = {}
    for i in range(a.shape[0]):
        for j in range(b.shape[1]):
            cands_ = [
                (a.m[(i, k)] if which == "any_first" else b.m[(k, j)])
                for k in range(a.shape[1])
                if (i, k) in a.m and (k, j) in b.m
            ]
            if cands_:
                got = c[i, j].new().value
                assert any(got is x for x in cands_), ("mxm", which)
                m[(i, j)] = got
    env[rng.choice(names)] = Var("M", c, m, (a.shape[0], b.shape[1]))


def op_concat_split():
    n = choose("M")
    if n is None:
        return
    a = env[n]
    b = new_from_coo("M", (a.shape[0], rng.randint(0, 4)))
    c = gb.ss.concat([[a.c, b.c]])
    m = dict(a.m)
    m.update({(i, j + a.shape[1]): v for (i, j), v in b.m.items()})
    shape = (a.shape[0], a.shape[1] + b.shape[1])
    if rng.random() < 0.5 and all(s > 0 for s in shape):
        # split into tiles and keep one
        rs, cs = rng.randint(1, shape[0]), rng.randint(1, shape[1])
        tiles = c.ss.split((rs, cs))
        ti, tj = rng.randrange(len(tiles)), rng.randrange(len(tiles[0]))
        t = tiles[ti][tj]
        r0, c0 = ti * rs, tj * cs
        m = {
            (i - r0, j - c0): v
            for (i, j), v in m.items()
            if r0 <= i < r0 + t.nrows and c0 <= j < c0 + t.ncols
        }
        c, shape = t, t.shape
        del tiles
    env[rng.choice(names)] = Var("M", c, m, shape)


def op_clear():
    n = choose()
    if n is None:
        return
    var = env[n]
    var.c.clear()
    var.m.clear()


def op_resize():
    n = choose()
    if n is None or env[n].kind == "S":
        return
    var = env[n]
    if var.kind == "M":
        shape = rand_shape("M")
        var.c.resize(*shape)
        var.m = {k: v for k, v in var.m.items() if k[0] < shape[0] and k[1] < shape[1]}
    else:
        shape = rand_shape("V")
        var.c.resize(shape[0])
        var.m = {k: v for k, v in var.m.items() if k < shape[0]}
    var.shape = shape


def op_delete():
    if (n := choose()) is not None:
        del env[n]


def op_alias():
    if (n := choose()) is not None:
        env[rng.choice(names)] = env[n]


def op_collect():
    objects.collect()


class Box:
    """Made by a Python op (in C or cffi) from a stored object."""

    __slots__ = ("x", "__weakref__")

    def __init__(self, x):
        self.x = x
        nbox[0] += 1
        weakref.finalize(self, box_freed.append, 1)


nbox = [0]
box_freed = []
box_op = unary.register_anonymous(Box, is_udt=True)
same_op = unary.register_anonymous(lambda x: x, is_udt=True)
pick_op = binary.register_anonymous(lambda x, y: y if isinstance(y, Box) else x, is_udt=True)


def op_pyapply():
    """Apply a Python op: Box each object (new handles) or return it (input handle reused)."""
    n = choose()
    if n is None or env[n].kind == "S":
        return
    var = env[n]
    if rng.random() < 0.5:
        c = var.c.apply(box_op).new()
        got = c.to_coo()[-1]
        assert all(type(b) is Box for b in got)
        m = {}
        idx = c.to_coo()[:-1]
        coords = list(zip(*[i.tolist() for i in idx])) if var.kind == "M" else idx[0].tolist()
        for k, b in zip(coords, got):
            assert b.x is var.m[k], "box content"
            m[k] = b
    else:
        c = var.c.apply(same_op).new()
        m = dict(var.m)
    env[rng.choice(names)] = Var(var.kind, c, m, var.shape)


def op_pyewise():
    """ewise_add with a Python op that keeps Boxes: results are stored objects (reused handles)."""
    n = choose()
    if n is None or env[n].kind == "S":
        return
    var = env[n]
    other = [v for v in env.values() if v.kind == var.kind and v.shape == var.shape]
    o = rng.choice(other)
    c = var.c.ewise_add(o.c, pick_op).new()
    m = {}
    for k in set(var.m) | set(o.m):
        if k in var.m and k in o.m:
            y = o.m[k]
            m[k] = y if isinstance(y, Box) else var.m[k]
        else:
            m[k] = var.m[k] if k in var.m else o.m[k]
    env[rng.choice(names)] = Var(var.kind, c, m, var.shape)


OPS = [
    (op_new, 6),
    (op_scalar, 2),
    (op_extract_element, 2),
    (op_build, 3),
    (op_setitem, 5),
    (op_setitem_locked, 2),
    (op_delitem, 4),
    (op_dup, 3),
    (op_extract, 3),
    (op_assign, 3),
    (op_iso_fill, 2),
    (op_masked, 2),
    (op_ewise, 4),
    (op_transpose, 2),
    (op_select, 2),
    (op_mxm, 2),
    (op_concat_split, 2),
    (op_clear, 1),
    (op_resize, 2),
    (op_delete, 3),
    (op_alias, 1),
    (op_collect, 3),
    (op_pyapply, 3),
    (op_pyewise, 2),
]
op_funcs = [f for f, w in OPS for _ in range(w)]


def check_var(var):
    c = var.c
    if var.kind == "S":
        if "v" in var.m:
            assert not c.is_empty and c.value is var.m["v"], "scalar"
        else:
            assert c.is_empty, "scalar empty"
        return
    assert c.shape == var.shape, (c.shape, var.shape)
    if var.kind == "M":
        r, cc, v = c.to_coo()
        got = {(int(i), int(j)): o for i, j, o in zip(r, cc, v)}
    else:
        idx, v = c.to_coo()
        got = {int(i): o for i, o in zip(idx, v)}
    assert v.dtype == object
    assert got.keys() == var.m.keys(), ("keys", sorted(got.keys() ^ var.m.keys())[:5])
    for k, o in got.items():
        assert o is var.m[k], ("identity", k, o, var.m[k])


stats = {
    "steps": 0,
    "checks": 0,
    "forced_gc": 0,
    "auto_gc": 0,
    "freed_checks": 0,
    "live_exact": 0,
    "live_over": 0,
    "objects_created": 0,
}


def check_freed():
    """After a forced collection: every Tok not in the model is finalized, and none in it is."""
    objects.collect()
    gc.collect()
    objects.collect()  # objects freed by Python's gc may drop containers
    model = live_model_objects()
    # An empty iso container keeps its iso value alive: SuiteSparse may call an op on it
    iso_kept = []
    for var in env.values():
        if var.kind != "S" and var.c.nvals == 0 and var.c.ss.is_iso:
            vals = var.c.ss.export(raw=True)["values"][:1]
            if vals.size and vals.view(np.uint64)[0] != 0:  # 0: never filled
                iso_kept.extend(objects.to_objects(vals).tolist())
    stats["empty_iso_kept"] = stats.get("empty_iso_kept", 0) + len(iso_kept)
    # A Box in the model keeps the object it boxes alive (which may be a Box)
    reach = {}
    stack = list(model.values()) + iso_kept
    while stack:
        o = stack.pop()
        if id(o) not in reach:
            reach[id(o)] = o
            if type(o) is Box:
                stack.append(o.x)
    model_toks = {o.n for o in reach.values() if isinstance(o, Tok) and o.n >= 0}
    alive = set(range(ntoks)) - finalized
    assert not (model_toks & finalized), "an object in the model was finalized"
    extra = alive - model_toks
    assert not extra, f"{len(extra)} objects alive but in no container: {sorted(extra)[:5]}"
    boxes_held = sum(type(o) is Box for o in reach.values())
    assert nbox[0] - len(box_freed) == boxes_held, ("boxes", nbox[0], len(box_freed), boxes_held)
    heap_live = objects.stats()["live"]
    assert heap_live >= len(model), (heap_live, len(model))
    stats["freed_checks"] += 1
    # The monoid any[OBJECT] keeps its identity (the object 0) in a Scalar of its own
    extra_live = heap_live - len(model)
    stats["live_extra_max"] = max(stats.get("live_extra_max", 0), extra_live)
    if heap_live == len(model):
        stats["live_exact"] += 1
    else:
        stats["live_over"] += 1


t0 = time.time()
gc0 = objects.stats()["collections"]
for step in range(nsteps):
    f = rng.choice(op_funcs)
    try:
        f()
    except Exception as exc:
        print(f"seed {seed} step {step} {f.__name__}: {type(exc).__name__}: {exc}")
        raise
    stats["steps"] += 1
    for name, var in env.items():
        try:
            check_var(var)
        except Exception as exc:
            print(f"seed {seed} step {step} after {f.__name__}: check {name} failed: {exc!r}")
            raise
        stats["checks"] += 1
    var = None  # the loop variable would keep the last checked container alive
    if step % 10 == 9 or f is op_collect:
        try:
            check_freed()
        except AssertionError as exc:
            print(f"seed {seed} step {step} after {f.__name__}: freed check failed: {exc}")
            raise
        stats["forced_gc"] += 1
    counts[f.__name__] = counts.get(f.__name__, 0) + 1
stats["objects_created"] = ntoks
stats["boxes_made_by_python_ops"] = nbox[0]
stats["finalized"] = len(finalized)
stats["collections_total"] = objects.stats()["collections"] - gc0
stats["seconds"] = round(time.time() - t0, 1)
print(f"PASS seed={seed} threshold={objects.threshold} paths={python_op_paths()}", stats)
print("op counts", dict(sorted(counts.items())))
