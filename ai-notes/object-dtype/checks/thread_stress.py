# ruff: noqa: INP001, T201, S101, S311, B905
"""Threads mutate their own OBJECT containers, read shared ones, and collect, all at once.

Each thread runs random steps on its own OBJECT Vector: setitem, delitem, from_coo,
ewise_add with a shared Vector, explicit collections, and Python ops through apply
(new objects) and ewise_add (an input object returned). After every step it checks its
Vector against its own model by identity. At the end, after a forced collection, every
object that no container holds must have been freed. Run it under a GIL build and a
free-threaded build.

Run: PYTHONPATH=<worktree> python thread_stress.py [--threads N] [--steps N] [--threshold N]
Success: exit status 0 and a line starting "PASS" with errors=0 and leaked=0.
Set GRAPHBLAS_CSHIM=0 for the Numba trampolines, and also GRAPHBLAS_NUMBA_CAPI=0 for
cffi callbacks; the PASS line names the one used (paths=...).
"""

import argparse
import gc
import random
import sys
import threading
import time
import weakref

import graphblas as gb
from graphblas import Vector, binary, unary

try:
    from graphblas.core import objects
    from graphblas.core.operator import pyobject
except ImportError:
    sys.exit(f"{gb.__file__} has no OBJECT dtype: put objdtype-core on PYTHONPATH")

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument("--threads", type=int, default=8, help="threads to run (default 8)")
parser.add_argument("--steps", type=int, default=1500, help="steps per thread (default 1500)")
parser.add_argument(
    "--threshold",
    type=int,
    default=16,
    help="objects.threshold: allocations between automatic collections (default 16)",
)
args = parser.parse_args()
print(f"graphblas {gb.__file__}; Python {sys.version.split()[0]}", flush=True)

O = gb.dtypes.OBJECT  # noqa: E741
nthreads, nsteps = args.threads, args.steps
objects.threshold = args.threshold
gil = getattr(sys, "_is_gil_enabled", lambda: True)()


def python_op_paths():
    """The trampolines that run the Python ops made so far: cshim (C), numba, or cffi."""
    c_name = {"_CHeap": "cshim", "_NumbaHeap": "numba"}.get(type(objects._heap.c).__name__, "c")
    kinds = {c_name if kind == "c" else kind for kind, _ in pyobject._callbacks.values()}
    return ",".join(sorted(kinds))


class Tok:
    __slots__ = ("n", "__weakref__")

    def __init__(self, n):
        self.n = n


counter = iter(range(10**9))
lock = threading.Lock()
finalized = set()
created = []


def tok():
    with lock:
        n = next(counter)
    t = Tok(n)
    weakref.finalize(t, finalized.add, n)
    return t


box = unary.register_anonymous(lambda x: (x,), is_udt=True)
keep = binary.register_anonymous(lambda x, y: x, is_udt=True)
shared_vals = [tok() for _ in range(50)]
shared = Vector.from_coo(range(50), shared_vals, dtype=O)
errors = []
checks = [0] * nthreads
barrier = threading.Barrier(nthreads)


def work(tid):
    rng = random.Random(tid)
    size = 20
    model = {}
    v = Vector(O, size)
    barrier.wait()
    try:
        for _ in range(nsteps):
            r = rng.random()
            if r < 0.3:
                i = rng.randrange(size)
                t = tok()
                v[i] = t
                model[i] = t
            elif r < 0.45 and model:
                i = rng.choice(list(model))
                del v[i]
                del model[i]
            elif r < 0.6:
                idx = rng.sample(range(size), rng.randint(0, size))
                vals = [tok() for _ in idx]
                v = Vector.from_coo(idx, vals, dtype=O, size=size)
                model = dict(zip(idx, vals))
            elif r < 0.7:
                w = v.ewise_add(shared[:size].new(), binary.first).new()
                m = {i: shared_vals[i] for i in range(size)}
                m.update(model)
                v, model = w, m
            elif r < 0.75:
                objects.collect()
            elif r < 0.8:
                # Python ops: new objects (boxes), and the input's handle reused
                b = v.apply(box).new()
                bi, bv = b.to_coo()
                assert all(
                    type(t) is tuple and t[0] is model[i] for i, t in zip(bi.tolist(), bv)
                ), "box"
                w = v.ewise_add(v, keep).new()
                v = w
            elif r < 0.85:
                # read the shared container
                _, got = shared.to_coo()
                assert all(a is b for a, b in zip(got, shared_vals)), "shared"
            idx, got = v.to_coo()
            assert dict(zip(idx.tolist(), got)).keys() == model.keys(), "keys"
            assert all(model[i] is o for i, o in zip(idx.tolist(), got)), "identity"
            checks[tid] += 1
    except Exception as exc:
        errors.append(f"thread {tid}: {type(exc).__name__}: {exc}")
    finally:
        results[tid] = model, v


results = [None] * nthreads
t0 = time.time()
threads = [threading.Thread(target=work, args=(i,)) for i in range(nthreads)]
for t in threads:
    t.start()
for t in threads:
    t.join()
elapsed = time.time() - t0
held = {id(o) for m, _ in results for o in m.values()} | {id(o) for o in shared_vals}
held_n = {o.n for m, _ in results for o in m.values()} | {o.n for o in shared_vals}
objects.collect()
gc.collect()
objects.collect()
total = next(counter)
alive = set(range(total)) - finalized
leaked = alive - held_n
ok = not errors and not leaked
print(
    f"{'PASS' if ok else 'FAIL'} paths={python_op_paths()} gil={gil} threads={nthreads} "
    f"steps/thread={nsteps} threshold={objects.threshold} "
    f"checks={sum(checks)} errors={len(errors)} created={total} alive={len(alive)} "
    f"held={len(held_n)} leaked={len(leaked)} collections={objects.stats()['collections']} "
    f"seconds={elapsed:.1f}"
)
for e in errors[:5]:
    print(e)
sys.exit(0 if ok else 1)
