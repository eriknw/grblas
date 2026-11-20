# Python objects as a GraphBLAS dtype (`dtypes.OBJECT`)

## Status

This note describes the branch `objdtype-core`. It builds on stage one
(`pyobject-udt.md`, branch `michelp/object-domain-stage-one`) and goes past that
note's plan. The work was done in an AI-assisted overnight run on 2026-11-19/20.
No person has reviewed the code yet. The note was checked against the code at
the commit that adds it.

Claims carry one of three labels:

- **Verified**: run and checked for correctness, including awkward cases.
- **Measured**: a benchmark with its setup stated and more than one data point.
- **Claimed**: reasoned or read from source, not tested.

All measurements come from one macOS arm64 machine (18 cores) that was shared
with other jobs. Most timings were taken at load 4 to 23, so treat them as
relative. Linux and Windows are untested. SuiteSparse:GraphBLAS was 10.5.2 or
10.5.3 (python-suitesparse-graphblas 10.5.2.0).

## What it does

`Matrix`, `Vector` and `Scalar` with `dtype=object` (`dtypes.OBJECT`) hold any
Python objects. Any Python callable can be a unary, binary or index-unary op, or
the parts of a monoid or semiring, on those values: pass `is_udt=True` when
registering it. SuiteSparse calls the Python function from inside its own
kernels. So `mxm`, masks, accumulation, every storage format, and both JIT
settings work as they do for other UDTs.

```python
import operator
from fractions import Fraction as F
from graphblas import Matrix, binary, monoid, semiring

A = Matrix.from_coo([0, 0, 1], [0, 1, 1], [F(1, 2), F(1, 3), F(2, 3)], dtype=object)
add = binary.register_anonymous(operator.add, is_udt=True)
mul = binary.register_anonymous(lambda x, y: x * y, is_udt=True)
plus_times = semiring.register_anonymous(monoid.register_anonymous(add, F(0)), mul)
C = A.mxm(A, plus_times).new()  # exact: 1/4, 7/18, 4/9
```

(Verified.) Any callable works: `def`, `lambda`, `operator.add`, `min`,
`functools.partial`, classes, and `numba.jit(forceobj=True)` functions. An
exception raised by the function is raised by the call that computes the
expression.

## What changed from stage one

Stage one's core idea stays: 8-byte handles live in native SuiteSparse storage,
and the objects are owned on the Python side. Nothing mirrors the sparse
structure in Python. Four things changed:

1. **Python ops run inside SuiteSparse kernels**, through compiled trampolines,
   instead of the eager plan (extract the values, loop in Python, rebuild). The
   eager plan cannot do `mxm` without materializing every product, and it costs
   2 to 3 times as much per call even where it works (see Performance).
2. **Objects are reclaimed by a tracing mark-and-sweep collector**, instead of
   an append-only domain or explicit compaction. Nothing has to be closed or
   compacted by hand.
3. **Handles index one process-wide heap** (a slot and a generation), instead of
   a dict per domain. The storage type is an opaque record UDT, not `UINT64`,
   so SuiteSparse itself refuses numeric ops on handles. A forged or stale
   handle raises instead of resolving.
4. **OBJECT is part of python-graphblas**, instead of `ObjectMatrix` and
   `ObjectVector` wrappers. Lazy expressions, masks, `<<`, `.new()` and the
   methods work unchanged. The hooks in existing code are small (see Code
   map).

Two conclusions of the stage-one note no longer hold. It is true that raw
`PyObject *` values cannot live in GraphBLAS storage without copy and free
hooks. But SuiteSparse can call Python functions on handles from inside its
kernels, safely (see Safety). And Python can run on SuiteSparse's OpenMP
threads: each worker gets one thread state, pinned for its lifetime. Under the
GIL a call still uses one thread, because more threads are always slower there.
On free-threaded Python, calls use SuiteSparse's thread count.

## Design

### Storage and handles

`OBJECT` is an 8-byte record UDT, `[("pyobject_handle", "<u8")]`, with a JIT C
definition. Built-in numeric and comparison ops refuse it at the SuiteSparse
level. Structural operations copy handles natively: `dup`, extract, assign,
transpose, concat, split, reshape, index selects, `.S` masks, positional apply,
and the `first`, `second`, `any` and `pair` ops and semirings.

A handle is `generation << 40 | slot`, with a 24-bit generation and a 40-bit
slot. Slot 0 is never used. The heap is a segmented slab that never moves.
Each chunk holds 65536 slots: an object array, plus a generation array and an
alive array. A C array of chunk pointers is published with a release store, so
C code can read and claim slots while Python adds chunks.

Bulk stores (`from_coo`, `build`) take one handle per occurrence. A single
value, such as a Scalar, is interned by identity. A result made by a Python op
gets a new handle, unless it `is` one of the op's inputs: then it reuses that
handle. So equal handles imply the same object, but one object may hold several
handles.

### Reclamation

GraphBLAS copies and drops handles freely, but it never invents them. So between
calls, the live set is the union of:

- the present entries of every live OBJECT Matrix, Vector and Scalar, tracked by
  weak reference and read with extractTuples, so bitmap garbage is never
  scanned. A typed monoid on OBJECT keeps its identity in a C Scalar of its
  own, so identities are covered. An empty iso container gives its iso value
  (read by an export), because SuiteSparse can call an op on it;
- "carriers": the handle arrays that `objects.to_handles` returns while values
  are on their way into a container;
- objects pinned with `objects.pin()` (nothing in graphblas pins today).

A collection runs when the allocations since the last one exceed
`max(objects.threshold, objects.GC_FACTOR * live count after the last
collection)`, which is amortized O(1) per allocation. `threshold` is 100,000
and `GC_FACTOR` is 2. `objects.collect()` collects now, and `objects.stats()`
reports the heap's state. Nothing is collected during a call.

### Locks and calls

Calls that touch OBJECT take a reader-writer lock: shared for calls, exclusive
for the collector. Numeric calls take no lock and pay nothing measurable. A
collection that becomes due during a call waits for the call to finish, so
objects made by a Python op survive until they are marked.

Calls that run Python ops are serialized process-wide, because there is one
error state. A Python op may itself run a GraphBLAS call with a Python op only
on its outer call's thread, and only when that call runs on one thread. A call
that waits behind another raises a RuntimeError naming the likely cause only
when the call it waits for has made no progress for `pyobject.wait_timeout`
(60 s). The usual cause is a Python op that waits on another thread that runs
such a call.

### Trampolines

Each Python op gets a compiled C function from a pool (1024 per arity, reused
when an op is garbage collected). It resolves the input handles, calls the
Python callable with vectorcall, claims a heap slot, stores the result and
writes its handle. Slots are claimed in per-thread blocks of 64 by an atomic
add, and counted per block. An input may also be a built-in bool, integer or
float value instead of a handle, and the result may be BOOL.

The pool exists because a GraphBLAS function pointer carries no user data (see
"Would upstream SuiteSparse changes help?" below). Three implementations are
tried in order:

1. **C shim** (`core/operator/_cshim.py`): C source compiled with cffi and the
   system compiler at first use, into `GRAPHBLAS_CSHIM_DIR` (default: a
   `_cshim_build` directory beside the module). `GRAPHBLAS_CSHIM=0` turns it
   off. The build prints two harmless `ld: warning: duplicate -rpath` lines on
   macOS.
2. **Numba trampolines** (`core/operator/_numba_capi.py`), when no compiler is
   available: a Numba cfunc calls the CPython C API through function pointers,
   with LLVM atomics. They cost 1.0 to 1.3 s of compile time per process. They
   are off on Windows and untested on Linux. `GRAPHBLAS_NUMBA_CAPI=0` turns
   them off.
3. **cffi callbacks**, when neither works: about 0.6 to 2 us per call.

### Values in

For OBJECT, a list, any ndarray, any Sequence except `str` and `bytes`, and any
value with `__array__` (such as a pandas Series) give one value per index. Any
other value (a `str`, a Fraction, `None`) is one object stored at every index,
as an iso build. A set, dict or iterator raises TypeError.

### Errors

The first exception is kept, and later calls of the op within that GraphBLAS
call return at once. The exception is raised after the GraphBLAS call returns.
`.new()` drops its result. `C << expr` is not rolled back: entries written after
the error hold handle 0, which raises when read. Ctrl-C works through a SIGINT
handler installed during the call (ctypes and cffi callbacks otherwise swallow
KeyboardInterrupt).

### Threads

With the GIL, or on the cffi path, calls with Python ops run with `nthreads=1`
by descriptor. On a free-threaded build, the C and Numba trampolines use
SuiteSparse's thread count.

### Code map

- `graphblas/core/objects.py`: the dtype, heap, collector and locks.
- `graphblas/core/operator/pyobject.py`: Python ops, `run()`, errors, threads.
- `graphblas/core/operator/_cshim.py`: the C shim (mostly C source).
- `graphblas/core/operator/_numba_capi.py`: the Numba trampolines.
- `graphblas/tests/test_objects.py`: the tests.
- Hooks in 16 existing core files: 209 added and 31 removed lines. The largest
  are in `base.py`, `dtypes.py`, `matrix.py`, `vector.py`, `scalar.py` and
  `operator/base.py` (17 to 32 lines each).

## When SuiteSparse calls user operators

These facts hold for any UDF, not only Python ones. They matter for ops that are
impure or expensive. (Measured with SuiteSparse 10.5.2, all formats, iso and
not, JIT off and on.)

- Ops are never called on absent entries or on bitmap garbage.
- Some ops are called extra times on present values:
  - A masked apply evaluates every entry of the input, then masks, so the waste
    grows as the mask gets sparser: 9996 calls for 97 kept entries with a mask
    of about 1% density.
  - `eWiseUnion` with a bitmap or full operand evaluates `op(a, fill)` and
    `op(a, b)` at the same entry.
  - A masked saxpy with a bitmap or full `A`, a sparse `B` and a bitmap mask
    computes products outside the mask (1.25 to 1.8x). `axb_method="dot"` is
    exact.
- An iso input calls the op once. This includes the iso value of an empty iso
  input.
- A full iso reduce recurses log2(n) times, so it assumes associativity.
- The monoid identity is an operand of reduce, and the result for zero entries.
- An output may alias an input.
- An op without a JIT C definition runs in the generic kernels, so JIT on and
  off perform the same for Python ops.

## Semantics users must know

- Ops must be pure: SuiteSparse may call them on iso values, extra times, and in
  any order.
- Monoid identities are real operands, and associativity is assumed.
- Containers hold references, not copies. `dup()` is shallow, and mutating a
  stored object is visible everywhere it is stored (as with numpy object
  arrays).
- `isequal` on OBJECT compares by identity.
- Values cannot leave the process as raw handles. pickle stores the objects.
  serialize, export, import, pack and unpack of OBJECT values work only within
  one process. A bad imported handle raises when it is resolved.
- `C << expr` is not atomic when a Python op raises.
- `from_coo(..., None)` stores `None` at every index, while `v[i] = None`
  deletes the entry.

## Safety

The design is memory-safe: C code checks every handle before use (the chunk
exists, the generation matches, the slot is alive), so garbage handles raise.
The evidence:

- **Safety probe** (Verified). On a standalone prototype written first (same
  trampoline design, a simpler heap): 55,296 scenario runs and 10.5M recorded
  op calls. They covered 18 configurations (5 trampoline kinds, 2 handle types,
  1 and 4 threads, JIT off and on), each over 16 format pairs
  (sparse, hypersparse, bitmap, full), iso and not, and 51 scenarios (masks,
  accum, replace, transposes, the four mxm methods, union, build with dups,
  bind1st and bind2nd, assign, extract, kronecker). There were zero calls on
  absent, garbage, zero or out-of-range handles, and every result equalled an
  INT64 shadow run. On free-threaded 3.14t at 8 threads: 9,216 runs and 1.02M
  calls, with the same result. A port to this branch (18 scenarios, 9 format
  pairs, iso, empty iso inputs, about 450 collections per run) was clean on the
  GIL and on 3.14t, for the C shim, Numba and cffi paths.
- **Collector fuzz** (Verified). A single-thread fuzz checks the heap against a
  pure-Python model, including Python-op steps. One run had 45,000 steps,
  319,716 identity checks, 6,480 checks that unreferenced objects were freed and
  15,743 collections, with 0 failures. Near the final code, 12 runs (C shim,
  Numba, cffi) had 18,000 steps and 127,242 checks, with 0 failures.
- **Thread stress** (Verified). Threads mutate their own containers, read shared
  ones and collect, all at once, at 8 and 16 threads, with the GIL and
  free-threaded: 72 runs and 1.0M checks, and later 24 of 24 runs on all three
  paths, with 0 errors and 0 leaks.
- **Bugs these found, all fixed**: a race in the C Scalar setter (thread stress);
  the iso value of an empty iso input being collected, and a never-filled iso
  value 0 (fuzz).
- **Reviews.** A review of the prototype's C trampolines found a segfault on a
  forged handle (a bounds check against the next free slot instead of the
  capacity), a deadlock, and possible memory-order races; all were fixed, and
  this branch's design does not have the bounds hole. Two rounds of adversarial
  review of this branch found a nested-call deadlock, a silently skipped handle
  0, a callback leak, a swallowed SystemExit, a numeric ndarray stored as one
  object, and a hang when a Python op waits on another thread's Python-op call,
  among others. All were fixed with tests written to fail first, except those
  listed under Open problems.
- **Test suite.** With this branch, the full suite gives 1369 passed and 161
  skipped on the suitesparse backend, against 1311 and 158 on stage one: the
  difference is exactly the new tests (58 + 3). The vanilla backend gives 846
  and 513. `test_objects.py` passes on each of the three paths, with the GIL
  and on 3.14t (tests that do not apply to a path skip).

**Can objects be useful without being safe?** There is nothing to gain. Raw
`PyObject *` handles kept alive save 0 to 4 ns of a call that costs about
45 ns, and leaking every result saves 3 to 6 ns (Measured, prototype).

## Performance

Cost per call of a Python op called from SuiteSparse, in ns, with int values
(Measured; best of 5, n = 2e5, load 6.0 to 7.2):

| op                          | C shim, GIL | C shim, free-threaded, 1 thread | cffi, GIL | eager, GIL | plain Python       | numpy object   |
| --------------------------- | ----------- | ------------------------------- | --------- | ---------- | ------------------ | -------------- |
| apply                       | 46          | 27                              | 577       | 122        | 16 (list comp.)    | 3.8            |
| ewise_add                   | 52          | 33                              | 850       | 143        | 27 (list comp.)    | 3.8            |
| reduce_rowwise              | 52          | 34                              | 1375      | 101        | 59 (dict loop)     | 7.8 (reduceat) |
| mxm plus_times, per product | 61          | 43                              | 1918      | n/a        | 84 (dict of dicts) | n/a            |

"Eager" is stage one's plan (extract, loop in Python, rebuild). The Numba
trampolines add 2 to 11 ns per call. With Fraction values the arithmetic
dominates: on the C shim with the GIL, apply costs 503 ns, ewise 315, reduce 296
and mxm 328 per product, against 470 and 300 in a list comprehension and 359
for a dict-of-dicts mxm. JIT on and off give the same times. When all values
are floats, a numeric dtype is 15 to 90x faster than any object op.

**Free-threaded Python** is the one setting where object ops gain from
SuiteSparse's threads (Measured):

- On this branch, an int apply cost 24.9 ns per call at 1 thread, 9.4 at 4 and
  13.2 at 16 (C shim; Numba 29.8, 10.5 and 12.7). An op of about 1.7 us went
  from 1684 to 323 ns at 16 threads (5.2x). Cheap ops scaled only after a
  shared counter was removed from the per-result path.
- On the prototype at 8 threads against 1: cheap ops 2.2 to 4.4x, ops of about
  3 us 3.5 to 6.1x (up to 8.3x at 16).
- Ops whose code touches shared objects (Fraction, module globals) get slower
  with threads (Fraction 0.33 to 0.73x at 8), just as with plain Python threads.
- Under the GIL, more threads are never faster (0.05 to 0.99x), which is why
  calls use one thread there.

**End to end** (Measured, each result checked against an independent reference;
ratio = our time / best baseline):

| demo                          | prototype | this branch (estimated) | best baseline         |
| ----------------------------- | --------- | ----------------------- | --------------------- |
| exact big-int walk counts     | 0.8-1.3x  | about 2.0x              | pure Python           |
| exact Fraction shortest paths | 0.9-1.3x  | about 1.1x              | networkx Bellman-Ford |
| why-provenance (frozensets)   | 0.6-1.2x  | about 1.1x              | pure Python           |
| path-carrying BFS             | 0.9-2.5x  | about 2.7x              | networkx              |
| Decimal group-by              | 2-3.5x    | about 3.5x              | dict loop or pandas   |

FP64 and INT64 walk counts are wrong at every node in that demo, which is the
point of exact objects. The "this branch" column divides times taken at load
7.6 by baseline times taken at load 4.4 to 5.2, so it is an estimate. This
branch is 1.1 to 1.4x slower than the prototype on the same demos, and about 2x
on BFS (not yet profiled). The case for objects is exactness and semiring
generality at about pure-Python speed, plus parallel speedups on free-threaded
Python. It is not a numeric performance path.

Memory: a mutation loop with 3M replacements kept at most 195k live objects and
grew RSS by 44 to 87 MB, against 3.1M live objects and +615 MB append-only. It
ran 1.3 to 1.7x slower (1.7 to 2.2 s against 1.3 s; Measured on an earlier heap).
Within one call nothing is collected: an mxm with a 1.0M-entry result claimed
1.4M slots (+191 MB).

## Answers to the stage-one questions

1. _Process- or domain-lifetime retention, or compaction?_ Neither: the
   collector reclaims automatically (see Memory above). Heap chunks never
   shrink.
2. _Delayed expressions?_ OBJECT containers use the normal lazy expressions,
   `.new()` and `<<`. A Python op's exception surfaces when the expression
   computes (Verified).
3. _Deterministic order?_ Not promised, as for any GraphBLAS op. SuiteSparse may
   call ops in any order, on iso values, and extra times, so ops must be pure
   and monoids associative.
4. _Mutable objects?_ Containers hold references, so a mutation is visible
   everywhere the object is stored, and `dup()` is shallow (as with numpy
   object arrays). There are no snapshot semantics.
5. _Serialization?_ pickle stores the objects (Verified). export and import
   carry raw handles and work only within one process (Verified).
   `ss.serialize`, pack and unpack carry raw handles too (Claimed).
6. _Pedagogical or production?_ The evidence supports production use for
   exactness and custom semirings at about pure-Python speed. It is not a
   numeric performance path.

## Other questions checked

**Would #262-style on-change hooks help?** Not needed. Marking costs 2.3 to 7 ns
per stored value (1e7 values in 24 to 36 ms), and sweeping 110 to 215 ns per
freed object (Measured on an earlier heap; sweeps are cheaper now). A write
barrier for incremental marking would save only the marking, which is already
cheap. If one is ever wanted, a hook in `base.call` (`args[0]` is the output)
plus the C Scalar setter covers what #262's 30 or so hook sites cover (Claimed).

**What numpy's object dtype teaches.** numpy object arrays hold real
`PyObject *` values and INCREF and DECREF them in their copy, clear and fill
loops. Its minimal DType hook set is clear, an optional zero-fill, and flags;
copies go through a generic transfer loop. numpy never releases the GIL for
object loops. Legacy loops have no error channel, so it checks
`PyErr_Occurred` after the loop. It refuses raw reinterpretation
(`.view(object)`, `frombuffer`), and it gives object arrays no thread safety
under free-threading. This branch follows numpy on errors (keep the first,
skip the rest, raise after the call) and on one thread under the GIL. It
follows numpy partly on raw reinterpretation: casts to and from OBJECT are
refused, but export, import and serialize carry raw handles within a process.
It departs on refcounting, since SuiteSparse has no copy or free hooks, so a
tracing collector replaces per-cell refcounts. It departs on threads: under
free-threading it runs object ops in parallel.

**Would upstream SuiteSparse changes help?** None are required. These would
help, and none has been filed (sizes are Claimed, from reading the source):

1. Skip the extra op calls listed above: a masked apply (small: apply after
   masking, in `GB_apply.c`), `eWiseUnion` with bitmap or full operands (small
   to moderate: the full and bitmap add templates), and the masked bitmap saxpy
   case (probably small).
2. A user-data pointer on operators, so trampolines need no fixed pool.
   Index-binary ops with `theta` already do this for eWiseMult and a semiring's
   multiply, and index-unary ops with a thunk do it for apply and select. What
   is missing is monoids: `GB_Monoid_new` rejects index-binary ops as
   positional. Medium.
3. An abort flag an operator can set, so a failed op stops the structure walk.
   Medium to large, and low value: this branch already turns later calls into
   no-ops.
4. A per-operator "serial" hint that caps a call at one thread. Small, and a
   convenience: bindings can set `nthreads` by descriptor today.

Per-value copy and free hooks on a type, for refcounting, would be a very large
change: hundreds of copy sites, 2207 factory kernels and the JIT templates
(Claimed). The collector makes them unnecessary.

**Plain Python functions or `numba.jit`?** Any Python callable works with
`is_udt=True`, including Numba dispatchers. No Numba variant beats a plain `def`
on objects: dispatch adds 25 to 30 ns. Bare `numba.jit` means nopython mode on
Numba 0.66, so it raises a TypingError on values Numba cannot type, such as
Fraction. `numba.jit(forceobj=True)` works on any object (Verified).

## Open problems

- Calls with Python ops are serialized process-wide. Per-call error state would
  let independent Python threads run object ops at the same time. A Python op
  that waits on another thread's Python-op call now raises after 60 s without
  progress instead of hanging; per-call error state is the real fix.
- Nothing is collected within one call, and heap chunks never shrink.
- One object may hold several slots: `[x] * n` takes n slots, and a result that
  is not an input gets a new slot. Memory is per occurrence, not per distinct
  object.
- Direct `lib` calls in `ss/` (serialize, sort, iterators, iso, memory usage)
  skip the OBJECT lock. That matters only with a collection running in another
  thread.
- Under the GIL, busy op threads delay explicit `collect()` calls. Automatic
  collections still run, and memory stays bounded.
- The first attach of the C shim can wait once behind a `collect()` loop.
- Handle 0 is allowed in empty iso inputs. That was checked in four accum cases,
  not in every kernel.
- The BFS demo is about 2x slower than on the prototype, even with collection
  off. Not profiled.
- `Vector.ss.itervalues()`, `iteritems()` and `iso_value` fail for every UDT,
  on `main` too: python-graphblas builds the name `GxB_Iterator_get_<name>`
  where it should use the UDT getter.

## Decisions still open

- API spelling: reuse the UDF entry points with `is_udt=True` (as built), or a
  separate API for object ops.
- Ship the C shim prebuilt in python-suitesparse-graphblas instead of compiling
  it at first use. Without a compiler, the Numba trampolines cost 2 to 11 ns
  more per call plus 1.0 to 1.3 s of compile per process. Without Numba either,
  cffi is about 30x slower.
- Whether to support OBJECT on the vanilla backend. It works there, on one
  thread only, and needs Numba for `build`.
- Whether to propose the upstream changes above.

## Roads not taken

- Per-cell refcounting: needs upstream copy and free hooks, a very large change.
- Raw `PyObject *` handles: unsafe, with no measurable speed gain.
- Eager execution: cannot do `mxm` without materializing every product.
- An on-change write barrier (#262) for incremental marking: marking is already
  cheap.

## Scripts

`object-dtype/` holds the scripts behind the evidence above, updated to run
against this branch: the five demos, the collector fuzz, the thread stress, the
safety probe, and the per-call benchmark. `object-dtype/README.md` says how to
run them. At their default sizes each finishes in seconds and passes with the
GIL and on 3.14t, on all three paths (Verified). The defaults are small, so
their timings say little; `demos/run_all.py --large` adds a bigger size.

The fuzz does catch a missing root: with carriers left out of marking, it
fails within 70 steps (Verified).
