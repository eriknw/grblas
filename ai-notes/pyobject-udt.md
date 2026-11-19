# PyObject-backed GraphBLAS UDT: design notes

## Status and intent

This note explores an experimental value domain whose stored value is a live
`PyObject *`. A sparse Matrix or Vector of this type could contain arbitrary
Python objects and invoke Python functions as GraphBLAS operators.

This is not a normal NumPy UDT. Existing python-graphblas UDTs are fixed-size,
inline byte layouts represented by `numpy.dtype`, and their operators compile
through Numba and, on SuiteSparse:GraphBLAS, can use generated C/JIT kernels.
A live Python object domain has different ownership, copying, failure, and
threading rules. It should be an explicitly opt-in, backend-specific feature
rather than an extension of `dtypes.register_new`.

The useful target is sparse structural computation over rich Python payloads:
for example, symbolic expressions, AST nodes, application records, graph-search
states, provenance values, or immutable user-domain objects. It is not a path
to numeric GraphBLAS performance.

## Feasibility boundary

At the GraphBLAS C level, a UDT is a fixed number of bytes copied, moved, and
destroyed by the implementation without Python-aware hooks. A pointer-sized UDT
can therefore physically store a `PyObject *`, but GraphBLAS has no type hooks
for `Py_INCREF`, `Py_DECREF`, traversal, finalization, or error propagation.
Blindly registering `sizeof(PyObject *)` with `GrB_Type_new` is unsafe.

The unsafe cases are fundamental:

- A GraphBLAS internal copy duplicates a pointer without incrementing its
  reference count, creating a potential use-after-free and later double-decref.
- Clearing, replacing, resizing, extracting, materializing, freeing, or
  abandoning a matrix drops pointer bytes without `Py_DECREF`, leaking objects.
- Internal temporaries may duplicate, discard, or retain values at times that
  Python cannot observe.
- SuiteSparse kernels may call a user operator in parallel and without the GIL.
- A Python exception cannot be reported through the `void`-style GraphBLAS
  binary-op ABI.

Consequently, a direct in-engine `PyObject *` UDT is not viable with the
portable GraphBLAS API or SuiteSparse's ordinary UDT/operator interfaces.

## Candidate representations

### A. Raw owned pointers in GraphBLAS storage

Store `PyObject *` bytes directly in a `GrB_Type` and teach wrapper entry and
exit points to reference-count values.

This is only viable if the backend adds ownership callbacks for every value
copy, overwrite, discard, and destruction, and guarantees them for all
temporaries. It also needs callback invocation with the GIL held. Neither is
provided by the current API. Do not pursue this as a wrapper-only design.

### B. Borrowed pointers plus a Python owner registry

Store pointer bytes but retain strong references in a Python-side registry for
every object that can appear in a matrix. This prevents use-after-free, but
does not solve identity accounting: deleting one matrix entry cannot determine
whether the pointer remains in another entry or hidden GraphBLAS temporary.
The registry becomes a leak unless it is global and permanent. Pointer reuse
after an object's destruction also makes raw addresses unsuitable as stable
logical identifiers.

This can be useful only as a tightly scoped debugging prototype with a global
keepalive and deliberate process-lifetime leaks. It is not a user feature.

### C. Handle/index UDT with an external object arena

Store an integer handle, not a pointer, in GraphBLAS. A Python-owned arena maps
handles to strong references. GraphBLAS may freely copy handles because they
are plain scalar bytes. All dereferencing happens in Python-aware code.

This is the most promising representation, but it still has a key limitation:
GraphBLAS does not tell the arena when internal handle copies or discards occur.
Use handles as *interned immutable object identities*, not reference-counted
per-cell resources. The arena owns each object independently of the number of
matrix occurrences; matrix lifetime is tracked at a coarser level.

Possible lifetime policies:

- **Append-only arena:** handles keep objects alive until the arena is closed.
  Simple and correct, but unbounded for long-lived mutation-heavy workloads.
- **Matrix-rooted arenas:** each matrix retains an arena; operations produce a
  result arena that retains its input arenas plus newly created objects. This
  preserves correctness but can retain large object graphs transitively.
- **Explicit compaction:** rebuild a matrix and arena by extracting visible
  handles, remapping only reachable objects, and dropping old arenas. This is
  predictable but O(number of stored values) and cannot account for backend
  temporaries until an operation completes.

Handles should contain an arena identity plus a generation/index, or use a
single monotonic 64-bit handle space. An arena must never recycle a handle while
any matrix or delayed expression could still contain it. Pointer addresses
should not be exposed as values.

### D. Python-managed sparse matrix implementation

Implement the same high-level Matrix/Vector operations over a Python-owned
coordinate/CSR/CSC structure whose slots are real Python references. This can
call Python operators naturally and implement exact ownership and GC behavior.
It is no longer execution by a GraphBLAS backend, though it could share the
public expression API and use GraphBLAS only for structure or compatible
numeric subproblems.

This is the cleanest semantic model if arbitrary objects and arbitrary Python
callables are core requirements.

## Recommended architecture: a separate object domain

Start with option C, but make its constraints visible in the API. For example,
an experimental `ObjectDomain` creates `ObjectMatrix` and `ObjectVector`
instances with an opaque `UINT64` (or suitable fixed-width) GraphBLAS UDT and a
retaining object arena. Do not make `dtype=object` mean this: NumPy's object
arrays have ownership semantics that cannot be passed to GraphBLAS storage.

The object domain should initially support only eager operations implemented by
the Python layer. A structural GraphBLAS operation is permitted only when its
stored handle values are not interpreted by backend callbacks. Examples include
structural select/extract/assign where the wrapper can safely maintain arena
roots, depending on the exact backend operation and its result semantics.

For value-transforming work, materialize visible handles, run Python code under
normal Python ownership, intern outputs, then build the result. This creates a
correct baseline and makes the performance boundary explicit. An optimized
backend integration should only follow a documented backend contract for
object-value lifecycle callbacks.

## Garbage collection and ownership

The arena holds normal strong Python references, so Python's cyclic GC can
collect cycles that are not reachable from an arena. However, an object may
refer to its matrix or arena, and the matrix retains the arena. Such a cycle is
collectable only if no external reference remains and the participating wrapper
objects are GC-tracked and do not use problematic finalizers.

Design rules:

- Arena entries own a strong reference. Matrix cells own only integer handles.
- A matrix retains every arena from which its visible or potentially delayed
  values may originate. A result must retain source arenas until materialized.
- Matrix mutation does not immediately decref a removed object's arena entry.
  Reclamation happens by arena close/compaction, never by observing individual
  backend value replacement.
- `close()` invalidates its matrices and releases arena references only after
  pending expressions are resolved or rejected. Operations on a closed domain
  raise a clear error.
- Do not allow an arena to release an entry by weak reference. A handle must
  always dereference reliably while it is valid.
- Provide `compact()` as the explicit answer to retention, and document that it
  forces materialization and may invalidate old handle-level views.

Python finalizers should not call GraphBLAS while interpreter shutdown is in
progress. Wrapper cleanup should be idempotent and tolerate a partially torn
down interpreter/backend.

## Matrix representation and interchange

Internally, GraphBLAS sees a sparse matrix of opaque handles. The Python wrapper
must preserve the arena root(s), dimensions, and any pending-operation state.
Iteration, scalar extraction, `to_coo`, and repr dereference handles in Python.
Bulk construction should intern objects in one pass and pass contiguous handle
arrays to GraphBLAS, avoiding per-entry GraphBLAS calls.

Important semantic choices:

- An absent entry is still GraphBLAS absence, not Python `None`.
- `None` is a valid stored object and must have its own handle.
- Equality of handles means object identity/interning identity, not necessarily
  `==`. Any value equality must be an explicit Python binary operator.
- Duplicated input coordinates need a specified Python duplicate reducer; no
  implicit backend `second` callback should manipulate handles unless it is
  purely selecting a handle and the output arena retains both inputs.
- Serialization cannot write live objects as raw handles. Require a codec
  (such as pickle with explicit security warnings, or user `encode`/`decode`)
  and reconstruct a fresh arena on load.
- Copying between processes, GPUs, remote backends, or non-SuiteSparse
  implementations is unsupported unless the values are encoded and rebuilt.

## Operators, monoids, and semirings

Python functions cannot be registered through the existing Numba UDF path.
Provide distinct object-domain operator wrappers, for example:

```python
merge = object_binary.register(lambda left, right: left.merge(right))
choose = object_monoid.register(lambda left, right: left if score(left) < score(right) else right,
                                identity=EMPTY)
extend = object_binary.register(lambda path, edge: path + (edge,))
paths = object_semiring.register(choose, extend)
```

The wrapper must define:

- Input and output are Python references obtained from arena handles; output is
  interned into the destination arena.
- A callback may return any Python object, including `None`, but may not mutate
  input objects if deterministic/algebraic behavior is expected.
- Exceptions are captured at the operation boundary, abort further Python work,
  release partial Python-owned outputs, and re-raise the original exception.
- Functions should be pure for reproducibility. GraphBLAS may choose evaluation
  order and grouping; side effects make results backend- and schedule-dependent.

A monoid additionally needs an identity object and should be associative. It
may declare commutativity and idempotence only if true for the actual object
semantics. `any` is a special selection operation, not a deterministic choice.

A semiring needs an additive monoid and multiplicative binary op over compatible
object domains. Matrix multiplication may evaluate and combine terms in an
unspecified order. A Python fallback can preserve a documented deterministic
order, but then differs from ordinary GraphBLAS scheduling and should say so.
Do not expose a generic object semiring as GraphBLAS `mxm` until its execution
and exception model is defined end-to-end.

## GIL, threads, and cost model

Every Python callback requires the GIL. If SuiteSparse invokes it from OpenMP
workers, each invocation would acquire the GIL and serialize bytecode execution;
it may also be illegal if the backend has not initialized thread state for those
workers. Acquiring and releasing the GIL once per scalar operation can dominate
the work by orders of magnitude, even for inexpensive functions.

The first implementation should never invoke Python from backend worker threads.
For object-valued operations, materialize work in the wrapper and execute it in
one Python thread with the GIL held. Set or require `nthreads=1` for any backend
phase that could invoke an object callback. Parallelism may still be useful for
pure structural operations that do not touch objects.

Expected costs include:

- One or more handle-to-object lookups and output interning per callback.
- Python call dispatch, reference-count traffic, allocations, and possible GC.
- Sparse extraction/build costs when bridging to a Python execution loop.
- Arena retention and occasional compaction proportional to stored values.
- Loss of GraphBLAS's vectorized/JIT arithmetic and most parallel speedups.

Benchmark separately: structural traversal, arena lookup/interning, Python
operator dispatch, result construction, and end-to-end sparse operations. Use
workloads with realistic object size and operator cost; a costly symbolic merge
may justify the overhead, while a numeric `lambda x, y: x + y` will not.

## Error handling and delayed expressions

Current GraphBLAS expressions may be delayed. Delaying an object operation
extends arena lifetimes and postpones Python exceptions, which is surprising and
complicates cleanup. The initial object domain should be eager: each operation
either returns a fully built object matrix/vector or raises at that call.

If delayed execution is later added, an expression must retain all input arenas
and operator objects, define where exceptions surface (`.new()` or assignment),
and atomically discard partially built output. It must not leave a backend
matrix containing handles whose arena has been released.

## Staged prototype plan

1. Add an isolated, clearly experimental `ObjectDomain` using a Python-managed
   sparse representation. Implement construction, extraction, elementwise
   binary operations, reduction with a monoid, and exact reference-lifetime
   tests.
2. Add arena handles and optional GraphBLAS storage for structural-only matrix
   operations. Verify that all wrapper copies retain source arenas.
3. Implement eager Python fallback kernels for elementwise operations and
   reductions. Define exception propagation and deterministic ordering.
4. Add `mxv`/`vxm`/`mxm` only after specifying semiring ordering and testing
   algebraic and exception behavior. Start single-threaded.
5. Investigate a backend-native implementation only if SuiteSparse or a new
   extension API offers per-value retain/release and GIL-safe callback hooks.

## Questions to settle before implementation

- Is process-lifetime or arena-lifetime retention acceptable for the target
  workload, or is compaction required in the initial release?
- Must object matrices participate in the existing delayed-expression API?
- Which operations need deterministic evaluation order, and is that a stronger
  promise than GraphBLAS normally makes?
- Are objects expected to be mutable? If so, what snapshot and cache semantics
  should matrix operations provide?
- Is serialization a requirement, and which codec/security model is acceptable?
- Is the goal a pedagogical/prototyping feature, or production support for
  expensive domain operations where Python callback overhead is tolerable?
