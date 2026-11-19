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

### C. Integer handles with an external object domain

Store an integer handle, not a pointer, in GraphBLAS. A Python-owned domain maps
handles to strong references. GraphBLAS may freely copy handles because they
are plain scalar bytes. All dereferencing happens in Python-aware code.

This is the most promising representation, but it still has a key limitation:
GraphBLAS does not tell the domain when internal handle copies or discards occur.
Use handles as *interned immutable object identities*, not reference-counted
per-cell resources. The domain owns each object independently of the number of
matrix occurrences; matrix lifetime is tracked at a coarser level.

Possible lifetime policies:

- **Append-only domain:** handles keep objects alive until the domain is closed.
  Simple and correct, but unbounded for long-lived mutation-heavy workloads.
- **Container-rooted domains:** each Matrix or Vector wrapper retains its domain;
  native structural results retain the same domain. This preserves correctness
  but can retain objects no longer present in any container.
- **Explicit compaction:** rebuild a container and domain by extracting visible
  handles, remapping only reachable objects, and dropping the old domain. This is
  predictable but O(number of stored values) and cannot account for backend
  temporaries until an operation completes.

Handles are domain-local monotonic 64-bit integers. A wrapper always resolves a
handle through its own domain, and a domain must never recycle a handle while
any Matrix, Vector, view, or delayed expression could still contain it. Pointer
addresses should not be exposed as values.

## Required architecture: native SuiteSparse handle storage

The object containers must use real `python-graphblas` Matrix and Vector
instances backed by SuiteSparse. A Python coordinate dictionary, CSR/CSC clone,
or other mirrored sparse representation is explicitly out of scope. The native
GraphBLAS object is the sole authority for dimensions, structure, and stored
handle values.

Every stored value is a `UINT64` handle. `UINT64` is preferable to a custom UDT
for the prototype because it is already a native GraphBLAS type with direct
build, extraction, and assignment support. The handle is opaque despite its
numeric representation: object containers must not expose arithmetic, value
masks, numeric reductions, sorting by value, or other operations that interpret
the handle as an integer.

`ObjectDomain` is not a sparse container. It exists only to allocate handles,
map handles to strong Python references, validate handles, and keep those
references alive while a native Matrix or Vector may contain them. It stores no
coordinates, dimensions, masks, or sparse-format data.

The core ownership shape is:

```text
ObjectDomain
    handle -> strong Python object reference

ObjectMatrix                         ObjectVector
    domain: ObjectDomain                 domain: ObjectDomain
    _matrix: graphblas.Matrix            _vector: graphblas.Vector
             dtype UINT64                         dtype UINT64
             SuiteSparse-backed                   SuiteSparse-backed
```

`ObjectMatrix` and `ObjectVector` use composition rather than subclassing.
`python-graphblas` performs exact concrete-type checks in several paths, so a
subclass or transparent proxy is not reliably accepted by native operations.
Each object wrapper contains only its domain and native container; it must not
cache or mirror coordinates or values. An explicit `.native` property may
expose the underlying container for inspection and carefully documented
structural operations.

The wrappers should be thin:

- Shape, size, `nvals`, structure iteration, format information, and structural
  masks defer directly to the native Matrix or Vector.
- Bulk construction interns Python objects in the domain, creates a contiguous
  `numpy.uint64` handle array, and calls `Matrix.from_coo` or `Vector.from_coo`.
- Scalar reads call native `get` to extract a handle, then ask the domain to
  resolve it. Presence must be checked structurally so an absent entry remains
  distinct from a stored `None`.
- Scalar writes intern the object and assign the resulting integer handle to
  the native container. Deletion removes the native entry.
- Bulk export calls native `to_coo` and resolves its returned handle array. It
  never reads from a Python-side sparse representation.
- Native copies and structural results are wrapped with the same domain. They
  do not copy Python objects or create a second sparse representation.
- Methods that return a Matrix or Vector must either wrap the native result or
  be omitted from the initial API. Blind `__getattr__` forwarding is unsafe
  because it could leak unwrapped handle containers or numeric handle
  operations.

The initial implementation must require `graphblas.backend == "suitesparse"`.
It should not silently support another backend, `suitesparse-vanilla`, device
storage, or ownership-transferring export APIs.

Structural operations may execute in SuiteSparse when they copy, move, select,
or reshape opaque handles without applying numeric semantics. Safe candidates
include native `dup`, structural extraction and assignment, structural masks,
`split`, `concat`, and reshape. Each operation needs an explicit audit before it
is exposed. Value masks (`.V`), numeric select, sort, scan, reduction, and
arithmetic operators are not structural merely because their input dtype is
`UINT64`.

## Garbage collection and ownership

The domain holds normal strong Python references, so Python's cyclic GC can
collect cycles that are not reachable from a domain. However, an object may
refer to its wrapper or domain, and the wrapper retains the domain. Such a cycle is
collectable only if no external reference remains and the participating wrapper
objects are GC-tracked and do not use problematic finalizers.

Design rules:

- Domain entries own strong references. Native Matrix and Vector entries own
  only integer handles.
- Every object container holds a strong reference to its domain. Native copies
  and structural results are wrapped with that same domain.
- Handles are monotonically allocated within a domain and are never recycled.
  Handle `0` should be reserved as invalid so accidental default or value-mask
  behavior cannot silently identify a real object.
- Interning is identity-based, not equality-based. Equal but distinct Python
  objects may receive different handles. Repeated insertion of the same live
  object may reuse its handle if the domain maintains an identity index that
  does not weaken the strong-reference guarantee.
- Mutation does not immediately remove a domain entry. GraphBLAS does not
  report every internal handle copy or discard, so per-cell reference counting
  is not sound. Replacement, deletion, clear, and resize alter only native
  storage; objects remain retained until explicit compaction or domain teardown.
- Domain closure is allowed only when no live object container depends on it,
  or it invalidates all dependent wrappers before releasing references. A
  handle must never resolve through a closed domain.
- Do not use weak references as the domain's owning entries. A stored handle must
  always resolve while its domain remains valid.
- Compaction discovers visible handles by extracting them from registered
  native containers. It must not construct a Python sparse mirror. The simplest
  safe first form compacts a newly copied container into a new domain and native
  `UINT64` container, leaving existing wrappers unchanged.

Python finalizers should not call GraphBLAS while interpreter shutdown is in
progress. Wrapper cleanup should be idempotent and tolerate a partially torn
down interpreter/backend.

## Matrix representation and interchange

Internally, SuiteSparse owns the sparse matrix or vector and sees only opaque
`UINT64` handles. The Python wrapper preserves the domain reference and delegates
all structural state to that native container. Iteration, scalar extraction,
`to_coo`, and repr obtain handles from the native object and dereference them
through the domain. Bulk construction interns objects in one pass and passes a
contiguous handle array to native `from_coo`, avoiding per-entry GraphBLAS calls.

Important semantic choices:

- An absent entry is still GraphBLAS absence, not Python `None`.
- `None` is a valid stored object and must have its own handle.
- Equality of handles means object identity/interning identity, not necessarily
  `==`. Any value equality must be an explicit Python binary operator.
- Duplicated input coordinates need a specified Python duplicate reducer; no
  implicit backend `second` callback should manipulate handles unless it is
  purely selecting a handle and the output domain retains both inputs.
- A wrapper may accept an existing native Matrix or Vector only when it is
  SuiteSparse-backed, has dtype `UINT64`, and every stored handle belongs to the
  supplied domain. Validation requires native handle extraction.
- The native object's public numeric API is not automatically safe. Exposing
  `.native` is an escape hatch whose caller must preserve opaque-handle
  semantics; replacing a handle with an arbitrary integer can make subsequent
  object extraction fail.
- `ss.export(give_ownership=True)`, unpacking with ownership transfer, or any
  operation that invalidates the native object is excluded from wrapped use.
- Serialization cannot write live objects as raw handles. Require a codec
  (such as pickle with explicit security warnings, or user `encode`/`decode`)
  and reconstruct a fresh domain on load.
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

- Input and output are Python references obtained from domain handles; output is
  interned into the destination domain.
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
unspecified order. An eager extract/callback/build implementation can preserve a
documented deterministic order, but then differs from ordinary GraphBLAS
scheduling and should say so.
Do not expose a generic object semiring as GraphBLAS `mxm` until its execution
and exception model is defined end-to-end.

## GIL, threads, and cost model

Every Python callback requires the GIL. If SuiteSparse invokes it from OpenMP
workers, each invocation would acquire the GIL and serialize bytecode execution;
it may also be illegal if the backend has not initialized thread state for those
workers. Acquiring and releasing the GIL once per scalar operation can dominate
the work by orders of magnitude, even for inexpensive functions.

The first implementation should never invoke Python from backend worker threads.
For object-valued operations, extract coordinates and handles from the native
container, execute callbacks in one Python thread with the GIL held, intern the
outputs, and build a new native container. There is no Python sparse fallback
or persistent coordinate mirror. Parallelism may still be useful for native
structural operations that do not interpret handles.

Expected costs include:

- One or more handle-to-object lookups and output interning per callback.
- Python call dispatch, reference-count traffic, allocations, and possible GC.
- Sparse extraction/build costs when bridging to a Python execution loop.
- Domain retention and occasional compaction proportional to stored values.
- Loss of GraphBLAS's vectorized/JIT arithmetic and most parallel speedups.

Benchmark separately: structural traversal, domain lookup/interning, Python
operator dispatch, result construction, and end-to-end sparse operations. Use
workloads with realistic object size and operator cost; a costly symbolic merge
may justify the overhead, while a numeric `lambda x, y: x + y` will not.

## Error handling and delayed expressions

Current GraphBLAS expressions may be delayed. Delaying an object operation
extends domain lifetimes and postpones Python exceptions, which is surprising and
complicates cleanup. The initial object domain should be eager: each operation
either returns a fully built object matrix/vector or raises at that call.

If delayed execution is later added, an expression must retain all input domains
and operator objects, define where exceptions surface (`.new()` or assignment),
and atomically discard partially built output. It must not leave a backend
matrix containing handles whose domain has been released.

## Staged prototype plan

### Stage one: native handle storage

Add an isolated experimental package with no changes to existing
`python-graphblas` code. Implement:

1. `ObjectDomain` with monotonic nonzero `UINT64` handle allocation, strong
   object retention, identity-based interning, handle validation, and lookup.
   It must contain no sparse coordinates or dimensions.
2. `ObjectMatrix` containing exactly an `ObjectDomain` and a native
   SuiteSparse-backed `graphblas.Matrix[UINT64]`.
3. `ObjectVector` containing exactly an `ObjectDomain` and a native
   SuiteSparse-backed `graphblas.Vector[UINT64]`.
4. Bulk `from_coo` construction that interns values and builds the native
   container. Reject duplicate coordinates initially; add Python duplicate
   reduction only after reducing objects before native construction.
5. Scalar presence, extraction, assignment, and deletion implemented against
   the native container. Absent entries and stored `None` must remain distinct.
6. Bulk `to_coo` that obtains coordinates and handles from the native container
   and resolves objects through the domain.
7. Native `dup` wrapped with the same domain, plus direct delegation of shape,
   size, `nvals`, structural mask, and SuiteSparse format introspection.
8. Tests that assert the native object is authoritative: mutate it through the
   documented native escape hatch and observe wrapper extraction, inspect dtype
   and backend, verify native `nvals` and shape, and ensure no Python coordinate
   store exists.
9. Lifetime tests proving objects survive while the domain is retained, native
   duplicates share the domain, removed values remain retained until
   compaction/teardown, invalid handles fail clearly, and wrapper/domain cycles
   are collectable.

Stage one does not implement Python elementwise operators, monoids, reductions,
or matrix multiplication. It first proves the central storage and ownership
contract.

### Stage two: audited native structural operations

Expose a small allowlist of SuiteSparse operations that preserve opaque handles:
native duplication, structural extraction/assignment, reshape, split, and
concat. Wrap every Matrix or Vector result with the same domain. Test that all
copies and views retain the domain and that no operation numerically interprets
the handles. Do not use broad attribute forwarding.

### Stage three: reclamation

Add explicit compaction by extracting visible handles from native containers,
building a new domain containing only reachable objects, remapping handles, and
building new native containers. Prefer a non-mutating API returning new wrapped
objects so old native containers and handles remain valid. Define domain close
semantics only after container registration and delayed native work are fully
understood.

### Stage four: eager Python value operations

Implement elementwise operations and reductions by extracting native COO data,
resolving handles, invoking Python callbacks under the GIL, interning outputs,
and building a native result. Define deterministic ordering and atomic exception
behavior. The native Matrix or Vector remains the only sparse representation
before and after the operation.

### Stage five: semirings and backend hooks

Add `mxv`/`vxm`/`mxm` only after specifying semiring ordering and exception
behavior. Investigate backend-native Python callbacks only if SuiteSparse or a
new extension API offers per-value retain/release and GIL-safe callback hooks.

## Questions to settle before implementation

- Is process-lifetime or domain-lifetime retention acceptable for the target
  workload, or is compaction required in the initial release?
- Must object matrices participate in the existing delayed-expression API?
- Which operations need deterministic evaluation order, and is that a stronger
  promise than GraphBLAS normally makes?
- Are objects expected to be mutable? If so, what snapshot and cache semantics
  should matrix operations provide?
- Is serialization a requirement, and which codec/security model is acceptable?
- Is the goal a pedagogical/prototyping feature, or production support for
  expensive domain operations where Python callback overhead is tolerable?
