"""Python objects as values: the ``OBJECT`` dtype, its heap, and its collector.

GraphBLAS stores each value of an ``OBJECT`` Matrix, Vector or Scalar as an
8-byte handle, a UDT that SuiteSparse copies and drops as bytes. The heap here
maps handles to the objects. GraphBLAS has no hook that tells us when it copies
or drops a value, so the heap does not count references. Instead, between
operations, the live handles are exactly those stored in live OBJECT containers
(GraphBLAS copies handles but never invents one), and the collector marks those
and frees the rest (:func:`collect`).

A handle is ``generation << 40 | slot``. ``_slab[slot]`` holds the object, and
``_gen[slot]`` the generation of the handle that may name it. Freeing a slot
bumps its generation, so a stale or forged handle fails to resolve instead of
naming another object. Slot 0 is never used, so a zeroed value (an identity,
say) never resolves either.

Interning is by identity: storing one object twice gives one handle (``_by_id``).
So two values are the same object exactly when their handles are equal, and
GraphBLAS can compare them (``isequal``, iso values) without calling Python.

The core hooks call :func:`to_handles` where values come in, :func:`to_objects`
where they go out, :func:`track` when an OBJECT container is made, and
:func:`check_op` when an operator is typed for OBJECT.
"""

import threading
import weakref

import numpy as np

from . import ffi, lib

_has_iso = hasattr(lib, "GxB_Matrix_iso")

# The storage of an element: a record with one uint64 field. A record (not
# UINT64) makes OBJECT a UDT, so GraphBLAS refuses its built-in numeric ops,
# and python-graphblas's UDT machinery gives it ``first``, ``second`` and ``any``
# and a C definition for the JIT.
STORAGE = np.dtype([("pyobject_handle", "<u8")])

_SLOT_BITS = 40
_SLOT_MASK = np.uint64((1 << _SLOT_BITS) - 1)
_GEN_SHIFT = np.uint64(_SLOT_BITS)
_GEN_MAX = (1 << 24) - 1

# Collect when the allocations since the last collection exceed
# max(threshold, live handles after it), so a collection costs O(1) amortized
# per allocation. Tests lower it.
threshold = 100_000

# The ops that only select or copy values (or ignore them), by kind and name.
# Positional ops never reach ``check_op``: they are typed for UINT64.
_STRUCTURAL_OPS = {
    "BinaryOp": {"first", "second", "any", "pair", "oneb"},
    "UnaryOp": {"identity", "one"},
    "Monoid": {"any"},
}


class InvalidHandle(ValueError):
    """A value is not the handle of a live object: stale, forged, or zero."""


class _OpLock:
    """Shared for operations, exclusive for the collector.

    Invariants (each needed for a collection to be sound):

    1. A collection marks and sweeps only while it holds the lock exclusively,
       so no GraphBLAS call, intern or resolve runs during it, in any thread.
       GraphBLAS finishes pending work when the collector extracts values, and
       that must not race with a call on the same object.
    2. Every handle not stored in a tracked container is held by a tracked
       carrier (the array :func:`to_handles` returns) or is inside a running
       GraphBLAS call (a callback's result). The first kind is marked; the
       second is why invariant 1 waits for calls to finish.
    3. Shared holding is reentrant per thread: a callback inside a call interns.
    4. A thread never waits for the lock exclusively while any thread holds it
       shared. An automatic collection requested then is deferred, and the last
       shared holder to release runs it. So a callback on an OpenMP worker that
       interns never blocks behind a waiting collector.

    There is no writer preference: :func:`collect` waits until no call is
    running, and new shared holders do not wait for it.
    """

    def __init__(self):
        self._cond = threading.Condition(threading.Lock())
        self._active = 0  # threads holding the lock shared
        self._owner = None  # thread holding it exclusively
        self._local = threading.local()
        self.requested = False

    def acquire_shared(self):
        local = self._local
        depth = getattr(local, "depth", 0)
        if depth or self._owner == threading.get_ident():
            local.depth = depth + 1
            return
        with self._cond:
            while self._owner is not None:
                self._cond.wait()
            self._active += 1
        local.depth = 1

    def release_shared(self):
        local = self._local
        local.depth -= 1
        if local.depth or self._owner == threading.get_ident():
            return
        with self._cond:
            self._active -= 1
            run = self._active == 0 and self.requested
            self._cond.notify_all()
        if run:
            _collect(blocking=False)

    def acquire_exclusive(self, blocking):
        """Return True with the lock held exclusively, or False if busy and not blocking."""
        if getattr(self._local, "depth", 0):
            return False  # invariant 4: we hold it shared
        with self._cond:
            while self._active or self._owner is not None:
                if not blocking:
                    return False
                self._cond.wait()
            self._owner = threading.get_ident()
        return True

    def release_exclusive(self):
        with self._cond:
            self._owner = None
            self._cond.notify_all()

    def __enter__(self):
        self.acquire_shared()

    def __exit__(self, *args):
        self.release_shared()


_lock = _OpLock()
# Mutators hold ``_lock`` shared, so several may intern at once; this serializes
# allocation and the identity index among them. Reentrant: a finalizer may intern.
_heap_lock = threading.RLock()
# Set when the first OBJECT container is made; until then ``base.call`` skips the lock
active = False


class _Heap:
    def __init__(self, capacity=1024):
        self.slab = np.empty(capacity, dtype=object)
        self.gen = np.zeros(capacity, dtype=np.uint32)
        self.alive = np.zeros(capacity, dtype=bool)
        self.top = 1  # next never-used slot; slot 0 is never used
        self.free = []  # freed slots, reused before ``top``
        self.by_id = {}  # id(object) -> handle, for every object in the slab
        self.carriers = {}  # id(array) -> weakref of each live array of fresh handles
        self.containers = {}  # id(container) -> weakref of each OBJECT container
        self.roots = {}  # handle -> count of pins
        self.nlive = 0
        self.allocated = 0  # since the last collection
        self.live_after_gc = 0
        self.ncollections = 0

    def _grow(self, needed):
        capacity = self.slab.size
        while capacity < needed:
            capacity *= 2
        if capacity == self.slab.size:
            return
        for name in ("slab", "gen", "alive"):
            old = getattr(self, name)
            new = (
                np.empty(capacity, dtype=old.dtype)
                if name == "slab"
                else np.zeros_like(old, shape=capacity)
            )
            new[: old.size] = old
            setattr(self, name, new)

    def allocate(self, objects):
        """Store each of ``objects`` (distinct, none in the heap) and return their handles."""
        n = len(objects)
        nreuse = min(n, len(self.free))
        slots = np.empty(n, dtype=np.int64)
        if nreuse:
            slots[:nreuse] = self.free[-nreuse:]
            del self.free[-nreuse:]
        if nreuse < n:
            start = self.top
            self.top += n - nreuse
            self._grow(self.top)
            slots[nreuse:] = np.arange(start, self.top)
        self.slab[slots] = objects
        self.alive[slots] = True
        gen = self.gen[slots]
        gen[gen == 0] = 1
        self.gen[slots] = gen
        handles = (gen.astype(np.uint64) << _GEN_SHIFT) | slots.astype(np.uint64)
        self.by_id.update(zip(map(id, objects), handles.tolist(), strict=True))
        self.nlive += n
        self.allocated += n
        return handles

    def intern(self, objects):
        """Return the handles of an object ndarray, allocating for objects not yet stored."""
        by_id = self.by_id
        n = objects.size
        if n == 1:
            obj = objects[0]
            handle = by_id.get(id(obj))
            if handle is None:
                return self.allocate(objects)
            return np.array([handle], dtype=np.uint64)
        ids = np.fromiter(map(id, objects), dtype=np.intp, count=n)
        uniq, first, inverse = np.unique(ids, return_index=True, return_inverse=True)
        handles = np.fromiter(
            (by_id.get(i, 0) for i in uniq.tolist()), dtype=np.uint64, count=uniq.size
        )
        missing = handles == 0
        if missing.any():
            handles[missing] = self.allocate(objects[first[missing]])
        return handles[inverse]

    def slots_of(self, handles):
        """The slots of ``handles``; raise InvalidHandle unless each names a live object."""
        slots = (handles & _SLOT_MASK).astype(np.intp)
        gens = (handles >> _GEN_SHIFT).astype(np.uint32)
        top = self.top
        if slots.size and (
            slots.max() >= top or not self.alive[slots].all() or (self.gen[slots] != gens).any()
        ):
            bad = (slots >= top) | (slots == 0)
            ok = ~bad
            ok[ok] = self.alive[slots[ok]] & (self.gen[slots[ok]] == gens[ok])
            raise InvalidHandle(
                f"{int((~ok).sum())} of {handles.size} values are not handles of live "
                f"objects; the first is {int(handles[~ok][0]):#x}"
            )
        return slots

    def mark(self):
        """Return a boolean array over slots: True for each slot a live handle names."""
        marked = np.zeros(self.top, dtype=bool)
        for table in (self.containers, self.carriers):
            for ref in list(table.values()):
                obj = ref()
                if obj is None:
                    continue
                handles = obj if isinstance(obj, np.ndarray) else _stored_handles(obj)
                if handles is not None and handles.size:
                    slots = (handles & _SLOT_MASK).astype(np.intp)
                    marked[slots[slots < self.top]] = True
        if self.roots:
            handles = np.fromiter(self.roots, dtype=np.uint64, count=len(self.roots))
            marked[(handles & _SLOT_MASK).astype(np.intp)] = True
        return marked

    def sweep(self, marked):
        """Free each allocated slot not marked; return the freed objects (to drop later)."""
        dead = self.alive[: self.top] & ~marked
        slots = np.flatnonzero(dead)
        if not slots.size:
            return []
        objects = self.slab[slots].tolist()
        by_id = self.by_id
        for obj in objects:
            del by_id[id(obj)]
        self.slab[slots] = None
        self.alive[slots] = False
        gen = self.gen[slots]
        gen = np.where(gen >= _GEN_MAX, 1, gen + 1).astype(np.uint32)
        self.gen[slots] = gen
        self.free.extend(slots.tolist())
        self.nlive -= slots.size
        return objects


_heap = _Heap()


def _register(table, obj):
    """Add a weak reference to ``obj`` to ``table``; it removes itself when ``obj`` dies.

    The key is the id of the weak reference, not of ``obj``: the table keeps the
    reference alive, so its id is not reused while it is a key. Keyed by
    ``id(obj)``, a new object at a dead one's address could be registered just
    before the dead one's callback runs, which would then remove the new one.
    A free-threaded stress run hit that: a container went unmarked.
    """
    ref = weakref.ref(obj, lambda ref: table.pop(id(ref), None))
    table[id(ref)] = ref


def _stored_handles(container):
    """Return the handles stored in an OBJECT Matrix, Vector or Scalar as a uint64 array.

    The caller holds the lock (the collector holds it exclusively), so this
    calls GraphBLAS directly, not through ``call``. Extracting values finishes
    the container's pending work. An iso container gives one value.
    """
    if container.dtype is not OBJECT:
        return None
    if container._is_scalar:
        if container._is_cscalar:
            if container._empty:
                return None
            return np.frombuffer(ffi.buffer(container.gb_obj, 8), dtype=np.uint64).copy()
        nvals = ffi.new("GrB_Index*")
        lib.GrB_Scalar_nvals(nvals, container.gb_obj[0])
        if not nvals[0]:
            return None
        out = np.empty(1, dtype=np.uint64)
        lib.GrB_Scalar_extractElement_UDT(ffi.from_buffer(out), container.gb_obj[0])
        return out
    gb_obj = container.gb_obj[0]
    if gb_obj == ffi.NULL:
        return None
    nvals = ffi.new("GrB_Index*")
    is_iso = ffi.new("bool*")
    if container.ndim == 2:
        lib.GrB_Matrix_nvals(nvals, gb_obj)
        if _has_iso:  # not on the vanilla backend
            lib.GxB_Matrix_iso(is_iso, gb_obj)
    else:
        lib.GrB_Vector_nvals(nvals, gb_obj)
        if _has_iso:
            lib.GxB_Vector_iso(is_iso, gb_obj)
    n = nvals[0]
    if n == 0:
        return None
    out = np.empty(n, dtype=np.uint64)
    count = ffi.new("GrB_Index*", n)
    buf = ffi.from_buffer("void*", out)
    if container.ndim == 2:
        lib.GrB_Matrix_extractTuples_UDT(ffi.NULL, ffi.NULL, buf, count, gb_obj)
    else:
        lib.GrB_Vector_extractTuples_UDT(ffi.NULL, buf, count, gb_obj)
    return out[: count[0]] if not is_iso[0] else out[:1]


# Hooks


def track(container):
    """Register a new OBJECT Matrix, Vector or Scalar, so the collector marks its values."""
    global active
    active = True
    _register(_heap.containers, container)


def to_handles(values, *, shape=None):
    """Return ``values`` (objects) as an array of handles with dtype ``STORAGE``.

    The result is a carrier: the collector keeps its handles alive while it
    exists, so the caller may store them later. An array that already has
    dtype ``STORAGE`` (from ``ss.export``) is checked and passed through.
    """
    if isinstance(values, np.ndarray) and values.dtype == STORAGE:
        with _lock:
            _heap.slots_of(values.view(np.uint64).ravel())
        return values
    if isinstance(values, np.ndarray) and values.dtype == object:
        objects = values.ravel()
        shape = values.shape
    else:
        if not isinstance(values, (list, tuple)):
            values = list(values) if hasattr(values, "__iter__") else [values]
        objects = np.empty(len(values), dtype=object)
        objects[:] = _as_cells(values)
        shape = (len(values),)
    _maybe_collect()
    with _lock, _heap_lock:
        handles = _heap.intern(objects)
        # Register the array that owns the data: a view of a view has the owner as
        # its ``base``, not the view between, so only the owner lives as long as
        # every view of it does. ``intern`` returns a new array.
        _register(_heap.carriers, handles)
    return handles.view(STORAGE).reshape(shape)


def _as_cells(values):
    """A list numpy assigns one item per cell, even when the items are sequences."""
    if any(isinstance(v, (list, tuple, np.ndarray)) for v in values):
        cells = np.empty(len(values), dtype=object)
        for i, v in enumerate(values):
            cells[i] = v
        return cells
    return values


def to_objects(values):
    """Return an object array of the objects that ``values`` (dtype ``STORAGE``) name."""
    handles = np.ascontiguousarray(values).view(np.uint64)
    with _lock:
        heap = _heap
        slots = heap.slots_of(handles.ravel())
        return heap.slab.take(slots).reshape(handles.shape)


def element_bytes(value):
    """Return the handle of one object as a uint8 array (a carrier) for a Scalar's buffer."""
    return to_handles([value]).view(np.uint8)


def element_value(data):
    """Return the object a Scalar's 8-byte buffer names."""
    handles = np.frombuffer(ffi.buffer(data, 8), dtype=np.uint64)
    return to_objects(handles)[0]


def check_op(op, *dtypes):
    """Raise KeyError unless ``op`` only selects, copies or ignores OBJECT values.

    Arithmetic, comparison and value-dependent ops would read handles as
    numbers. An op made to call Python on objects sets ``_object_op``.
    """
    if not any(dtype is OBJECT for dtype in dtypes if dtype is not None):
        return
    if getattr(op, "_object_op", False):
        return
    allowed = _STRUCTURAL_OPS.get(type(op).__name__, ())
    if op.name in allowed and not getattr(op, "_anonymous", False):
        return
    raise KeyError(
        f"{op.name} does not work with OBJECT: GraphBLAS stores objects as handles, so only "
        "ops that select or copy values (first, second, any, pair, identity) and positional "
        "ops apply"
    )


def isequal(left, right):
    """Whether ``left`` and ``right`` hold the same objects (by identity) at the same indices.

    Interning is by identity, so equal handles are the same object, and this
    compares handles without resolving them.
    """
    if left.ndim == 0 and type(right) is not type(left):
        # A Scalar compares with an object; None means empty, as for other dtypes
        return left._is_empty if right is None else not left._is_empty and left.value is right
    if right.dtype is not OBJECT:
        return False
    if left.ndim == 0:
        return left._is_empty == right._is_empty and left.value is right.value
    left_indices, left_handles = _sorted_coo(left)
    right_indices, right_handles = _sorted_coo(right)
    return all(
        np.array_equal(x, y) for x, y in zip(left_indices, right_indices, strict=True)
    ) and np.array_equal(left_handles, right_handles)


def _sorted_coo(x):
    """Return ``(indices, handles)`` of a Matrix or Vector, sorted by index."""
    if getattr(x, "_is_transposed", False):
        x = x.new()
    *indices, _ = x.to_coo(values=False)
    with _lock:
        handles = _stored_handles(x)
    if handles is None:
        handles = np.empty(0, dtype=np.uint64)
    elif handles.size == 1 and indices[0].size > 1:
        handles = np.repeat(handles, indices[0].size)  # iso
    order = np.lexsort(indices[::-1])
    return [i[order] for i in indices], handles[order]


# The collector


def _maybe_collect():
    heap = _heap
    if heap.allocated > max(threshold, heap.live_after_gc):
        _collect(blocking=False)


def collect():
    """Free each object that no OBJECT Matrix, Vector or Scalar holds; return how many.

    Waits for GraphBLAS calls in other threads to finish. Inside a call (in a
    callback), it does nothing and returns 0: the last call to finish runs it.
    """
    return _collect(blocking=True)


def _collect(blocking):
    if not _lock.acquire_exclusive(blocking):
        _lock.requested = True
        return 0
    try:
        _lock.requested = False
        heap = _heap
        freed = heap.sweep(heap.mark())
        heap.allocated = 0
        heap.live_after_gc = heap.nlive
        heap.ncollections += 1
    finally:
        _lock.release_exclusive()
    # Drop the objects after releasing the lock: a finalizer may use graphblas
    n = len(freed)
    del freed
    return n


def pin(obj):
    """Keep ``obj`` alive as a root until :func:`unpin`; return its handle."""
    handle = int(to_handles([obj]).view(np.uint64)[0])
    with _lock, _heap_lock:
        _heap.roots[handle] = _heap.roots.get(handle, 0) + 1
    return handle


def unpin(handle):
    with _lock, _heap_lock:
        count = _heap.roots.pop(handle)
        if count > 1:
            _heap.roots[handle] = count - 1


def stats():
    """A dict of heap counters, for tests and benchmarks."""
    heap = _heap
    return {
        "live": heap.nlive,
        "capacity": heap.slab.size,
        "top": heap.top,
        "free": len(heap.free),
        "containers": len(heap.containers),
        "carriers": len(heap.carriers),
        "allocated_since_gc": heap.allocated,
        "collections": heap.ncollections,
    }


def _make_dtype():
    from ..exceptions import check_status_carg
    from . import _has_numba
    from .dtypes import DataType, _numba_type, _registry, _set_udt_jit_c_definition

    gb_obj = ffi.new("GrB_Type*")
    status = lib.GrB_Type_new(gb_obj, STORAGE.itemsize)
    check_status_carg(status, "Type", gb_obj[0])
    status = lib.GrB_Type_set_String(gb_obj[0], ffi.new("char[]", b"OBJECT"), lib.GrB_NAME)
    check_status_carg(status, "Type", gb_obj[0])
    numba_type = _numba_type(STORAGE) if _has_numba else None
    rv = DataType("OBJECT", gb_obj, None, "uint8_t[8]", numba_type, STORAGE)
    rv._is_object = True
    # ``object`` and ``np.dtype(object)`` are not keys: see ``dtypes.lookup_dtype``
    for key in (gb_obj, STORAGE, "OBJECT"):
        _registry[key] = rv
    _set_udt_jit_c_definition(rv)
    return rv


OBJECT = None  # set by ``dtypes._object_dtype`` on first use
