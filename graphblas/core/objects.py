"""Python objects as values: the ``OBJECT`` dtype, its heap, and its collector.

GraphBLAS stores each value of an ``OBJECT`` Matrix, Vector or Scalar as an
8-byte handle, a UDT that SuiteSparse copies and drops as bytes. The heap here
maps handles to the objects. GraphBLAS has no hook that tells us when it copies
or drops a value, so the heap does not count references. Instead, between
operations, the live handles are exactly those stored in live OBJECT containers
(GraphBLAS copies handles but never invents one), and the collector marks those
and frees the rest (:func:`collect`).

A handle is ``generation << 40 | slot``. The heap's slot holds the object and the
generation of the handle that may name it (``_Heap``). Freeing a slot bumps its
generation, so a stale or forged handle fails to resolve instead
of naming another object (until a slot is reused 2**24 times). Slot 0 is never
used, so a zeroed value (an identity, say) never resolves either.

One value (a Scalar's, say) is interned by identity: storing one object twice
gives one handle (``_heap.by_id``). An array of values (``from_coo``) and the
results of the C trampolines (``operator._cshim``) get a handle per occurrence,
except that a result that is one of the inputs keeps its handle. So equal
handles are the same object, but one object may have several handles:
``isequal`` compares handles and resolves only the pairs that differ. A slot's
alive mark is 2 when it is in ``_heap.by_id``, 1 otherwise.

The core hooks call :func:`to_handles` where values come in, :func:`to_objects`
where they go out, :func:`track` when an OBJECT container is made, and
:func:`check_op` when an operator is typed for OBJECT.
"""

import itertools
import threading
import weakref
from threading import get_ident as _get_ident

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

# Collect when the allocations since the last collection exceed this threshold,
# so a collection costs O(1) amortized per allocation. Tests lower it.
threshold = 100_000
# ... or GC_FACTOR times the live handles after the last collection, so a heap
# that holds much more than the threshold collects less often. 2 measured best on
# the demos (walks, BFS, group-by, shortest paths): fewer sweeps of a larger heap.
GC_FACTOR = 2

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
       so no intern, resolve or GraphBLAS call on an OBJECT container (as
       output, input or mask; ``touches_object``) runs during it, in any thread.
       GraphBLAS finishes pending work when the collector extracts values, and
       that must not race with a call on the same object.
    2. Every handle not stored in a tracked container is held by a tracked
       carrier (the array :func:`to_handles` returns, or a view of it) or is
       inside a running GraphBLAS call (a callback's result). The first kind is
       marked; the second is why invariant 1 waits for calls to finish. Code
       that stores a carrier's handles keeps the carrier until the container
       shows them (``Scalar.value`` keeps it until ``_empty`` is False).
       Direct ``lib`` calls on an OBJECT container that may finish pending
       work take the lock too (``_nvals``).
    3. Shared holding is reentrant per thread: a callback inside a call interns.
    4. A thread never waits for the lock exclusively while any thread holds it
       shared. An automatic collection requested then is deferred, and the last
       shared holder to release runs it. So a callback on an OpenMP worker that
       interns never blocks behind a waiting collector.

    There is no writer preference: :func:`collect` waits until no call is
    running, and new shared holders do not wait for it.
    """

    def __init__(self):
        self._mutex = threading.Lock()
        self._cond = threading.Condition(self._mutex)
        self._active = 0  # threads holding the lock shared
        self._owner = None  # thread holding it exclusively
        self._waiting = 0  # threads waiting for it exclusively
        self._local = threading.local()
        self.requested = False

    # Every GraphBLAS call takes the lock shared once OBJECT is in use, so the shared
    # paths use the mutex directly and touch the condition only when a thread waits.

    def acquire_shared(self):
        local = self._local
        depth = getattr(local, "depth", 0)
        if depth or self._owner == _get_ident():
            local.depth = depth + 1
            return
        mutex = self._mutex
        mutex.acquire()
        if self._owner is None:
            self._active += 1
            mutex.release()
        else:
            mutex.release()
            with self._cond:
                while self._owner is not None:
                    self._cond.wait()
                self._active += 1
        local.depth = 1

    def release_shared(self):
        local = self._local
        local.depth -= 1
        if local.depth or self._owner == _get_ident():
            return
        mutex = self._mutex
        mutex.acquire()
        self._active -= 1
        idle = self._active == 0
        run = idle and (self.requested or _heap.state[S_GC])
        notify = idle and self._waiting
        mutex.release()
        if notify:
            with self._cond:
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
                self._waiting += 1
                try:
                    self._cond.wait()
                finally:
                    self._waiting -= 1
            self._owner = _get_ident()
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


# The heap is a list of chunks of _CHUNK slots that never move once made, so C
# code (``operator._cshim``) can index them while Python adds more. Each chunk has
# an object array, a generation array and an alive array; slot ``s`` is element
# ``s & _OFF_MASK`` of chunk ``s >> _CHUNK_BITS``.
_CHUNK_BITS = 16
_CHUNK = 1 << _CHUNK_BITS
_OFF_MASK = _CHUNK - 1
MAX_CHUNKS = 1 << 16  # 2**32 slots

# Fields of ``_Heap.state``, which C code reads and updates with atomics
S_TOP, S_NFREE, S_ALLOCATED, S_NLIVE, S_EPOCH, S_GC = range(6)


class _Heap:
    """The slots, their generations, and the counters, shared with ``operator._cshim``.

    Slots are claimed (``claim``) from a stack of free slots, then from ``top``.
    The free stack is rebuilt by each collection from the alive arrays, so a slot
    claimed but never filled (the tail of a C thread's block) is free again then.
    Each collection also bumps the epoch, which retires those blocks.
    """

    def __init__(self):
        self.objs = []  # per chunk, object arrays
        self.gens = []  # per chunk, uint32
        self.alives = []  # per chunk, uint8 (1 while the slot holds an object)
        self.state = np.zeros(8, dtype=np.uint64)
        self.state[S_TOP] = 1  # slot 0 is never used
        self.free = np.zeros(_CHUNK, dtype=np.int64)
        self.by_id = {}  # id(object) -> a handle of it (C-made handles are not here)
        # id(weakref) -> weakref (``_register``) of each live array of fresh handles,
        # and of each OBJECT container
        self.carriers = {}
        self.containers = {}
        self.roots = {}  # handle -> count of pins
        self.live_after_gc = 0
        self.ncollections = 0
        self.c = None  # the C directory (``operator._cshim``), once loaded
        self._chunk_lock = threading.RLock()
        self.add_chunks(1)

    @property
    def top(self):
        return int(self.state[S_TOP])

    @property
    def nlive(self):
        return int(self.state[S_NLIVE])

    @property
    def allocated(self):
        return int(self.state[S_ALLOCATED])

    def add_chunks(self, upto):
        """Make chunks until slot ``upto - 1`` exists; C publishes each new one."""
        with self._chunk_lock:
            while len(self.objs) * _CHUNK < upto:
                if len(self.objs) >= MAX_CHUNKS:
                    raise MemoryError("the OBJECT heap is full")
                objs = np.empty(_CHUNK, dtype=object)
                gens = np.zeros(_CHUNK, dtype=np.uint32)
                alives = np.zeros(_CHUNK, dtype=np.uint8)
                self.objs.append(objs)
                self.gens.append(gens)
                self.alives.append(alives)
                if self.c is not None:
                    self.c.publish(len(self.objs) - 1, objs, gens, alives)

    def claim(self, n):
        """Return ``n`` slots (int64) to fill, from the free stack first."""
        if self.c is not None:
            slots = self.c.claim(n)
        else:
            state = self.state
            nfree = int(state[S_NFREE])
            take = min(n, nfree)
            slots = np.empty(n, dtype=np.int64)
            slots[:take] = self.free[nfree - take : nfree]
            state[S_NFREE] = nfree - take
            top = int(state[S_TOP])
            slots[take:] = np.arange(top, top + n - take)
            state[S_TOP] = top + n - take
        if n:
            self.add_chunks(int(slots.max()) + 1)
        return slots

    def _count(self, n):
        if self.c is not None:
            self.c.count(n)
        else:
            self.state[S_NLIVE] += np.uint64(n)
            self.state[S_ALLOCATED] += np.uint64(n)

    @staticmethod
    def _groups(chunk):
        """Yield ``(chunk, index)`` for each chunk among ``chunk``; index is a slice if sorted.

        The heap's own slots (claims, sweeps) come sorted, so no mask or sort is needed
        for them; a boolean mask per chunk was most of a collection's time.
        """
        n = chunk.size
        steps = np.diff(chunk)
        order = None
        if (steps < 0).any():
            order = np.argsort(chunk, kind="stable")
            chunk = chunk[order]
            steps = np.diff(chunk)
        bounds = [0, *(np.flatnonzero(steps) + 1).tolist(), n]
        for start, stop in itertools.pairwise(bounds):
            index = slice(start, stop) if order is None else order[start:stop]
            yield int(chunk[start]), index

    def gather(self, arrays, slots):
        """Return ``arrays[chunk][offset]`` for each slot, as one array."""
        chunk = slots >> _CHUNK_BITS
        off = slots & _OFF_MASK
        if not slots.size or chunk.min() == chunk.max():
            return arrays[int(chunk[0]) if slots.size else 0][off]
        out = np.empty(slots.size, dtype=arrays[0].dtype)
        for c, index in self._groups(chunk):
            out[index] = arrays[c][off[index]]
        return out

    def scatter(self, arrays, slots, values):
        """Set ``arrays[chunk][offset]`` for each slot."""
        chunk = slots >> _CHUNK_BITS
        off = slots & _OFF_MASK
        if not slots.size:
            return
        if chunk.min() == chunk.max():
            arrays[int(chunk[0])][off] = values
            return
        scalar = np.ndim(values) == 0
        for c, index in self._groups(chunk):
            arrays[c][off[index]] = values if scalar else values[index]

    def allocate(self, objects, by_id=True):
        """Store each of ``objects`` (distinct, none in the heap) and return their handles."""
        n = len(objects)
        slots = self.claim(n)
        self.scatter(self.objs, slots, objects)
        gen = self.gather(self.gens, slots)
        gen[gen == 0] = 1
        self.scatter(self.gens, slots, gen)
        handles = (gen.astype(np.uint64) << _GEN_SHIFT) | slots.astype(np.uint64)
        if by_id:
            self.scatter(self.alives, slots, 2)
            self.by_id.update(zip(map(id, objects), handles.tolist(), strict=True))
        else:
            self.scatter(self.alives, slots, 1)
        self._count(n)
        return handles

    def claim_one(self):
        """``claim`` for one slot, without numpy temporaries (the cffi trampolines' path)."""
        if self.c is not None:
            slot = self.c.claim_one()
        else:
            state = self.state
            nfree = int(state[S_NFREE])
            if nfree:
                slot = int(self.free[nfree - 1])
                state[S_NFREE] = nfree - 1
            else:
                slot = int(state[S_TOP])
                state[S_TOP] = slot + 1
        if slot >= len(self.objs) * _CHUNK:
            self.add_chunks(slot + 1)
        return slot

    def allocate_one(self, obj):
        """``allocate`` for one object."""
        slot = self.claim_one()
        c, off = slot >> _CHUNK_BITS, slot & _OFF_MASK
        self.objs[c][off] = obj
        gens = self.gens[c]
        gen = gens.item(off) or 1
        gens[off] = gen
        self.alives[c][off] = 2
        handle = gen << _SLOT_BITS | slot
        self.by_id[id(obj)] = handle
        self._count(1)
        return handle

    def intern(self, objects):
        """Return handles for an object ndarray: one value by identity, more per occurrence."""
        by_id = self.by_id
        n = objects.size
        if n == 1:
            obj = objects[0]
            handle = by_id.get(id(obj))
            if handle is None:
                return self.allocate(objects)
            return np.array([handle], dtype=np.uint64)
        # More than one: a handle per occurrence, a slice write. Interning by
        # identity costs a dict lookup per value, about 200 ns, and the demos
        # store many values once each (``isequal`` compares mismatched handles).
        return self.allocate(objects, by_id=False)

    def slots_of(self, handles):
        """The slots of ``handles``; raise InvalidHandle unless each names a live object."""
        slots = (handles & _SLOT_MASK).astype(np.intp)
        gens = (handles >> _GEN_SHIFT).astype(np.uint32)
        if not slots.size:
            return slots
        # A slot below top may be in a chunk not made yet (``claim``)
        ok = (slots < min(self.top, len(self.objs) * _CHUNK)) & (slots != 0)
        if ok.all():
            ok = (self.gather(self.gens, slots) == gens) & (self.gather(self.alives, slots) != 0)
            if ok.all():
                return slots
        else:
            ok[ok] = (self.gather(self.gens, slots[ok]) == gens[ok]) & (
                self.gather(self.alives, slots[ok]) != 0
            )
        raise InvalidHandle(
            f"{int((~ok).sum())} of {handles.size} values are not handles of live "
            f"objects; the first is {int(handles[~ok][0]):#x}"
        )

    def mark(self):
        """Return a boolean array over slots: True for each slot a live handle names."""
        top = self.top
        marked = np.zeros(top, dtype=bool)
        for table in (self.containers, self.carriers):
            for ref in list(table.values()):
                obj = ref()
                if obj is None:
                    continue
                handles = obj if isinstance(obj, np.ndarray) else _stored_handles(obj)
                if handles is not None and handles.size:
                    slots = (handles & _SLOT_MASK).astype(np.intp)
                    marked[slots[slots < top]] = True
        if self.roots:
            handles = np.fromiter(self.roots, dtype=np.uint64, count=len(self.roots))
            marked[(handles & _SLOT_MASK).astype(np.intp)] = True
        return marked

    def sweep(self, marked):
        """Free each filled slot not marked; return the freed objects (to drop later).

        Runs with the lock held exclusively: no C code is running.
        """
        top = len(marked)
        marks = np.concatenate(self.alives)[:top]
        alive = marks != 0
        slots = np.flatnonzero(alive & ~marked)
        objects = []
        if slots.size:
            objects = self.gather(self.objs, slots)
            gen = self.gather(self.gens, slots)
            handles = (gen.astype(np.uint64) << _GEN_SHIFT) | slots.astype(np.uint64)
            by_id = self.by_id
            # Only slots marked 2 were interned by identity (``by_id``); checking every
            # slot was half a collection's time
            sel = marks[slots] == 2
            for obj, handle in zip(objects[sel].tolist(), handles[sel].tolist(), strict=True):
                if by_id.get(id(obj)) == handle:
                    del by_id[id(obj)]
            self.scatter(self.objs, slots, None)
            self.scatter(self.alives, slots, 0)
            self.scatter(self.gens, slots, np.where(gen >= _GEN_MAX, 1, gen + 1).astype(np.uint32))
            alive[slots] = False
        # Every empty slot below top is free, including slots claimed and never filled
        alive[0] = True
        free = np.flatnonzero(~alive)
        if free.size > self.free.size:
            self.free = np.zeros(max(free.size, 2 * self.free.size), dtype=np.int64)
        self.free[: free.size] = free
        state = self.state
        state[S_NFREE] = free.size
        # C counts the slots it claims, not those it fills (``_cshim.put_obj``)
        state[S_NLIVE] = top - 1 - free.size
        state[S_ALLOCATED] = 0
        state[S_EPOCH] += np.uint64(1)
        state[S_GC] = 0
        if self.c is not None:
            self.c.set_free(self.free)
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
        if not is_iso[0]:
            return None
        # SuiteSparse calls an op on the iso values of two iso inputs even when they
        # have no entries (an iso 1x0 Matrix, say), so the iso value of an empty
        # container is live too. Only an export reads it.
        values = container.ss.export(raw=True)["values"]
        return values.view(np.uint64)[:1].copy() if values.size else None
    out = np.empty(n, dtype=np.uint64)
    count = ffi.new("GrB_Index*", n)
    buf = ffi.from_buffer("void*", out)
    if container.ndim == 2:
        lib.GrB_Matrix_extractTuples_UDT(ffi.NULL, ffi.NULL, buf, count, gb_obj)
    else:
        lib.GrB_Vector_extractTuples_UDT(ffi.NULL, buf, count, gb_obj)
    return out[: count[0]] if not is_iso[0] else out[:1]


# Hooks


def touches_object(args):
    """Whether any of ``call``'s arguments is, or wraps, an OBJECT container.

    A call on other containers cannot race with a collection, which reads only
    OBJECT containers, so it skips the lock. Dispatch is on the exact type, as
    ``getattr`` with a default costs too much on the many arguments without one.
    """
    for x in args:
        kind = _arg_kinds.get(type(x))
        if kind is None:
            continue
        if kind == 1:  # _Pointer
            x = x.val
            if type(x) not in _arg_kinds:
                continue
        elif kind == 2:  # Mask
            x = x.parent
        if x.dtype._is_object:
            return True
    return False


# type -> 0 (has ``dtype``), 1 (``_Pointer``) or 2 (mask), filled by ``track``
_arg_kinds = {}


def _fill_arg_kinds():
    from .mask import Mask
    from .matrix import Matrix, TransposedMatrix
    from .scalar import Scalar
    from .utils import _CArray, _Pointer
    from .vector import Vector

    kinds = dict.fromkeys((Matrix, TransposedMatrix, Vector, Scalar, _CArray), 0)
    kinds[_Pointer] = 1
    kinds.update((cls, 2) for cls in Mask.__subclasses__())
    _arg_kinds.update(kinds)


def track(container):
    """Register a new OBJECT Matrix, Vector or Scalar, so the collector marks its values."""
    global active
    if not active:
        _fill_arg_kinds()
        active = True
    _register(_heap.containers, container)


def to_handles(values, *, shape=None):
    """Return ``values`` (objects) as an array of handles with dtype ``STORAGE``.

    The result is a carrier: the collector keeps its handles alive while it,
    or any view of it, exists, so the caller may store them later. An array
    that already has dtype ``STORAGE`` (from ``ss.export``) passes through.
    """
    if isinstance(values, np.ndarray) and values.dtype == STORAGE:
        # Not checked: the values of a bitmap export include garbage where there is
        # no entry. A stale handle fails when it is resolved.
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
        return heap.gather(heap.objs, slots).reshape(handles.shape)


def resolve_in_call(handle):
    """Return the object of one handle (an int), in a call that holds the lock shared.

    The cffi trampolines of ``operator.pyobject`` use it: no collection runs
    during the call, so this takes no lock. ``item`` is a few times faster than
    indexing.
    """
    heap = _heap
    slot = handle & 0xFFFFFFFFFF
    c, off = slot >> _CHUNK_BITS, slot & _OFF_MASK
    if (
        slot
        and c < len(heap.gens)
        and heap.gens[c].item(off) == handle >> _SLOT_BITS
        and heap.alives[c].item(off)
    ):
        return heap.objs[c].item(off)
    raise InvalidHandle(f"{handle:#x} is not the handle of a live object")


def intern_in_call(obj):
    """Return the handle of ``obj``, in a call that holds the lock shared.

    The handle goes straight into GraphBLAS's output, so it needs no carrier: no
    collection runs until the call returns, and by then the output container
    holds it. A collection due now is deferred to then.
    """
    heap = _heap
    with _heap_lock:
        handle = heap.by_id.get(id(obj))
        if handle is None:
            handle = heap.allocate_one(obj)
            if heap.allocated > max(threshold, GC_FACTOR * heap.live_after_gc):
                _lock.requested = True
    return handle


def element_bytes(value):
    """Return the handle of one object as a uint8 array (a carrier) for a Scalar's buffer."""
    return to_handles([value]).view(np.uint8)


def element_value(data):
    """Return the object a Scalar's 8-byte buffer names."""
    handles = np.frombuffer(ffi.buffer(data, 8), dtype=np.uint64)
    return to_objects(handles)[0]


def check_op(op, *dtypes, typed_cls=None):
    """Return None, a typed op that calls Python, or raise KeyError, for ``op`` on OBJECT.

    None (the usual path) for an op on no OBJECT input, or one that only
    selects, copies or ignores values. A user's function (``orig_func``) on an
    OBJECT input becomes an op that calls it on the objects
    (``operator.pyobject``); ``typed_cls`` is the class of that op. Any other op,
    arithmetic or comparison, would read handles as numbers: KeyError.
    """
    if not any(dtype is OBJECT for dtype in dtypes if dtype is not None):
        return None
    allowed = _STRUCTURAL_OPS.get(type(op).__name__, ())
    if op.name in allowed and not getattr(op, "_anonymous", False):
        return None
    if typed_cls is not None and getattr(op, "orig_func", None) is not None:
        from .operator.pyobject import compile_op

        return compile_op(op, dtypes[0], dtypes[1] if len(dtypes) > 1 else None, typed_cls)
    raise KeyError(
        f"{op.name} does not work with OBJECT: GraphBLAS stores objects as handles, so only "
        "ops that select or copy values (first, second, any, pair, identity), positional "
        "ops and your own Python functions apply"
    )


def isequal(left, right):
    """Whether ``left`` and ``right`` hold the same objects (by identity) at the same indices.

    Equal handles are the same object, so only pairs of different handles are
    resolved and compared with ``is``.
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
    if not all(np.array_equal(x, y) for x, y in zip(left_indices, right_indices, strict=True)):
        return False
    differ = left_handles != right_handles
    if not differ.any():
        return True
    a = to_objects(left_handles[differ].view(STORAGE))
    b = to_objects(right_handles[differ].view(STORAGE))
    return all(x is y for x, y in zip(a.tolist(), b.tolist(), strict=True))


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
    if heap.allocated > max(threshold, GC_FACTOR * heap.live_after_gc):
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
        "capacity": len(heap.objs) * _CHUNK,
        "top": heap.top,
        "free": int(heap.state[S_NFREE]),
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
