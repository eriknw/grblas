"""Operators that call Python functions on OBJECT values from inside GraphBLAS.

A user's function registered as a BinaryOp, UnaryOp, IndexUnaryOp or SelectOp
(``binary.register_anonymous(func)`` and the like) and typed for OBJECT does not
compile with Numba. It becomes a GraphBLAS op whose C function is a trampoline:
it reads the handles SuiteSparse passes, resolves them to objects, calls the
function, and writes the handle of the result (or a bool, for a SelectOp).
Monoids and semirings built from such a BinaryOp then work too, so ``mxm``,
``reduce`` and the rest run the Python function inside SuiteSparse's kernels.

The trampoline is C code (``_cshim``, about 50 ns a call) when a C compiler can
build it, and otherwise a cffi callback (here, about 30 times slower), which
needs neither a compiler nor Numba. With the GIL, or with a cffi callback, a
call that runs a Python op asks SuiteSparse for one thread (``run``).

A result that ``is`` one of the inputs reuses that input's handle, so ``min``
and the like return a stored object without allocating, and a monoid's
terminal value can be recognized. Other results get new handles: one object
may have several (``objects.isequal``).

SuiteSparse also calls an op on the iso values of empty iso inputs and
discards the result. Such a value may be handle 0 (never filled), and then
the function is not called.

Errors: the first exception in a call is kept, the function is not called
again in that call (each later element gets handle 0), and ``run`` raises it
after GraphBLAS returns. ``.new()`` then drops its result. ``C << expr`` has
already written into ``C``, which may hold some new values and some handles 0;
handle 0 fails when it is read (``objects.InvalidHandle``). A KeyboardInterrupt
in the function is handled the same way, so Ctrl-C stops the call.
"""

import sys
import threading

from ... import backend
from ...dtypes import BOOL
from ...exceptions import check_status_carg
from .. import ffi, lib, objects

_HAS_GIL = getattr(sys, "_is_gil_enabled", lambda: True)()

# Python-op calls run one at a time, so one error slot serves them. Reentrant:
# a Python op may itself run a GraphBLAS call that runs a Python op.
_run_lock = threading.RLock()
_error = None
_error_lock = threading.Lock()

# id of each typed op that calls Python -> its cffi callback (kept alive here)
_callbacks = {}

_SIGNATURES = {
    "UnaryOp": "GxB_unary_function",
    "BinaryOp": "GxB_binary_function",
    "IndexUnaryOp": "GxB_index_unary_function",
    "SelectOp": "GxB_index_unary_function",
}


class _Skip(Exception):
    """Handle 0, the iso value of an empty iso input: the call is skipped (``_cshim.skip``)."""


def _reader(dtype):
    """Return ``read(void_ptr) -> value`` for an input of type ``dtype``."""
    if dtype._is_object:
        resolve = objects.resolve_in_call

        def read(ptr):
            handle = ffi.cast("uint64_t*", ptr)[0]
            if handle == 0:
                raise _Skip
            return resolve(handle)

        return read
    if dtype._is_udt:
        raise KeyError(f"a Python op on OBJECT cannot also take {dtype}")
    c_ptr = f"{dtype.c_type}*"

    def read(ptr):
        return ffi.cast(c_ptr, ptr)[0]

    return read


def _make_body(kind, func, dtype, dtype2, ret_type):
    """Return the Python function the cffi callback runs."""
    read_x = _reader(dtype)
    is_bool = ret_type is BOOL
    intern = objects.intern_in_call
    in_x = dtype._is_object
    if kind == "BinaryOp":
        read_y = _reader(dtype2)
        in_y = dtype2._is_object

        def body(z, x, y):
            global _error
            if _error is not None:
                ffi.cast("uint64_t*", z)[0] = 0
                return
            try:
                a = read_x(x)
                b = read_y(y)
                r = func(a, b)
                # Reuse an input's handle when the result is that input
                if in_x and r is a:
                    h = ffi.cast("uint64_t*", x)[0]
                elif in_y and r is b:
                    h = ffi.cast("uint64_t*", y)[0]
                else:
                    h = intern(r)
                ffi.cast("uint64_t*", z)[0] = h
            except _Skip:
                ffi.cast("uint64_t*", z)[0] = 0
            except (Exception, KeyboardInterrupt) as exc:  # ``run`` raises it
                _error = exc
                ffi.cast("uint64_t*", z)[0] = 0

    elif kind == "UnaryOp":

        def body(z, x):
            global _error
            if _error is not None:
                ffi.cast("uint64_t*", z)[0] = 0
                return
            try:
                a = read_x(x)
                r = func(a)
                h = ffi.cast("uint64_t*", x)[0] if in_x and r is a else intern(r)
                ffi.cast("uint64_t*", z)[0] = h
            except _Skip:
                ffi.cast("uint64_t*", z)[0] = 0
            except (Exception, KeyboardInterrupt) as exc:  # ``run`` raises it
                _error = exc
                ffi.cast("uint64_t*", z)[0] = 0

    else:  # IndexUnaryOp, SelectOp: func(x, row, col, thunk)
        read_y = _reader(dtype2)

        if is_bool:

            def body(z, x, i, j, y):
                global _error
                if _error is not None:
                    ffi.cast("bool*", z)[0] = False
                    return
                try:
                    ffi.cast("bool*", z)[0] = bool(func(read_x(x), i, j, read_y(y)))
                except _Skip:
                    ffi.cast("bool*", z)[0] = False
                except (Exception, KeyboardInterrupt) as exc:  # ``run`` raises it
                    _error = exc
                    ffi.cast("bool*", z)[0] = False

        else:

            def body(z, x, i, j, y):
                global _error
                if _error is not None:
                    ffi.cast("uint64_t*", z)[0] = 0
                    return
                try:
                    a = read_x(x)
                    r = func(a, i, j, read_y(y))
                    h = ffi.cast("uint64_t*", x)[0] if in_x and r is a else intern(r)
                    ffi.cast("uint64_t*", z)[0] = h
                except _Skip:
                    ffi.cast("uint64_t*", z)[0] = 0
                except (Exception, KeyboardInterrupt) as exc:  # ``run`` raises it
                    _error = exc
                    ffi.cast("uint64_t*", z)[0] = 0

    return body


def compile_op(parent, dtype, dtype2, typed_cls):
    """Make, cache and return the typed op that calls ``parent.orig_func`` on OBJECT values.

    Called from an op class's ``_compile_udt`` when an input is OBJECT and the
    op is a user's function. Its result is OBJECT, or BOOL for a SelectOp. The
    op's C function is a C trampoline (``_cshim``) when one can be built, else a
    cffi callback.
    """
    from ..dtypes import _object_dtype
    from . import _cshim

    kind = typed_cls.opclass
    ret_type = BOOL if kind == "SelectOp" else _object_dtype()
    func = parent.orig_func
    if (made := _cshim.make(kind, func, dtype, dtype2, ret_type is BOOL)) is not None:
        fnptr, release = made
        entry = ("c", func)
    else:
        body = _make_body(kind, func, dtype, dtype2, ret_type)
        fnptr = ffi.callback(_SIGNATURES[kind], body)
        release = None
        entry = ("cffi", fnptr)
    c_typename = {"SelectOp": "GrB_IndexUnaryOp"}.get(kind, f"GrB_{kind}")
    gb_obj = ffi.new(f"{c_typename}*")
    new_func = getattr(lib, f"{c_typename}_new")
    label = c_typename.removeprefix("GrB_")
    if kind == "UnaryOp":
        status = new_func(gb_obj, fnptr, ret_type._carg, dtype._carg)
        op = typed_cls(parent, parent.name, dtype, ret_type, gb_obj[0])
        key = dtype
    else:
        status = new_func(gb_obj, fnptr, ret_type._carg, dtype._carg, dtype2._carg)
        op = typed_cls(parent, parent.name, dtype, ret_type, gb_obj[0], dtype2=dtype2)
        key = (dtype, dtype2)
    check_status_carg(status, label, gb_obj[0])
    _callbacks[id(op)] = entry
    if release is not None:
        _cshim.finalize_release(op, release)
    parent._udt_types[key] = ret_type
    parent._udt_ops[key] = op
    return op


def _set_error(exc):
    """Keep the first exception of the running call (C trampolines report here)."""
    global _error
    with _error_lock:
        if _error is None:
            _error = exc


def python_op_kind(args):
    """``"cffi"`` or ``"c"`` if a typed op in ``call``'s arguments (or in a monoid or
    semiring) calls Python, through a cffi callback or a C trampoline; else None.
    """
    kind = None
    for x in args:
        opclass = getattr(x, "opclass", None)
        if opclass is None:
            continue
        if opclass == "Monoid":
            ops = (x.binaryop,)
        elif opclass == "Semiring":
            ops = (x.binaryop, x.monoid.binaryop)
        else:
            ops = (x,)
        for op in ops:
            entry = _callbacks.get(id(op))
            if entry is not None:
                if entry[0] == "cffi":
                    return "cffi"
                kind = "c"
    return kind


def _one_thread(desc):
    """Return ``desc`` (a Descriptor or None) asking SuiteSparse for one thread.

    A descriptor made with other options (``axb_method``, ``nthreads``) is kept
    as given: its options cannot be read back to copy them.
    """
    from ..descriptor import BuiltinDescriptor, lookup

    if desc is not None and type(desc) is not BuiltinDescriptor:
        return desc
    if desc is None:
        return lookup(nthreads=1)
    return lookup(
        output_replace=desc.output_replace,
        mask_complement=desc.mask_complement,
        mask_structure=desc.mask_structure,
        transpose_first=desc.transpose_first,
        transpose_second=desc.transpose_second,
        nthreads=1,
    )


def run(call, cfunc_name, args, kind):
    """Run a GraphBLAS call whose op calls Python; raise the first error it hit.

    ``kind`` is from ``python_op_kind``. A cffi callback takes the GIL per call
    and does not scale, and with the GIL no trampoline does, so those calls ask
    SuiteSparse for one thread. A C trampoline on a free-threaded build uses
    SuiteSparse's thread count.

    Holds the collector's lock shared, so a collection asked for by the many
    allocations of a call waits until the call returns (``objects._OpLock``).
    """
    global _error
    from ..descriptor import Descriptor
    from . import _cshim

    if (
        (kind == "cffi" or _HAS_GIL)
        and backend == "suitesparse"
        and args
        and (args[-1] is None or isinstance(args[-1], Descriptor))
    ):
        args = [*args[:-1], _one_thread(args[-1])]
    with _run_lock:
        outer, _error = _error, None
        _cshim.reset_error()
        objects._lock.acquire_shared()
        try:
            rv = call(cfunc_name, args)
        finally:
            objects._lock.release_shared()
            error, _error = _error, outer
            _cshim.reset_error()
    if error is not None:
        raise error
    return rv
