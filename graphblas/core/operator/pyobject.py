"""Operators that call Python functions on OBJECT values from inside GraphBLAS.

A user's function registered as a BinaryOp, UnaryOp, IndexUnaryOp or SelectOp
(``binary.register_anonymous(func)`` and the like) and typed for OBJECT does not
compile with Numba. It becomes a GraphBLAS op whose C function is a trampoline:
it reads the handles SuiteSparse passes, resolves them to objects, calls the
function, and writes the handle of the result (or a bool, for a SelectOp).
Monoids and semirings built from such a BinaryOp then work too, so ``mxm``,
``reduce`` and the rest run the Python function inside SuiteSparse's kernels.

The trampoline is C code (``_cshim``, about 50 ns a call) when a C compiler can
build it, and otherwise a cffi callback (10 to 30 times slower), which
needs neither a compiler nor Numba. With the GIL, or with a cffi callback, a
call that runs a Python op asks SuiteSparse for one thread (``run``).

A result that ``is`` one of the inputs reuses that input's handle, so ``min``
and the like return a stored object without allocating, and a monoid's
terminal value can be recognized. Other results get new handles: one object
may have several (``objects.isequal``).

SuiteSparse also calls an op on the iso values of empty iso inputs and
discards the result. Such a value may be handle 0 (never filled): in a call
with an empty iso input the function is then not called for it. Handle 0
anywhere else raises ``objects.InvalidHandle``.

A Python op may itself run a GraphBLAS call with a Python op, on its own
thread, when its call uses one thread (always with the GIL; ``nthreads=1``
without it). Otherwise that raises RuntimeError: the calls share one error
state, and a worker would wait for its own call.

Errors: the first exception in a call is kept, the function is not called
again in that call (each later element gets handle 0), and ``run`` raises it
after GraphBLAS returns. ``.new()`` then drops its result. ``C << expr`` has
already written into ``C``, which may hold some new values and some handles 0;
handle 0 fails when it is read or used by a Python op (``objects.InvalidHandle``).
Every exception is handled so, SystemExit and KeyboardInterrupt included, and
while a call runs Ctrl-C goes to the error slot (``_route_sigint``).
"""

import signal
import sys
import threading
import weakref

from ... import backend
from ...dtypes import BOOL
from ...exceptions import check_status_carg
from .. import ffi, lib, objects

_HAS_GIL = getattr(sys, "_is_gil_enabled", lambda: True)()

# Python-op calls run one at a time, so one error slot serves them. Reentrant:
# a Python op may itself run a GraphBLAS call that runs a Python op.
_run_lock = threading.RLock()
_run_owner = None  # thread id holding _run_lock
_error = None
_error_lock = threading.Lock()

# id of each typed op that calls Python -> ("c", func) or ("cffi", callback);
# removed when the op is garbage (``_forget``)
_callbacks = {}

# BaseException's own subclasses: what the bodies below catch, as the C trampolines
# do (flake8 refuses ``except BaseException`` that is not re-raised in place)
_ANY = (Exception, BaseExceptionGroup, GeneratorExit, KeyboardInterrupt, SystemExit)

# Whether a running call may see handle 0 in an input: only with an empty iso input
# (``objects.has_empty_iso``), whose iso value SuiteSparse passes without using it
_allow_zero = False
_run_threads = 1  # threads the running call may use (1, or None for SuiteSparse's count)
_in_cffi = threading.local()  # depth: this thread runs a function for a cffi callback


def _marked(func):
    """``func``, counting in ``_in_cffi`` while it runs (``run`` checks it)."""

    def call(*args):
        _in_cffi.depth = getattr(_in_cffi, "depth", 0) + 1
        try:
            return func(*args)
        finally:
            _in_cffi.depth -= 1

    return call


_SIGNATURES = {
    "UnaryOp": "GxB_unary_function",
    "BinaryOp": "GxB_binary_function",
    "IndexUnaryOp": "GxB_index_unary_function",
    "SelectOp": "GxB_index_unary_function",
}


class _Skip(Exception):
    """Handle 0 in a call with an empty iso input: the element is skipped (``_cshim.skip``)."""


def _reader(dtype):
    """Return ``read(void_ptr) -> value`` for an input of type ``dtype``."""
    if dtype._is_object:
        resolve = objects.resolve_in_call

        def read(ptr):
            handle = ffi.cast("uint64_t*", ptr)[0]
            if handle == 0 and _allow_zero:
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
    func = _marked(func)
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
            except _ANY as exc:  # every exception, as the C path; ``run`` raises it
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
            except _ANY as exc:  # every exception, as the C path; ``run`` raises it
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
                except _ANY as exc:  # every exception, as the C path; ``run`` raises it
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
                except _ANY as exc:  # every exception, as the C path; ``run`` raises it
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
    weakref.finalize(op, _forget, id(op), release)
    parent._udt_types[key] = ret_type
    parent._udt_ops[key] = op
    return op


def _forget(key, release):
    """Drop a dead op's entry (its function and callback), and free its C trampoline."""
    _callbacks.pop(key, None)
    if release is not None:
        release()


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

    A descriptor made with other options (``axb_method``) is made for this call
    (``descriptor.lookup``), so its thread count is set in place.
    """
    from ..descriptor import BuiltinDescriptor, lookup

    if desc is None:
        return lookup(nthreads=1)
    if type(desc) is BuiltinDescriptor:
        return lookup(
            output_replace=desc.output_replace,
            mask_complement=desc.mask_complement,
            mask_structure=desc.mask_structure,
            transpose_first=desc.transpose_first,
            transpose_second=desc.transpose_second,
            nthreads=1,
        )
    _desc_config(desc)["nthreads"] = 1
    return desc


def _asks_one_thread(args):
    """Whether the call's descriptor (``.new(nthreads=1)``) limits it to one thread."""
    from ..descriptor import BuiltinDescriptor, Descriptor

    desc = args[-1] if args else None
    if not isinstance(desc, Descriptor) or type(desc) is BuiltinDescriptor:
        return False
    return desc._context is not None and _desc_config(desc)["nthreads"] == 1


def _desc_config(desc):
    """The options of an existing descriptor (``_DescriptorConfig()`` makes a new one)."""
    from ..ss.config import BaseConfig
    from ..ss.descriptor import _DescriptorConfig

    config = _DescriptorConfig.__new__(_DescriptorConfig)
    # nthreads lives in the descriptor's Context, which setting it makes if needed
    BaseConfig.__init__(config, desc, desc._context)
    return config


def _route_sigint():
    """While a call runs, send Ctrl-C to the error slot; return the handler to restore.

    The interpreter runs a SIGINT handler at a later point in the main thread,
    which may be inside cffi's wrapper (which drops the KeyboardInterrupt) or
    after the call. Only the default handler is replaced, from the main thread.
    """
    if threading.current_thread() is not threading.main_thread():
        return None
    old = signal.getsignal(signal.SIGINT)
    if old is not signal.default_int_handler:
        return None
    signal.signal(signal.SIGINT, _on_sigint)
    return old


def _on_sigint(signum, frame):
    from . import _cshim

    _set_error(KeyboardInterrupt())
    _cshim.set_error_flag()


def run(call, cfunc_name, args, kind):
    """Run a GraphBLAS call whose op calls Python; raise the first error it hit.

    ``kind`` is from ``python_op_kind``. A cffi callback takes the GIL per call
    and does not scale, and with the GIL no trampoline does, so those calls ask
    SuiteSparse for one thread. A C trampoline on a free-threaded build uses
    SuiteSparse's thread count.

    Holds the collector's lock shared, so a collection asked for by the many
    allocations of a call waits until the call returns (``objects._OpLock``).
    """
    global _error, _run_owner, _allow_zero, _run_threads
    from ..descriptor import Descriptor
    from . import _cshim

    one = kind == "cffi" or _HAS_GIL
    if (
        one
        and backend == "suitesparse"
        and args
        and (args[-1] is None or isinstance(args[-1], Descriptor))
    ):
        args = [*args[:-1], _one_thread(args[-1])]
    # A nested call: on a worker of a running call, it would wait forever for the
    # _run_lock its call holds; on the call's own thread, it is fine unless other
    # threads run the outer call's op meanwhile (one error state). Another Python
    # thread's call just waits for its turn.
    if (_run_owner == threading.get_ident() and _run_threads != 1) or (
        _run_owner != threading.get_ident()
        and (_cshim.in_callback() or getattr(_in_cffi, "depth", 0))
    ):
        raise RuntimeError(
            "a Python op on OBJECT cannot run a GraphBLAS call with a Python op while its "
            "own call may use several threads; use nthreads=1 for the outer call"
        )
    with _run_lock:
        owner, _run_owner = _run_owner, threading.get_ident()
        # Only the suitesparse backend can ask for one thread (``_one_thread``)
        single = backend == "suitesparse" and (one or _asks_one_thread(args))
        threads, _run_threads = _run_threads, 1 if single else None
        outer, _error = _error, None
        # args[0] is the output: an empty one passes no values to the op
        allow_zero, _allow_zero = _allow_zero, objects.has_empty_iso(args[1:])
        saved = _cshim.enter_call(_allow_zero)
        sigint = _route_sigint()
        objects._lock.acquire_shared()
        try:
            rv = call(cfunc_name, args)
        finally:
            objects._lock.release_shared()
            if sigint is not None:
                signal.signal(signal.SIGINT, sigint)
            error, _error = _error, outer
            _run_owner, _run_threads, _allow_zero = owner, threads, allow_zero
            _cshim.exit_call(saved)
    if error is not None:
        raise error
    return rv
