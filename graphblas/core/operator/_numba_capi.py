"""Trampolines compiled by Numba that call the CPython C API: the no-compiler fast path.

``operator.pyobject`` uses these when ``_cshim`` cannot be built (no C compiler)
and Numba is installed. They do what ``_cshim``'s C functions do at a similar
cost: take the GIL (attach, without it), read the values SuiteSparse passes,
resolve handles in the heap of ``core.objects``, call the Python function with
``PyObject_Vectorcall``, and store the result in a slot of this thread's block.

A Numba ``cfunc`` cannot hold Python objects, so object pointers are ``uint64``
values: the C API is called through ctypes function pointers declared with
``uint64`` for ``PyObject *``. Memory is read and written with small LLVM
intrinsics, which also give the atomics the heap needs on every platform. Each
thread's block of slots, its call depth and its argument arrays live in a
calloc'ed area that a pthread key points to (freed when the thread exits).

The heap is shared with ``core.objects`` as ``_cshim`` shares it: this module
becomes ``objects._heap.c``, so every claim, Python's too, is atomic; chunk
pointers are published before the chunk count, which is a release store;
collections run only between calls and bump the epoch, which retires every
thread's block.

A GraphBLAS function pointer carries no user data, so each op gets an entry of
a pool. One cfunc (``run``) takes the entry as an argument, and each pool entry
is a three-instruction LLVM function that calls ``run`` with its entry number.
The pool is compiled once, in about 0.2 seconds; ``run`` and its helpers take
about a second to compile with Numba, once per process (they hold the addresses
of C API functions, so Numba cannot cache them).
"""

import ctypes
import os
import sys
import threading
import weakref

import numpy as np

from .. import ffi as gb_ffi
from .. import objects

NPOOL = 1024  # trampolines per arity
_BLOCK = 64  # slots a thread claims at once (as _cshim's BLOCK)
_MAXDEPTH = 8  # nested trampolines on one thread whose arguments live in its area
# A thread's area, in uint64 words: epoch, next, n, depth, args (4 per depth), block
_T_EPOCH, _T_NEXT, _T_N, _T_DEPTH, _T_ARGS = 0, 1, 2, 3, 4
_T_BLK = _T_ARGS + 4 * _MAXDEPTH
_T_WORDS = _T_BLK + _BLOCK
# How to read an input, as in _cshim
K_HANDLE = 0
_KINDS = {
    "BOOL": 1,
    "INT8": 2,
    "INT16": 3,
    "INT32": 4,
    "INT64": 5,
    "UINT8": 6,
    "UINT16": 7,
    "UINT32": 8,
    "UINT64": 9,
    "FP32": 10,
    "FP64": 11,
}
_ARITY = {
    "UnaryOp": "unary",
    "BinaryOp": "binary",
    "IndexUnaryOp": "indexunary",
    "SelectOp": "indexunary",
}
_NARGS = {"unary": 1, "binary": 2, "indexunary": 3}
_SIGNATURES = {
    "unary": "GxB_unary_function",
    "binary": "GxB_binary_function",
    "indexunary": "GxB_index_unary_function",
}
# Words of _INFO, which the compiled code reads
I_STATE, I_FREE, I_NCHUNKS, I_THRESH, I_ERR, I_ADD, I_ONERR, I_BAD, I_KEY, I_ZERO = range(10)

_lock = threading.Lock()
_impl = None  # dict of compiled functions; False if Numba is missing or this is disabled
_engine = None  # the LLVM engine that holds the pool
_pool = {}  # arity -> addresses of its trampolines
_free = {"unary": [], "binary": [], "indexunary": []}
_next = {"unary": 0, "binary": 0, "indexunary": 0}
_funcs = {}  # (arity, entry) -> the callable its context names
_keep = []  # callables the compiled code calls
_INFO = np.zeros(16, dtype=np.uint64)
_INFO[I_ZERO] = 1  # skip handle 0 unless a call says otherwise (``enter_call``)
_OBJS = np.zeros(objects.MAX_CHUNKS, dtype=np.uint64)  # chunk -> PyObject **
_GENS = np.zeros(objects.MAX_CHUNKS, dtype=np.uint64)  # chunk -> uint32_t *
_ALIVES = np.zeros(objects.MAX_CHUNKS, dtype=np.uint64)  # chunk -> uint8_t *
_CTX = {a: np.zeros((NPOOL, 4), dtype=np.uint64) for a in _NARGS}  # func, kx, ky, out_bool


def enabled():
    """False when ``GRAPHBLAS_NUMBA_CAPI=0``: Python ops then use cffi callbacks."""
    return os.environ.get("GRAPHBLAS_NUMBA_CAPI", "1") != "0"


def _cfn(lib, name, restype, *argtypes):
    f = getattr(lib, name)
    f.restype = restype
    f.argtypes = list(argtypes)
    return f


def _pthread_key_t():
    return ctypes.c_ulong if sys.platform == "darwin" else ctypes.c_uint


def _build():
    """Compile the shared functions (once per process)."""
    import numba
    from llvmlite import ir
    from numba import types as nt
    from numba.core import errors
    from numba.extending import intrinsic

    U64 = ctypes.c_uint64
    api = ctypes.pythonapi
    libc = ctypes.CDLL(None)
    gil_ensure = _cfn(api, "PyGILState_Ensure", ctypes.c_int)
    gil_release = _cfn(api, "PyGILState_Release", None, ctypes.c_int)
    gil_this = _cfn(api, "PyGILState_GetThisThreadState", U64)
    save_thread = _cfn(api, "PyEval_SaveThread", U64)
    vectorcall = _cfn(api, "PyObject_Vectorcall", U64, U64, U64, ctypes.c_size_t, U64)
    incref = _cfn(api, "Py_IncRef", None, U64)
    decref = _cfn(api, "Py_DecRef", None, U64)
    err_get = _cfn(api, "PyErr_GetRaisedException", U64)
    err_clear = _cfn(api, "PyErr_Clear", None)
    from_i64 = _cfn(api, "PyLong_FromLongLong", U64, ctypes.c_int64)
    from_u64 = _cfn(api, "PyLong_FromUnsignedLongLong", U64, U64)
    from_f64 = _cfn(api, "PyFloat_FromDouble", U64, ctypes.c_double)
    from_bool = _cfn(api, "PyBool_FromLong", U64, ctypes.c_long)
    is_true = _cfn(api, "PyObject_IsTrue", ctypes.c_int, U64)
    key_t = _pthread_key_t()
    tls_get = _cfn(libc, "pthread_getspecific", U64, key_t)
    tls_set = _cfn(libc, "pthread_setspecific", ctypes.c_int, key_t, U64)
    calloc = _cfn(libc, "calloc", U64, ctypes.c_size_t, ctypes.c_size_t)
    cfree = _cfn(libc, "free", None, U64)
    none = id(None)

    def _ints(*tys):
        if not all(isinstance(t, nt.Integer) for t in tys):
            raise errors.TypingError(f"an address or a word must be an integer, not {tys}")

    def _args(context, builder, sig, args):
        return [context.cast(builder, v, t, nt.uint64) for v, t in zip(args, sig.args, strict=True)]

    def _ptr(builder, addr, bits):
        return builder.inttoptr(addr, ir.IntType(bits).as_pointer())

    def _load(bits, ordering=None):
        @intrinsic
        def load(typingctx, addr):
            _ints(addr)

            def codegen(context, builder, sig, args):
                p = _ptr(builder, _args(context, builder, sig, args)[0], bits)
                if ordering is None:
                    return builder.load(p)
                return builder.load_atomic(p, ordering, bits // 8)

            return getattr(nt, f"uint{bits}")(addr), codegen

        return load

    def _store(bits, ordering=None):
        @intrinsic
        def store(typingctx, addr, val):
            _ints(addr, val)

            def codegen(context, builder, sig, args):
                a, v = _args(context, builder, sig, args)
                if bits < 64:
                    v = builder.trunc(v, ir.IntType(bits))
                if ordering is None:
                    builder.store(v, _ptr(builder, a, bits))
                else:
                    builder.store_atomic(v, _ptr(builder, a, bits), ordering, bits // 8)
                return context.get_dummy_value()

            return nt.void(addr, val), codegen

        return store

    load8, load16, load32, load64 = (_load(b) for b in (8, 16, 32, 64))
    load8_acq, load64_acq, load64_rlx = (
        _load(8, "acquire"),
        _load(64, "acquire"),
        _load(64, "monotonic"),
    )
    store8, store32, store64 = _store(8), _store(32), _store(64)
    store8_rel, store64_rel, store64_rlx = (
        _store(8, "release"),
        _store(64, "release"),
        _store(64, "monotonic"),
    )

    @intrinsic
    def fetch_add(typingctx, addr, val):
        """Atomically add val at addr; return the old value."""
        _ints(addr, val)

        def codegen(context, builder, sig, args):
            a, v = _args(context, builder, sig, args)
            return builder.atomic_rmw("add", _ptr(builder, a, 64), v, "seq_cst")

        return nt.uint64(addr, val), codegen

    @intrinsic
    def cas(typingctx, addr, expected, desired):
        """Atomically replace expected with desired at addr; return 1 if it did."""
        _ints(addr, expected, desired)

        def codegen(context, builder, sig, args):
            a, e, d = _args(context, builder, sig, args)
            res = builder.cmpxchg(_ptr(builder, a, 64), e, d, "acq_rel", "acquire")
            return builder.zext(builder.extract_value(res, 1), ir.IntType(64))

        return nt.uint64(addr, expected, desired), codegen

    @intrinsic
    def as_f64(typingctx, bits):
        _ints(bits)

        def codegen(context, builder, sig, args):
            return builder.bitcast(_args(context, builder, sig, args)[0], ir.DoubleType())

        return nt.float64(bits), codegen

    @intrinsic
    def as_f32(typingctx, bits):
        _ints(bits)

        def codegen(context, builder, sig, args):
            v = builder.trunc(_args(context, builder, sig, args)[0], ir.IntType(32))
            return builder.fpext(builder.bitcast(v, ir.FloatType()), ir.DoubleType())

        return nt.float64(bits), codegen

    U = np.uint64
    INFO = U(_INFO.ctypes.data)
    DOBJS, DGENS, DALIVES = U(_OBJS.ctypes.data), U(_GENS.ctypes.data), U(_ALIVES.ctypes.data)
    CTX = {_NARGS[a]: U(c.ctypes.data) for a, c in _CTX.items()}
    CTX1, CTX2, CTX3 = CTX[1], CTX[2], CTX[3]
    CB = U(objects._CHUNK_BITS)
    OFF = U((1 << objects._CHUNK_BITS) - 1)
    SLOT_BITS = U(objects._SLOT_BITS)
    SLOT_MASK = U(objects._SLOT_MASK)
    S_TOP, S_NFREE, S_ALLOC, S_NLIVE, S_EPOCH, S_GC = (
        U(objects.S_TOP),
        U(objects.S_NFREE),
        U(objects.S_ALLOCATED),
        U(objects.S_NLIVE),
        U(objects.S_EPOCH),
        U(objects.S_GC),
    )
    BLOCK, MAXDEPTH, TWORDS = U(_BLOCK), U(_MAXDEPTH), U(_T_WORDS)
    T_EPOCH, T_NEXT, T_N, T_DEPTH, T_ARGS, T_BLK = (
        U(v) for v in (_T_EPOCH, _T_NEXT, _T_N, _T_DEPTH, _T_ARGS, _T_BLK)
    )
    W = U(8)
    ZERO, ONE = U(0), U(1)
    KH = U(K_HANDLE)

    @numba.njit
    def word(i):
        return INFO + W * U(i)

    @numba.njit
    def state(i):
        return load64(word(I_STATE)) + W * i

    @numba.njit
    def claim_into(out, n):
        """Claim n slots into the int64 array at out: from the free stack, then from top."""
        out = U(out)
        n = U(n)
        nfree = state(S_NFREE)
        got = ZERO
        k = load64_acq(nfree)
        while k > ZERO:
            take = min(n, k)
            if cas(nfree, k, k - take):
                fr = load64(word(I_FREE)) + W * (k - take)
                for i in range(take):
                    store64(out + W * U(i), load64(fr + W * U(i)))
                got = take
                break
            k = load64_acq(nfree)
        if got < n:
            start = fetch_add(state(S_TOP), n - got)
            for i in range(got, n):
                store64(out + W * U(i), start + U(i) - got)

    @numba.njit
    def count(n):
        n = U(n)
        fetch_add(state(S_NLIVE), n)
        if fetch_add(state(S_ALLOC), n) + n > load64_rlx(word(I_THRESH)):
            store64_rlx(state(S_GC), ONE)

    @numba.njit
    def store_release(addr, val):
        store64_rel(addr, val)

    @numba.njit
    def ensure():
        # An OpenMP worker has no thread state: make one on its first call and keep it
        # (one per worker, never freed), as _cshim's ensure_gil does
        if gil_this() == ZERO:
            gil_ensure()
            save_thread()
        return gil_ensure()

    @numba.njit
    def area():
        """The calling thread's area, made on its first call."""
        key = load64(word(I_KEY))
        t = tls_get(key)
        if t == ZERO:
            t = calloc(TWORDS, W)
            if t != ZERO:
                tls_set(key, t)
        return t

    @numba.njit
    def call1(fn, a):
        """fn(a) for objects at fn and a: a new reference, or 0."""
        buf = calloc(ONE, W)
        if buf == ZERO:
            return ZERO
        store64(buf, a)
        r = vectorcall(fn, buf, 1, ZERO)
        cfree(buf)
        return r

    @numba.njit
    def drop(o):
        if o != ZERO:
            decref(o)

    @numba.njit
    def report():
        """Hand the raised exception (or None) to on_error, which keeps the first."""
        exc = err_get()
        store64_rlx(word(I_ERR), ONE)
        drop(call1(load64(word(I_ONERR)), exc if exc != ZERO else U(none)))
        drop(exc)
        err_clear()

    @numba.njit
    def bad(h):
        ph = from_u64(h)
        drop(call1(load64(word(I_BAD)), ph) if ph != ZERO else ZERO)
        drop(ph)
        err_clear()
        store64_rlx(word(I_ERR), ONE)

    @numba.njit
    def get_handle(h):
        """The object of handle h, as a new reference; 0 (error reported) unless live."""
        slot = h & SLOT_MASK
        k = slot >> CB
        off = slot & OFF
        if (
            slot != ZERO
            and k < load64_acq(word(I_NCHUNKS))
            and U(load32(load64(DGENS + W * k) + U(4) * off)) == h >> SLOT_BITS
            and load8_acq(load64(DALIVES + W * k) + off) != 0
        ):
            o = load64(load64(DOBJS + W * k) + W * off)
            incref(o)
            return o
        bad(h)
        return ZERO

    @numba.njit
    def get_value(kind, p):
        if kind == KH:
            return get_handle(load64(p))
        if kind == U(1):
            o = from_bool(1 if load8(p) != 0 else 0)
        elif kind == U(2):
            o = from_i64(np.int64(np.int8(load8(p))))
        elif kind == U(3):
            o = from_i64(np.int64(np.int16(load16(p))))
        elif kind == U(4):
            o = from_i64(np.int64(np.int32(load32(p))))
        elif kind == U(5):
            o = from_i64(np.int64(load64(p)))
        elif kind == U(6):
            o = from_u64(U(load8(p)))
        elif kind == U(7):
            o = from_u64(U(load16(p)))
        elif kind == U(8):
            o = from_u64(U(load32(p)))
        elif kind == U(9):
            o = from_u64(load64(p))
        elif kind == U(10):
            o = from_f64(as_f32(load32(p)))
        else:
            o = from_f64(as_f64(load64(p)))
        if o == ZERO:
            report()
        return o

    @numba.njit
    def put_obj(t, r):
        """Store r (a new reference, stolen) in a fresh slot; return its handle, or 0."""
        epoch = load64_acq(state(S_EPOCH))
        blk = t + W * T_BLK
        if load64(t + W * T_EPOCH) != epoch or load64(t + W * T_NEXT) >= load64(t + W * T_N):
            claim_into(blk, BLOCK)
            # Count a block when claimed, not each result, as _cshim does: a shared atomic
            # per result keeps threads from scaling. The collector recounts live slots.
            count(BLOCK)
            store64(t + W * T_NEXT, ZERO)
            store64(t + W * T_N, BLOCK)
            store64(t + W * T_EPOCH, epoch)
            mx = ZERO
            for i in range(BLOCK):
                s = load64(blk + W * U(i))
                mx = max(mx, s)
            if mx >> CB >= load64_acq(word(I_NCHUNKS)):
                upto = from_u64(mx + ONE)
                res = ZERO
                if upto != ZERO:
                    res = call1(load64(word(I_ADD)), upto)
                    decref(upto)
                if res == ZERO:
                    store64(t + W * T_N, ZERO)
                    report()
                    decref(r)
                    return ZERO
                decref(res)
        i = load64(t + W * T_NEXT)
        store64(t + W * T_NEXT, i + ONE)
        slot = load64(blk + W * i)
        k = slot >> CB
        off = slot & OFF
        pobj = load64(DOBJS + W * k) + W * off
        drop_old = load64(pobj)  # None in a free slot
        store64(pobj, r)
        drop(drop_old)
        pgen = load64(DGENS + W * k) + U(4) * off
        g = U(load32(pgen))
        if g == ZERO:
            g = ONE
            store32(pgen, g)
        store8_rel(load64(DALIVES + W * k) + off, ONE)
        return (g << SLOT_BITS) | slot

    @numba.cfunc(nt.void(*[nt.uint64] * 7))
    def run(entry, nargs, z, x, i, j, y):
        """The body of every trampoline; nargs is 1 (unary), 2 (binary) or 3 (index unary)."""
        g = ensure()
        t = area()
        if t == ZERO:  # out of memory: no area, so no result
            store64(z, ZERO)
            gil_release(g)
            return
        d = load64(t + W * T_DEPTH)
        store64(t + W * T_DEPTH, d + ONE)
        c = (CTX1 if nargs == ONE else (CTX2 if nargs == U(2) else CTX3)) + U(32) * entry
        fn = load64(c)
        kx = load64(c + W)
        ky = load64(c + U(16))
        out_bool = load64(c + U(24)) != ZERO
        on_heap = d >= MAXDEPTH
        args = calloc(U(4), W) if on_heap else t + W * (T_ARGS + U(4) * d)
        if not on_heap:
            for q in range(4):
                store64(args + W * U(q), ZERO)
        out = ZERO
        truth = 0
        # Skip after an error, and on handle 0 when the call has an empty iso input
        # (``_cshim.skip``); handle 0 anywhere else fails in get_handle
        if not (
            min(args, fn) == ZERO  # no memory for the arguments, or a released entry
            or load64_rlx(word(I_ERR)) != ZERO
            or (
                load64_rlx(word(I_ZERO)) != ZERO
                and (
                    (kx == KH and load64(x) == ZERO)
                    or (nargs != ONE and ky == KH and load64(y) == ZERO)
                )
            )
        ):
            a = get_value(kx, x)
            store64(args, a)
            b = ZERO
            if a != ZERO and nargs != ONE:
                b = get_value(ky, y)
                store64(args + (W if nargs == U(2) else U(24)), b)
            if a != ZERO and (nargs == ONE or b != ZERO):
                ok = True
                if nargs == U(3):
                    pi = from_u64(i)
                    store64(args + W, pi)
                    pj = from_u64(j)
                    store64(args + U(16), pj)
                    ok = min(pi, pj) != ZERO
                    if not ok:
                        report()
                if ok:
                    r = vectorcall(fn, args, 4 if nargs == U(3) else nargs, ZERO)
                    if r == ZERO:
                        report()
                    elif out_bool:
                        truth = is_true(r)
                        decref(r)
                        if truth < 0:
                            truth = 0
                            report()
                    elif kx == KH and r == a:  # the result is an input: reuse its handle
                        decref(r)
                        out = load64(x)
                    elif nargs == U(2) and ky == KH and r == b:
                        decref(r)
                        out = load64(y)
                    else:
                        out = put_obj(t, r)
            for q in range(4):
                drop(load64(args + W * U(q)))
        if on_heap and args != ZERO:
            cfree(args)
        if out_bool:
            store8(z, ONE if truth else ZERO)
        else:
            store64(z, out)
        store64(t + W * T_DEPTH, d)
        gil_release(g)

    @numba.njit
    def depth():
        t = tls_get(load64(word(I_KEY)))
        return ZERO if t == ZERO else load64(t + W * T_DEPTH)

    return {
        "run": run,
        "claim_into": claim_into,
        "count": count,
        "store_release": store_release,
        "depth": depth,
        "free_addr": ctypes.cast(libc.free, ctypes.c_void_p).value,
        "helpers": (
            word,
            state,
            claim_into,
            count,
            ensure,
            area,
            call1,
            drop,
            report,
            bad,
            get_handle,
            get_value,
            put_obj,
            depth,
        ),
    }


def _make_pool(run_addr):
    """Compile NPOOL trampolines per arity, each calling ``run`` with its entry number."""
    global _engine
    import llvmlite.binding as llvm
    from llvmlite import ir

    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()
    i64 = ir.IntType(64)
    runty = ir.FunctionType(ir.VoidType(), [i64] * 7)
    mod = ir.Module("graphblas_numba_capi_pool")
    mod.triple = llvm.get_process_triple()
    zero = ir.Constant(i64, 0)
    names = {}
    for arity, nargs in _NARGS.items():
        nparams = {1: 2, 2: 3, 3: 5}[nargs]  # z, x[, i, j][, y]
        names[arity] = []
        for k in range(NPOOL):
            fn = ir.Function(mod, ir.FunctionType(ir.VoidType(), [i64] * nparams), f"{arity}_{k}")
            b = ir.IRBuilder(fn.append_basic_block())
            a = list(fn.args)
            if nargs == 1:
                ijy = [zero, zero, zero]
            elif nargs == 2:
                ijy = [zero, zero, a[2]]
            else:
                ijy = a[2:5]
            target = b.inttoptr(ir.Constant(i64, run_addr), runty.as_pointer())
            b.call(
                target, [ir.Constant(i64, k), ir.Constant(i64, nargs), a[0], a[1], *ijy], tail=True
            )
            b.ret_void()
            names[arity].append(fn.name)
    llmod = llvm.parse_assembly(str(mod))
    llmod.verify()
    tm = llvm.Target.from_default_triple().create_target_machine(opt=2, jit=True)
    _engine = llvm.create_mcjit_compiler(llmod, tm)
    _engine.finalize_object()
    for arity, fnames in names.items():
        _pool[arity] = [_engine.get_function_address(n) for n in fnames]


def load():
    """Compile everything once; False if this cannot run here or ``GRAPHBLAS_NUMBA_CAPI=0``.

    It needs Numba and POSIX threads (pthread keys), so not Windows.
    """
    global _impl
    with _lock:
        if _impl is None:
            _impl = False
            if enabled() and sys.platform != "win32":
                try:
                    _impl = _load()
                except Exception:  # no Numba, or it cannot compile this: use cffi
                    _impl = False
        return bool(_impl)


def _load():
    import numba  # noqa: F401

    impl = _build()
    key = _pthread_key_t()(0)
    create = _cfn(
        ctypes.CDLL(None), "pthread_key_create", ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p
    )
    if create(ctypes.byref(key), impl["free_addr"]) != 0:
        raise RuntimeError("pthread_key_create failed")
    _INFO[I_KEY] = key.value
    _make_pool(impl["run"].address)
    return impl


class _NumbaHeap:
    """What ``objects._Heap`` calls once these trampolines share the heap (as ``_CHeap``)."""

    def __init__(self, impl):
        self._claim = impl["claim_into"]
        self._count = impl["count"]
        self._store_release = impl["store_release"]

    def publish(self, k, objs, gens, alives):
        _OBJS[k] = objs.ctypes.data
        _GENS[k] = gens.ctypes.data
        _ALIVES[k] = alives.ctypes.data
        self._store_release(_INFO.ctypes.data + 8 * I_NCHUNKS, k + 1)

    def claim(self, n):
        out = np.empty(n, dtype=np.int64)
        if n:
            self._claim(out.ctypes.data, n)
        return out

    def claim_one(self):
        out = np.empty(1, dtype=np.int64)
        self._claim(out.ctypes.data, 1)
        return int(out[0])

    def count(self, n):
        self._count(n)

    def set_free(self, free):
        _INFO[I_FREE] = free.ctypes.data
        _INFO[I_THRESH] = _threshold()


def _threshold():
    return int(max(objects.threshold, objects.GC_FACTOR * objects._heap.live_after_gc))


def _attach():
    """Share the heap; False if a call is running (try again later) or _cshim has it."""
    heap = objects._heap
    if heap.c is not None:
        return isinstance(heap.c, _NumbaHeap)
    # Never wait on a worker inside a call: the call would wait for the worker. The op
    # then uses a cffi callback.
    if not objects._lock.acquire_exclusive(blocking=not in_callback()):
        return False  # inside a call: claims are not atomic until no call runs
    try:
        if heap.c is not None:  # another thread attached while we waited
            return isinstance(heap.c, _NumbaHeap)
        from . import pyobject

        def add_chunks(upto):
            heap.add_chunks(upto)

        def on_error(exc):
            if exc is None:  # a failure with no exception set must still fail the call
                exc = RuntimeError("a Python op on OBJECT failed")
            pyobject._set_error(exc)

        def invalid_handle(h):
            pyobject._set_error(objects.InvalidHandle(f"{h:#x} is not the handle of a live object"))

        _keep.extend([add_chunks, on_error, invalid_handle])
        _INFO[I_STATE] = heap.state.ctypes.data
        _INFO[I_ADD] = id(add_chunks)
        _INFO[I_ONERR] = id(on_error)
        _INFO[I_BAD] = id(invalid_handle)
        c = _NumbaHeap(_impl)
        c.set_free(heap.free)
        with heap._chunk_lock:
            for k, (objs, gens, alives) in enumerate(
                zip(heap.objs, heap.gens, heap.alives, strict=True)
            ):
                c.publish(k, objs, gens, alives)
            heap.c = c
    finally:
        objects._lock.release_exclusive()
    return True


def _kind(dtype):
    if dtype._is_object:
        return K_HANDLE
    return _KINDS.get(dtype.name)


def make(kind, func, dtype, dtype2, out_bool):
    """Like ``_cshim.make``: (function pointer, release), or None to use a cffi callback."""
    if not load() or not _attach():
        return None
    kx = _kind(dtype)
    ky = K_HANDLE if dtype2 is None else _kind(dtype2)
    if kx is None or ky is None:
        return None
    arity = _ARITY[kind]
    with _lock:
        if _free[arity]:
            k = _free[arity].pop()
        elif _next[arity] < NPOOL:
            k = _next[arity]
            _next[arity] += 1
        else:
            return None
        _funcs[arity, k] = func
        _CTX[arity][k] = (id(func), kx, ky, 1 if out_bool else 0)
    fnptr = gb_ffi.cast(_SIGNATURES[arity], _pool[arity][k])

    def release():
        with _lock:
            _CTX[arity][k, 0] = 0
            del _funcs[arity, k]
            _free[arity].append(k)

    return fnptr, release


def in_callback():
    """Whether this thread is running a Python op for one of these trampolines."""
    return bool(_impl) and _impl["depth"]() != 0


def reset_error():
    """Clear the error flag, and take up a new ``objects.threshold`` (as ``_cshim``)."""
    if _impl:
        _INFO[I_ERR] = 0
        _INFO[I_THRESH] = _threshold()


def enter_call(allow_zero):
    """Start a Python-op call: return the error state to restore (``exit_call``)."""
    if not _impl:
        return None
    saved = (int(_INFO[I_ERR]), int(_INFO[I_ZERO]))
    _INFO[I_ERR] = 0
    _INFO[I_ZERO] = 1 if allow_zero else 0
    _INFO[I_THRESH] = _threshold()
    return saved


def exit_call(saved):
    """End a Python-op call: put back the state of the call around it, if any."""
    if saved is not None:
        _INFO[I_ERR], _INFO[I_ZERO] = saved


def set_error_flag():
    """Stop the running call's trampolines (Ctrl-C)."""
    if _impl:
        _INFO[I_ERR] = 1


def finalize_release(op, release):
    """Free the pool entry when the typed op (and so its GrB op) is garbage."""
    weakref.finalize(op, release)
