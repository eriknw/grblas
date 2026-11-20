"""C trampolines for Python ops on OBJECT values, and C access to the OBJECT heap.

``operator.pyobject`` uses these when a C compiler is available; otherwise it
uses cffi callbacks, which cost about 20 times more per call. The C source is
built with cffi on first use into ``GRAPHBLAS_CSHIM_DIR`` (default: a
``_cshim_build`` directory beside this file) and loaded from there next time.

A GraphBLAS function pointer carries no user data, so the module has a pool of
distinct C functions per arity, each bound to a context: the Python callable and
how to read each input. A context is reused once its op is garbage.

The C code reads the heap of ``core.objects`` directly. Its chunks never move,
and each chunk's arrays are published once, before the chunk count, with a
release store, so a reader that sees the count sees the chunk. A result gets a
new slot from a per-thread block of slots (one atomic claim per block), or the
handle of an input when the result is that input. Collections run only between
GraphBLAS calls (``objects._OpLock``): they rebuild the free stack and bump an
epoch, which retires every thread's block.
"""

import hashlib
import importlib
import os
import sys
import tempfile
import threading
import weakref
from pathlib import Path

import numpy as np

from .. import ffi as gb_ffi
from .. import objects

NPOOL = 1024  # trampolines per arity

_CDEF = """
typedef struct {
    uint64_t *state;
    int64_t *free;
    uint64_t nchunks;
    uint64_t threshold;
    int errflag;
    uintptr_t add_chunks;
    uintptr_t on_error;
    uintptr_t invalid_handle;
    ...;
} heap_dir;
extern heap_dir HEAP;

typedef struct {
    uintptr_t func;
    int kx;
    int ky;
    int out_bool;
} tramp_ctx;

extern tramp_ctx unary_ctx[@NPOOL@];
extern tramp_ctx binary_ctx[@NPOOL@];
extern tramp_ctx indexunary_ctx[@NPOOL@];
extern void *unary_fn[@NPOOL@];
extern void *binary_fn[@NPOOL@];
extern void *indexunary_fn[@NPOOL@];

void heap_publish(uint64_t k, void *objs, uint32_t *gens, uint8_t *alives);
void heap_claim(uint64_t n, int64_t *out);
void heap_count(uint64_t n);
""".replace("@NPOOL@", str(NPOOL))

_SOURCE = r"""
#include <Python.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>

#define NPOOL @NPOOL@
#define CHUNK_BITS @CHUNK_BITS@
#define OFF_MASK ((1ull << CHUNK_BITS) - 1)
#define MAX_CHUNKS @MAX_CHUNKS@
#define SLOT_BITS 40
#define SLOT_MASK ((1ull << SLOT_BITS) - 1)
enum { S_TOP, S_NFREE, S_ALLOCATED, S_NLIVE, S_EPOCH, S_GC };
/* how to read an input: a handle, or a built-in type */
enum { K_HANDLE, K_BOOL, K_INT8, K_INT16, K_INT32, K_INT64,
       K_UINT8, K_UINT16, K_UINT32, K_UINT64, K_FP32, K_FP64 };

typedef struct {
    uint64_t *state;        /* objects._heap.state */
    int64_t *free;          /* objects._heap.free, set by each collection */
    uint64_t nchunks;       /* chunks published (release store after the pointers) */
    uint64_t threshold;     /* allocations after which a collection is due */
    int errflag;            /* set by the first error in a call; reset per call */
    uintptr_t add_chunks;   /* PyObject *: add_chunks(upto) */
    uintptr_t on_error;     /* PyObject *: on_error(exc) */
    uintptr_t invalid_handle;  /* PyObject *: objects.InvalidHandle */
    PyObject **objs[MAX_CHUNKS];
    uint32_t *gens[MAX_CHUNKS];
    uint8_t *alives[MAX_CHUNKS];
} heap_dir;
heap_dir HEAP;

typedef struct {
    uintptr_t func;  /* PyObject *, kept alive by the typed op */
    int kx, ky;
    int out_bool;
} tramp_ctx;
tramp_ctx unary_ctx[NPOOL];
tramp_ctx binary_ctx[NPOOL];
tramp_ctx indexunary_ctx[NPOOL];

void heap_publish(uint64_t k, void *objs, uint32_t *gens, uint8_t *alives)
{
    HEAP.objs[k] = (PyObject **) objs;
    HEAP.gens[k] = gens;
    HEAP.alives[k] = alives;
    __atomic_store_n(&HEAP.nchunks, k + 1, __ATOMIC_RELEASE);
}

/* Claim n slots, from the free stack first, then from top. Python's claims come
   here too once the module is loaded, so every claim is atomic. */
void heap_claim(uint64_t n, int64_t *out)
{
    uint64_t *nfree = &HEAP.state[S_NFREE];
    uint64_t got = 0;
    uint64_t k = __atomic_load_n(nfree, __ATOMIC_ACQUIRE);
    while (k > 0) {
        uint64_t take = k < n ? k : n;
        if (__atomic_compare_exchange_n(nfree, &k, k - take, 0, __ATOMIC_ACQ_REL,
                                        __ATOMIC_ACQUIRE)) {
            memcpy(out, HEAP.free + (k - take), take * sizeof(int64_t));
            got = take;
            break;
        }
    }
    if (got < n) {
        uint64_t start = __atomic_fetch_add(&HEAP.state[S_TOP], n - got, __ATOMIC_ACQ_REL);
        for (uint64_t i = got; i < n; i++) out[i] = (int64_t) (start + i - got);
    }
}

void heap_count(uint64_t n)
{
    __atomic_fetch_add(&HEAP.state[S_NLIVE], n, __ATOMIC_RELAXED);
    uint64_t a = __atomic_add_fetch(&HEAP.state[S_ALLOCATED], n, __ATOMIC_RELAXED);
    if (a > HEAP.threshold) __atomic_store_n(&HEAP.state[S_GC], 1, __ATOMIC_RELAXED);
}

/* Take the GIL (attach, in a free-threaded build) on any thread. An OpenMP worker has
   no thread state, and PyGILState_Ensure on one makes a thread state that the matching
   Release destroys, which costs microseconds a call; so the first call on each thread
   makes one and keeps it (one per OpenMP worker, never freed). */
static PyGILState_STATE ensure_gil(void)
{
    if (PyGILState_GetThisThreadState() == NULL) {
        PyGILState_Ensure();
        PyEval_SaveThread();
    }
    return PyGILState_Ensure();
}

/* Hand the raised exception to on_error (which keeps the first) and stop the call. */
static void report_error(void)
{
    PyObject *exc = PyErr_GetRaisedException();
    __atomic_store_n(&HEAP.errflag, 1, __ATOMIC_RELAXED);
    if (exc != NULL) {
        PyObject *r = PyObject_Vectorcall((PyObject *) HEAP.on_error, &exc, 1, NULL);
        Py_XDECREF(r);
        Py_DECREF(exc);
        PyErr_Clear();
    }
}

/* The object of a handle, as a new reference; NULL (error reported) unless live */
static PyObject *get_handle(uint64_t h)
{
    uint64_t slot = h & SLOT_MASK;
    uint64_t k = slot >> CHUNK_BITS, off = slot & OFF_MASK;
    if (slot != 0 && k < __atomic_load_n(&HEAP.nchunks, __ATOMIC_ACQUIRE)
        && HEAP.gens[k][off] == (uint32_t) (h >> SLOT_BITS)
        && __atomic_load_n(&HEAP.alives[k][off], __ATOMIC_ACQUIRE)) {
        PyObject *o = HEAP.objs[k][off];
        Py_INCREF(o);
        return o;
    }
    PyErr_Format((PyObject *) HEAP.invalid_handle,
                 "%#llx is not the handle of a live object", (unsigned long long) h);
    report_error();
    return NULL;
}

static PyObject *get_value(int kind, const void *p)
{
    PyObject *o;
    switch (kind) {
    case K_HANDLE: return get_handle(*(const uint64_t *) p);
    case K_BOOL: o = PyBool_FromLong(*(const bool *) p); break;
    case K_INT8: o = PyLong_FromLong(*(const int8_t *) p); break;
    case K_INT16: o = PyLong_FromLong(*(const int16_t *) p); break;
    case K_INT32: o = PyLong_FromLong(*(const int32_t *) p); break;
    case K_INT64: o = PyLong_FromLongLong(*(const int64_t *) p); break;
    case K_UINT8: o = PyLong_FromUnsignedLong(*(const uint8_t *) p); break;
    case K_UINT16: o = PyLong_FromUnsignedLong(*(const uint16_t *) p); break;
    case K_UINT32: o = PyLong_FromUnsignedLong(*(const uint32_t *) p); break;
    case K_UINT64: o = PyLong_FromUnsignedLongLong(*(const uint64_t *) p); break;
    case K_FP32: o = PyFloat_FromDouble(*(const float *) p); break;
    case K_FP64: o = PyFloat_FromDouble(*(const double *) p); break;
    default: PyErr_SetString(PyExc_SystemError, "bad input kind"); o = NULL;
    }
    if (o == NULL) report_error();
    return o;
}

/* Results take slots from a per-thread block: one claim per BLOCK results. A block
   belongs to an epoch; a collection bumps it, frees unused block tails, and every
   thread then claims a new block. */
#define BLOCK 64
static _Thread_local int64_t blk[BLOCK];
static _Thread_local int blk_i, blk_n;
static _Thread_local uint64_t blk_epoch;

/* Store r (a new reference, stolen) in a fresh slot; return its handle, or 0 */
static uint64_t put_obj(PyObject *r)
{
    uint64_t epoch = __atomic_load_n(&HEAP.state[S_EPOCH], __ATOMIC_ACQUIRE);
    if (blk_epoch != epoch || blk_i >= blk_n) {
        heap_claim(BLOCK, blk);
        blk_i = 0;
        blk_n = BLOCK;
        blk_epoch = epoch;
        int64_t mx = 0;
        for (int i = 0; i < BLOCK; i++) if (blk[i] > mx) mx = blk[i];
        if ((uint64_t) mx >> CHUNK_BITS >= __atomic_load_n(&HEAP.nchunks, __ATOMIC_ACQUIRE)) {
            PyObject *upto = PyLong_FromLongLong(mx + 1);
            PyObject *res = upto ? PyObject_Vectorcall((PyObject *) HEAP.add_chunks, &upto, 1,
                                                       NULL) : NULL;
            Py_XDECREF(upto);
            if (res == NULL) {
                blk_n = 0;
                report_error();
                Py_DECREF(r);
                return 0;
            }
            Py_DECREF(res);
        }
    }
    uint64_t slot = (uint64_t) blk[blk_i++];
    uint64_t k = slot >> CHUNK_BITS, off = slot & OFF_MASK;
    PyObject **objs = HEAP.objs[k];
    PyObject *old = objs[off];
    objs[off] = r;
    Py_XDECREF(old);  /* None in a free slot */
    uint32_t g = HEAP.gens[k][off];
    if (g == 0) HEAP.gens[k][off] = g = 1;
    __atomic_store_n(&HEAP.alives[k][off], 1, __ATOMIC_RELEASE);
    heap_count(1);
    return ((uint64_t) g << SLOT_BITS) | slot;
}

static void unary_impl(tramp_ctx *c, void *z, const void *x)
{
    PyGILState_STATE g = ensure_gil();
    uint64_t out = 0;
    PyObject *a = NULL;
    if (__atomic_load_n(&HEAP.errflag, __ATOMIC_RELAXED)) goto done;
    a = get_value(c->kx, x);
    if (a == NULL) goto done;
    PyObject *r = PyObject_Vectorcall((PyObject *) c->func, &a, 1, NULL);
    if (r == NULL) { report_error(); goto done; }
    if (c->kx == K_HANDLE && r == a) { Py_DECREF(r); out = *(const uint64_t *) x; goto done; }
    out = put_obj(r);
done:
    Py_XDECREF(a);
    *(uint64_t *) z = out;
    PyGILState_Release(g);
}

static void binary_impl(tramp_ctx *c, void *z, const void *x, const void *y)
{
    PyGILState_STATE g = ensure_gil();
    uint64_t out = 0;
    PyObject *args[2] = {NULL, NULL};
    if (__atomic_load_n(&HEAP.errflag, __ATOMIC_RELAXED)) goto done;
    args[0] = get_value(c->kx, x);
    if (args[0] == NULL) goto done;
    args[1] = get_value(c->ky, y);
    if (args[1] == NULL) goto done;
    PyObject *r = PyObject_Vectorcall((PyObject *) c->func, args, 2, NULL);
    if (r == NULL) { report_error(); goto done; }
    /* the result is an input (min, max, first, ...): reuse its handle */
    if (c->kx == K_HANDLE && r == args[0]) { Py_DECREF(r); out = *(const uint64_t *) x; }
    else if (c->ky == K_HANDLE && r == args[1]) { Py_DECREF(r); out = *(const uint64_t *) y; }
    else out = put_obj(r);
done:
    Py_XDECREF(args[0]);
    Py_XDECREF(args[1]);
    *(uint64_t *) z = out;
    PyGILState_Release(g);
}

static void indexunary_impl(tramp_ctx *c, void *z, const void *x, uint64_t i, uint64_t j,
                            const void *y)
{
    PyGILState_STATE g = ensure_gil();
    uint64_t out = 0;
    int truth = 0;
    PyObject *args[4] = {NULL, NULL, NULL, NULL};
    if (__atomic_load_n(&HEAP.errflag, __ATOMIC_RELAXED)) goto done;
    args[0] = get_value(c->kx, x);
    if (args[0] == NULL) goto done;
    args[3] = get_value(c->ky, y);
    if (args[3] == NULL) goto done;
    args[1] = PyLong_FromUnsignedLongLong(i);
    args[2] = PyLong_FromUnsignedLongLong(j);
    if (args[1] == NULL || args[2] == NULL) { report_error(); goto done; }
    PyObject *r = PyObject_Vectorcall((PyObject *) c->func, args, 4, NULL);
    if (r == NULL) { report_error(); goto done; }
    if (c->out_bool) {
        truth = PyObject_IsTrue(r);
        Py_DECREF(r);
        if (truth < 0) { truth = 0; report_error(); }
        goto done;
    }
    if (c->kx == K_HANDLE && r == args[0]) { Py_DECREF(r); out = *(const uint64_t *) x; }
    else out = put_obj(r);
done:
    for (int k = 0; k < 4; k++) Py_XDECREF(args[k]);
    if (c->out_bool) *(bool *) z = truth;
    else *(uint64_t *) z = out;
    PyGILState_Release(g);
}

#define U(k) static void unary_##k(void *z, const void *x) \
    { unary_impl(&unary_ctx[k], z, x); }
#define B(k) static void binary_##k(void *z, const void *x, const void *y) \
    { binary_impl(&binary_ctx[k], z, x, y); }
#define I(k) static void indexunary_##k(void *z, const void *x, uint64_t i, uint64_t j, \
    const void *y) { indexunary_impl(&indexunary_ctx[k], z, x, i, j, y); }
#define ALL(M) @ALLM@
ALL(U)
ALL(B)
ALL(I)
#define UP(k) (void *) unary_##k,
#define BP(k) (void *) binary_##k,
#define IP(k) (void *) indexunary_##k,
void *unary_fn[NPOOL] = { ALL(UP) };
void *binary_fn[NPOOL] = { ALL(BP) };
void *indexunary_fn[NPOOL] = { ALL(IP) };
"""
_SOURCE = (
    _SOURCE.replace("@NPOOL@", str(NPOOL))
    .replace("@CHUNK_BITS@", str(objects._CHUNK_BITS))
    .replace("@MAX_CHUNKS@", str(objects.MAX_CHUNKS))
    .replace("@ALLM@", " ".join(f"M({k})" for k in range(NPOOL)))
)

# The source hash in the name rebuilds the module whenever the source changes
_MODNAME = "_graphblas_cshim_" + hashlib.sha256((_CDEF + _SOURCE).encode()).hexdigest()[:12]

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
_SIGNATURES = {
    "unary": "GxB_unary_function",
    "binary": "GxB_binary_function",
    "indexunary": "GxB_index_unary_function",
}

_load_lock = threading.Lock()
_mod = None  # the module, False if it cannot be built
_free = {"unary": [], "binary": [], "indexunary": []}
_next = {"unary": 0, "binary": 0, "indexunary": 0}
_keep = []  # callables the C code calls


def enabled():
    """False when ``GRAPHBLAS_CSHIM=0``: Python ops then use cffi callbacks."""
    return os.environ.get("GRAPHBLAS_CSHIM", "1") != "0"


def build_dir():
    return os.environ.get("GRAPHBLAS_CSHIM_DIR", str(Path(__file__).parent / "_cshim_build"))


def load():
    """Return the C module, building it if needed, or None if it cannot be built."""
    global _mod
    with _load_lock:
        if _mod is None and not enabled():
            _mod = False
        if _mod is None:
            try:
                _mod = _import_or_build()
            except Exception:
                _mod = False
        return _mod or None


def _import_or_build():
    bdir = build_dir()
    Path(bdir).mkdir(parents=True, exist_ok=True)
    if bdir not in sys.path:
        sys.path.insert(0, bdir)
    try:
        return importlib.import_module(_MODNAME)
    except ImportError:
        pass
    import cffi

    builder = cffi.FFI()
    builder.cdef(_CDEF)
    builder.set_source(
        _MODNAME,
        _SOURCE,
        extra_compile_args=["-O2", "-D_CFFI_NO_LIMITED_API"],
        py_limited_api=False,
    )
    # Build in a directory of our own and move the result in: several processes
    # (pytest -n) may build at once
    with tempfile.TemporaryDirectory(dir=bdir) as tmp:
        try:
            path = builder.compile(tmpdir=tmp, verbose=False)
        except Exception as exc:
            if "setuptools" not in str(exc):
                raise
            path = _compile_without_setuptools(builder, tmp)
        Path(path).replace(Path(bdir) / Path(path).name)
    importlib.invalidate_caches()
    return importlib.import_module(_MODNAME)


def _compile_without_setuptools(builder, tmp):
    """Emit the C file with cffi and build it with the C compiler directly.

    cffi needs setuptools to compile on Python 3.12 and later; this does not.
    """
    import subprocess
    import sysconfig

    src = Path(tmp) / f"{_MODNAME}.c"
    builder.emit_c_code(str(src))
    out = Path(tmp) / f"{_MODNAME}{sysconfig.get_config_var('EXT_SUFFIX')}"
    cc = os.environ.get("CC") or sysconfig.get_config_var("CC").split()[0]
    flags = ["-O2", "-fPIC", "-D_CFFI_NO_LIMITED_API", f"-I{sysconfig.get_paths()['include']}"]
    link = ["-bundle", "-undefined", "dynamic_lookup"] if sys.platform == "darwin" else ["-shared"]
    subprocess.run([cc, *flags, *link, str(src), "-o", str(out)], check=True, capture_output=True)
    return str(out)


class _CHeap:
    """What ``objects._Heap`` calls once the C code shares the heap."""

    def __init__(self, mod):
        self.mod = mod
        self.ffi = mod.ffi
        self.lib = mod.lib

    def publish(self, k, objs, gens, alives):
        ffi = self.ffi
        self.lib.heap_publish(
            k,
            ffi.cast("void *", objs.ctypes.data),
            ffi.cast("uint32_t *", gens.ctypes.data),
            ffi.cast("uint8_t *", alives.ctypes.data),
        )

    def claim(self, n):
        out = np.empty(n, dtype=np.int64)
        if n:
            self.lib.heap_claim(n, self.ffi.cast("int64_t *", out.ctypes.data))
        return out

    def count(self, n):
        self.lib.heap_count(n)

    def set_free(self, free):
        self.lib.HEAP.free = self.ffi.cast("int64_t *", free.ctypes.data)
        heap = objects._heap
        self.lib.HEAP.threshold = max(objects.threshold, heap.live_after_gc)


def _attach(mod):
    """Make the C code share the heap; False if a call is running (try again later)."""
    heap = objects._heap
    if heap.c is not None:
        return True
    if not objects._lock.acquire_exclusive(blocking=True):
        return False  # inside a call: claims are not atomic until no call runs
    try:
        lib, ffi = mod.lib, mod.ffi
        c = _CHeap(mod)
        lib.HEAP.state = ffi.cast("uint64_t *", heap.state.ctypes.data)
        c.set_free(heap.free)

        def add_chunks(upto):
            heap.add_chunks(upto)

        def on_error(exc):
            from . import pyobject

            pyobject._set_error(exc)

        _keep.extend([add_chunks, on_error, objects.InvalidHandle])
        lib.HEAP.add_chunks = id(add_chunks)
        lib.HEAP.on_error = id(on_error)
        lib.HEAP.invalid_handle = id(objects.InvalidHandle)
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
        return 0
    return _KINDS.get(dtype.name)


def make(kind, func, dtype, dtype2, out_bool):
    """Return a C function pointer (cdata of the GxB function type) and a release function.

    None if the C module is not available, an input type has no C reader here,
    or the pool is full: the caller then uses a cffi callback.
    """
    mod = load()
    if mod is None or not _attach(mod):
        return None
    kx = _kind(dtype)
    ky = 0 if dtype2 is None else _kind(dtype2)
    if kx is None or ky is None:
        return None
    arity = _ARITY[kind]
    with _load_lock:
        if _free[arity]:
            k = _free[arity].pop()
        elif _next[arity] < NPOOL:
            k = _next[arity]
            _next[arity] += 1
        else:
            return None
    ctx = getattr(mod.lib, f"{arity}_ctx")[k]
    ctx.func = id(func)
    ctx.kx = kx
    ctx.ky = ky
    ctx.out_bool = 1 if out_bool else 0
    fn = getattr(mod.lib, f"{arity}_fn")[k]
    fnptr = gb_ffi.cast(_SIGNATURES[arity], int(mod.ffi.cast("uintptr_t", fn)))

    def release():
        with _load_lock:
            ctx.func = 0
            _free[arity].append(k)

    return fnptr, release


def reset_error():
    if _mod:
        _mod.lib.HEAP.errflag = 0


def finalize_release(op, release):
    """Free the pool entry when the typed op (and so its GrB op) is garbage."""
    weakref.finalize(op, release)
