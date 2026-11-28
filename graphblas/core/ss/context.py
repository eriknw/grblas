import threading

from ...exceptions import InvalidValue, check_status, check_status_carg
from .. import ffi, lib
from .config import BaseConfig

ffi_new = ffi.new


class Context(BaseConfig):
    """Set ``nthreads`` and ``chunk`` for the GraphBLAS calls on one thread.

    While a Context is engaged, GraphBLAS calls on the thread that engaged it use its
    settings in place of the global ones in ``gb.ss.config``; other threads keep theirs.

    .. code-block:: python

        import graphblas as gb

        A = gb.Matrix.from_coo([0, 1], [1, 0], [1.0, 2.0])
        with gb.ss.Context(nthreads=4):
            C = (A @ A).new()

    A new Context is engaged at once unless ``engage=False``. Leaving the ``with`` block,
    or ``disengage()``, engages the Context that was current before, so Contexts nest.
    Options left as None are copied from the current Context, or from the global settings
    when ``stack=False``. Read and change options as items: ``ctx["nthreads"] = 2``.
    For one operation, pass the options to it instead: ``(A @ A).new(nthreads=2)``.

    Config parameters
    -----------------
    nthreads : int
        Maximum number of OpenMP threads to use. 0 or negative copies the global
        setting, ``gb.ss.config["nthreads"]``.
    chunk : double
        The least work to give each thread, so a small problem runs on fewer threads
        (``floor(work / chunk)``, at most ``nthreads``). Below 1 means SuiteSparse's
        default, 65536.
    gpu_id : int
        Only for SuiteSparse:GraphBLAS 10.1 and earlier; see ``gb.ss.config``.
    """

    _context_keys = {"chunk", "gpu_id", "nthreads"}
    _options = {
        "chunk": (lib.GxB_CONTEXT_CHUNK, "double"),
        "nthreads": (lib.GxB_CONTEXT_NTHREADS, "int"),
    }
    # GxB_CONTEXT_GPU_ID existed in SS:GraphBLAS <=10.1 as a scalar int.
    # It was renamed to GxB_CONTEXT_GPU_IDS in 10.2 and changed to a list type.
    if hasattr(lib, "GxB_CONTEXT_GPU_ID"):
        _options["gpu_id"] = (lib.GxB_CONTEXT_GPU_ID, "int")
    _defaults = {
        "nthreads": 0,
        "chunk": 0,
        "gpu_id": -1,  # -1 means no GPU
    }

    def __init__(self, engage=True, *, stack=True, nthreads=None, chunk=None, gpu_id=None):
        # Set before super().__init__(), which fixes the allowed attributes (see BaseConfig).
        # Set _prev_context first: __del__ reads it if a later line raises.
        self._prev_context = None
        self.gb_obj = ffi_new("GxB_Context*")
        check_status_carg(lib.GxB_Context_new(self.gb_obj), "Context", self.gb_obj[0])
        super().__init__()
        if stack:
            context = threadlocal.context
            self["nthreads"] = context["nthreads"] if nthreads is None else nthreads
            self["chunk"] = context["chunk"] if chunk is None else chunk
            if "gpu_id" in self._options:
                self["gpu_id"] = context["gpu_id"] if gpu_id is None else gpu_id
        else:
            if nthreads is not None:
                self["nthreads"] = nthreads
            if chunk is not None:
                self["chunk"] = chunk
            if gpu_id is not None and "gpu_id" in self._options:
                self["gpu_id"] = gpu_id
        if engage:
            self.engage()

    @classmethod
    def _from_obj(cls, gb_obj=None):
        self = object.__new__(cls)
        self.gb_obj = gb_obj
        self._prev_context = None
        super().__init__(self)
        return self

    @property
    def _carg(self):
        return self.gb_obj[0]

    def dup(self, engage=True, *, nthreads=None, chunk=None, gpu_id=None):
        if nthreads is None:
            nthreads = self["nthreads"]
        if chunk is None:
            chunk = self["chunk"]
        if gpu_id is None and "gpu_id" in self._options:
            gpu_id = self["gpu_id"]
        return type(self)(engage, stack=False, nthreads=nthreads, chunk=chunk, gpu_id=gpu_id)

    def __del__(self):
        gb_obj = getattr(self, "gb_obj", None)
        if gb_obj is not None and lib is not None:  # pragma: no branch (safety)
            # Freeing an engaged Context is undefined behavior in GraphBLAS. A Context
            # that ``engage`` engaged has ``_prev_context`` set or is the current one
            # until ``disengage``.
            if self._prev_context is not None or threadlocal.context is self:
                try:
                    self.disengage()
                except InvalidValue:
                    pass
            elif gb_obj[0] != ffi.NULL:  # NULL would disengage any Context
                # A descriptor's Context is engaged only inside ``_call``. Disengage it
                # anyway, in case interrupts skipped both restores there. The status is
                # ignored: it is GrB_INVALID_VALUE, with no change, if another Context
                # is engaged.
                lib.GxB_Context_disengage(gb_obj[0])
            lib.GxB_Context_free(gb_obj)

    def engage(self):
        if self._prev_context is None and (context := threadlocal.context) is not self:
            self._prev_context = context
        check_status(lib.GxB_Context_engage(self._carg), self)
        threadlocal.context = self

    def _call(self, func, *args):
        """Return ``func(*args)`` with this Context engaged on this thread.

        The current Context (``threadlocal.context``: the global one, or a user's
        ``with Context(...)``) is engaged again afterwards, even if ``func`` raises.
        This Context is the current one while ``func`` runs, so a nested ``_call``
        restores it. A descriptor's Context is engaged only this way (see
        ``call_in_context``).
        """
        prev = threadlocal.context
        try:
            try:
                check_status(lib.GxB_Context_engage(self._carg), self)
                threadlocal.context = self
                return func(*args)
            finally:
                threadlocal.context = prev
                lib.GxB_Context_engage(prev._carg)
        finally:
            # Again, because PyPy can raise KeyboardInterrupt at the first bytecode of
            # the inner ``finally``, before its restore runs. Engaging ``prev`` twice is
            # harmless. Both restores ignore the status: ``prev`` is GxB_CONTEXT_WORLD
            # or a live Context, so engaging it fails only where GraphBLAS disables
            # Contexts (GxB_atfork_child), and there the first engage raised.
            threadlocal.context = prev
            lib.GxB_Context_engage(prev._carg)

    def disengage(self):
        prev_context = self._prev_context
        self._prev_context = None
        if threadlocal.context is self:
            if prev_context is not None:
                threadlocal.context = prev_context
                prev_context.engage()
            else:
                threadlocal.context = global_context
                check_status(lib.GxB_Context_disengage(self._carg), self)
        elif prev_context is not None and threadlocal.context is prev_context:
            prev_context.engage()
        else:
            check_status(lib.GxB_Context_disengage(self._carg), self)

    def __enter__(self):
        self.engage()
        return self

    def __exit__(self, exc_type, exc, exc_tb):
        self.disengage()

    @property
    def _context(self):
        return self

    @_context.setter
    def _context(self, val):
        if val is not None and val is not self:
            raise AttributeError("'_context' attribute is read-only")


class GlobalContext(Context):
    @property
    def _carg(self):
        return self.gb_obj

    def __del__(self):  # pragma: no cover (safety)
        pass


global_context = GlobalContext._from_obj(lib.GxB_CONTEXT_WORLD)


class ThreadLocal(threading.local):
    """Hold the active context for the current thread."""

    context = global_context


threadlocal = ThreadLocal()
