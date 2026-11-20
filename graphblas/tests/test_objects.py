import gc
import importlib.util
import threading
import weakref

import numpy as np
import pytest

import graphblas as gb
from graphblas import (
    Matrix,
    Scalar,
    Vector,
    backend,
    binary,
    dtypes,
    monoid,
    select,
    semiring,
    unary,
)
from graphblas.core import _supports_udfs as supports_udfs
from graphblas.core import objects

OBJECT = dtypes.OBJECT
suitesparse = backend == "suitesparse"
if not suitesparse and not supports_udfs:
    # build needs a dup_op there, and ``any`` on a UDT needs Numba
    pytestmark = pytest.mark.skip("OBJECT needs Numba on the vanilla backend")


class Thing:
    """A weakref-able object, so tests can see when it is freed."""

    def __init__(self, label):
        self.label = label

    def __repr__(self):
        return f"Thing({self.label})"


@pytest.fixture
def things():
    return [Thing(i) for i in range(6)]


@pytest.fixture
def A(things):
    # 0 . 1 .
    # . 2 . .
    # 3 . . 4
    # . . . 5
    rows = [0, 0, 1, 2, 2, 3]
    cols = [0, 2, 1, 0, 3, 3]
    return Matrix.from_coo(rows, cols, things, dtype=object, nrows=4, ncols=4)


def _same(values, expected):
    return len(values) == len(expected) and all(
        x is y for x, y in zip(values, expected, strict=True)
    )


def test_lookup():
    """``object``, ``"object"`` and ``np.dtype(object)`` all name OBJECT."""
    assert dtypes.lookup_dtype(object) is OBJECT
    assert dtypes.lookup_dtype("object") is OBJECT
    assert dtypes.lookup_dtype(np.dtype(object)) is OBJECT
    assert OBJECT._is_udt
    assert OBJECT.np_type == objects.STORAGE
    with pytest.raises(ValueError, match="storage of OBJECT"):
        dtypes.register_anonymous(objects.STORAGE, "mine")


def test_roundtrip(A, things):
    """Values come back from ``to_coo`` as the same objects, in an object array."""
    *_, values = A.to_coo()
    assert values.dtype == object
    assert _same(values, things)
    v = Vector.from_coo([1, 3, 4], [None, (1, 2), [3]], dtype=object, size=5)
    idx, values = v.to_coo()
    assert idx.tolist() == [1, 3, 4]
    assert values[0] is None
    assert values[1] == (1, 2)
    assert values[2] == [3]
    if importlib.util.find_spec("pandas") is not None:  # the repr shows values with pandas
        assert "Thing(5)" in repr(A)


def test_elements(A, things):
    """Single elements read, write and delete as for other dtypes."""
    assert A[0, 2].new().value is things[1]
    assert A.get(0, 2) is things[1]
    assert A.get(0, 1, "missing") == "missing"
    new = Thing("new")
    A[1, 2] = new
    assert A.get(1, 2) is new
    del A[0, 0]
    assert (0, 0) not in A
    for is_cscalar in [False, True]:
        s = Scalar.from_value(things[3], dtype=OBJECT, is_cscalar=is_cscalar)
        assert s.value is things[3]
        assert s.isequal(things[3])
        assert not s.isequal(Thing(3))


def test_structural(A, things):
    """Ops that copy or select values keep the objects."""
    assert A.dup().isequal(A)
    B = A.T.new()
    assert B.get(2, 0) is things[1]
    assert A[[0, 2], :].new().get(1, 3) is things[4]
    assert A.select("tril").new().nvals == 4
    assert A.select(select.rowle, 1).new().nvals == 3
    C = Matrix(OBJECT, 4, 4)
    C[:, :] = A
    assert C.isequal(A)
    C = Matrix(OBJECT, 3, 3)
    C[:, :] = things[0]
    _, _, values = C.to_coo()
    assert _same(values, [things[0]] * 9)
    assert A.dup(mask=A.S).isequal(A)
    if supports_udfs:  # identity on a UDT needs Numba
        assert A.apply(unary.identity).new().isequal(A)
    assert A.mxm(A, semiring.any_pair).new().dtype == dtypes.INT64
    v = A.diag()
    assert _same(v.to_coo()[1], [things[0], things[2], things[5]])


@pytest.mark.skipif(not supports_udfs, reason="first, second and any on a UDT need Numba")
def test_select_ops(A, things):
    """``first``, ``second`` and ``any`` pick objects; ``mxm`` with ``any_first`` too."""
    B = Matrix.from_coo([0, 3], [0, 0], [Thing("b0"), Thing("b1")], dtype=OBJECT, nrows=4, ncols=4)
    C = A.ewise_add(B, binary.first).new()
    assert C.get(0, 0) is things[0]
    assert C.get(3, 0) is B.get(3, 0)
    C = A.ewise_mult(B, binary.second).new()
    assert C.nvals == 1
    assert C.get(0, 0) is B.get(0, 0)
    C = A.ewise_add(B, binary.any).new()
    assert C.get(0, 0) in {things[0], B.get(0, 0)}
    C = A.mxm(A, semiring.any_first).new()
    assert C.get(0, 0) in {things[0], things[1]}  # A[0, 0] A[0, 0] or A[0, 2] A[2, 0]
    assert C.get(3, 3) is things[5]
    assert A.reduce_rowwise(monoid.any).new().get(3) is things[5]


@pytest.mark.skipif("not suitesparse")
def test_ss_structural(A):
    """concat, split and reshape copy handles."""
    B = gb.ss.concat([[A, A]])
    assert B[:, 4:].new().isequal(A)
    tiles = A.ss.split(2)
    assert gb.ss.concat(tiles).isequal(A)
    assert A.ss.reshape(2, 8).ss.reshape(4, 4).isequal(A)


def test_refuses_value_ops(A):
    """Ops that would read handles as numbers raise KeyError."""
    with pytest.raises(KeyError, match="OBJECT"):
        A.ewise_add(A, binary.plus).new()
    with pytest.raises(KeyError, match="OBJECT"):
        A.ewise_mult(A, binary.eq).new()
    with pytest.raises(KeyError, match="OBJECT"):
        A.reduce_rowwise(monoid.min).new()
    with pytest.raises(KeyError, match="OBJECT"):
        A.mxm(A, semiring.plus_times).new()
    with pytest.raises(KeyError):
        A.select(select.valueeq, 1).new()
    with pytest.raises(KeyError, match="OBJECT"):
        A.apply(unary.ainv).new()


def test_isequal_is_identity():
    """Isequal compares objects by identity, not ``==``."""
    a = [[1], [2]]
    b = [[1], [2]]  # equal, not identical
    v1 = Vector.from_coo([0, 1], a, dtype=OBJECT)
    v2 = Vector.from_coo([0, 1], a, dtype=OBJECT)
    v3 = Vector.from_coo([0, 1], b, dtype=OBJECT)
    assert v1.isequal(v2)
    assert v1.isclose(v2)
    assert not v1.isequal(v3)
    assert not v1.isequal(Vector.from_coo([0, 1], [1, 2]))


def test_interning():
    """One value is interned by identity; an array gets a handle per occurrence."""
    t = Thing("x")
    v = Vector.from_coo([0, 1, 2], [t, t, Thing("y")], dtype=OBJECT)
    handles = objects._sorted_coo(v)[1]
    assert len(set(handles.tolist())) == 3
    assert v.isequal(Vector.from_coo([0, 1, 2], [t, t, v.get(2)], dtype=OBJECT))
    s1 = Scalar.from_value(t, dtype=OBJECT)
    s2 = Scalar.from_value(t, dtype=OBJECT)
    assert objects._stored_handles(s1)[0] == objects._stored_handles(s2)[0]


def test_collect_frees_unreferenced(things):
    """An object no container holds is freed by ``collect``; one a container holds is not."""
    freed = []
    for t in things:
        weakref.finalize(t, freed.append, t.label)
    del t
    keep = Vector.from_coo([0], [things[0]], dtype=OBJECT, size=3)
    drop = Vector.from_coo(range(6), things, dtype=OBJECT)
    things.clear()
    del drop
    gc.collect()
    objects.collect()
    assert sorted(freed) == [1, 2, 3, 4, 5]
    assert keep.get(0).label == 0


def test_stale_handle_is_detected():
    """A handle whose object was freed fails to resolve; so does zero."""
    t = Thing("stale")
    v = Vector.from_coo([0], [t], dtype=OBJECT)
    handles = objects._sorted_coo(v)[1]
    del v, t
    objects.collect()
    with pytest.raises(objects.InvalidHandle):
        objects.to_objects(handles.view(objects.STORAGE))
    with pytest.raises(objects.InvalidHandle):
        objects.to_objects(np.zeros(1, dtype=objects.STORAGE))


def test_automatic_collection(monkeypatch):
    """Allocation past the threshold collects, so a mutation loop stays bounded."""
    monkeypatch.setattr(objects, "threshold", 100)
    v = Vector(OBJECT, 10)
    before = objects.stats()["collections"]
    for i in range(2000):
        v[i % 10] = Thing(i)
    assert objects.stats()["collections"] > before
    assert objects.stats()["live"] < 1000
    assert v.get(9).label == 1999


def test_deferred_inside_shared_section():
    """A collection requested while the lock is held shared runs at the release."""
    v = Vector(OBJECT, 3)
    before = objects.stats()["collections"]
    with objects._lock:
        assert objects.collect() == 0
        assert objects._lock.requested
        v[0] = Thing("inside")
        assert objects.stats()["collections"] == before
    assert not objects._lock.requested
    assert objects.stats()["collections"] == before + 1
    assert v.get(0).label == "inside"


def test_threads():
    """Threads that store, read and collect at once see their own objects."""
    errors = []

    def work(seed):
        try:
            rng = np.random.default_rng(seed)
            for _ in range(30):
                vals = [Thing((seed, i)) for i in range(20)]
                v = Vector.from_coo(range(20), vals, dtype=OBJECT)
                v[int(rng.integers(2, 20))] = vals[0]
                if rng.random() < 0.3:
                    objects.collect()
                _, got = v.to_coo()
                assert all(g is vals[0] or g in vals for g in got)
                assert got[1] is vals[1]
        except Exception as exc:  # pragma: no cover (failure)
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors


def test_carrier_survives_collection():
    """Handles from ``to_handles`` stay valid while any view of them lives.

    The tracked array must own the data: numpy points a view of a view at the
    owner, so a tracked view in between would die at once.
    """
    h = objects.to_handles([Thing("carried")]).view(np.uint8).view(objects.STORAGE)
    objects.collect()
    assert objects.to_objects(h)[0].label == "carried"


def test_pickle(A, things):
    """Pickling stores the objects, not their handles."""
    import pickle

    B = pickle.loads(pickle.dumps(A))
    assert B.dtype is OBJECT
    assert [x.label for x in B.to_coo()[2]] == [t.label for t in things]
    v = pickle.loads(pickle.dumps(A.diag()))
    assert [x.label for x in v.to_coo()[1]] == [0, 2, 5]
    assert pickle.loads(pickle.dumps(OBJECT)) is OBJECT


@pytest.mark.skipif("not suitesparse")
def test_export_import(A):
    """An export holds handles; importing them while the objects live gives the objects."""
    B = Matrix.ss.import_any(**A.ss.export())
    assert B.isequal(A)


# Python functions on OBJECT values, run inside GraphBLAS (``operator.pyobject``)

# A UDF registered with is_udt=True needs no Numba to run on OBJECT values
pyops = pytest.mark.skipif(
    not suitesparse and not supports_udfs, reason="OBJECT needs Numba on the vanilla backend"
)


@pytest.fixture
def F():
    from fractions import Fraction

    return Fraction


@pytest.fixture
def FA(F):
    return Matrix.from_coo(
        [0, 0, 1, 2],
        [0, 1, 1, 2],
        [F(1, 2), F(1, 3), F(2, 3), F(5)],
        dtype=OBJECT,
        nrows=3,
        ncols=3,
    )


@pyops
def test_python_ops_elementwise(F, FA):
    """apply, ewise_add, ewise_mult and ewise_union call the function on the objects."""
    add = binary.register_anonymous(lambda x, y: x + y, is_udt=True)
    neg = unary.register_anonymous(lambda x: -x, is_udt=True)
    assert FA.apply(neg).new().to_coo()[2].tolist() == [F(-1, 2), F(-1, 3), F(-2, 3), F(-5)]
    assert FA.apply(add, right=F(1)).new().get(2, 2) == F(6)
    assert FA.apply(add, right=10).new().get(2, 2) == F(15)  # a number stays a number
    C = FA.ewise_add(FA.T, add).new()
    assert C.dtype is OBJECT
    assert C.get(0, 0) == F(1)
    assert C.get(1, 0) == F(1, 3)  # one input only: copied, not added
    assert FA.ewise_mult(FA, add).new().get(1, 1) == F(4, 3)
    C = FA.ewise_union(FA.T, add, F(100), F(1000)).new()
    assert C.get(0, 1) == F(3001, 3)
    if supports_udfs:
        # A UDF registered without is_udt (Numba types it for numbers) also works on OBJECT
        plain = binary.register_anonymous(lambda x, y: x * y)
        assert FA.ewise_mult(FA, plain).new().get(2, 2) == F(25)


@pyops
def test_python_ops_reduce_mxm(F, FA):
    """Monoids and semirings made of Python functions reduce and multiply."""
    add = binary.register_anonymous(lambda x, y: x + y, is_udt=True)
    mul = binary.register_anonymous(lambda x, y: x * y, is_udt=True)

    def least(x, y):
        return min(x, y)

    least = binary.register_anonymous(least, is_udt=True)
    plus = monoid.register_anonymous(add, F(0))
    assert FA.reduce_rowwise(plus).new().to_coo()[1].tolist() == [F(5, 6), F(2, 3), F(5)]
    assert FA.reduce_scalar(plus).new().value == F(13, 2)
    smallest = monoid.register_anonymous(least, F(10**9))
    s = FA.reduce_scalar(smallest).new().value
    assert s == F(1, 3)
    assert s is FA.get(0, 1)  # the input's handle is reused: the same object
    plus_times = semiring.register_anonymous(plus, mul)
    v = Vector.from_coo([0, 1, 2], [F(1), F(2), F(3)], dtype=OBJECT)
    assert FA.mxv(v, plus_times).new().to_coo()[1].tolist() == [F(7, 6), F(4, 3), F(15)]
    assert FA.mxm(FA, plus_times).new().get(0, 1) == F(7, 18)


@pyops
def test_python_ops_select(F, FA):
    """A SelectOp on OBJECT gives BOOL; an IndexUnaryOp gives OBJECT."""
    big = select.register_anonymous(lambda x, i, j, t: x > t, is_udt=True)
    assert FA.select(big, F(1, 2)).new().to_coo()[2].tolist() == [F(2, 3), F(5)]
    assert FA.select(big, 1).new().nvals == 1
    where = gb.indexunary.register_anonymous(lambda x, i, j, t: (i, j, x), is_udt=True)
    assert FA.apply(where, 0).new().get(1, 1) == (1, 1, F(2, 3))


@pyops
def test_python_ops_errors(F, FA):
    """The first exception in a Python op is raised after GraphBLAS returns."""
    bad = binary.register_anonymous(lambda x, y: x / (y - y), is_udt=True)
    with pytest.raises(ZeroDivisionError):
        FA.ewise_mult(FA, bad).new()
    add = binary.register_anonymous(lambda x, y: x + y, is_udt=True)
    assert FA.ewise_mult(FA, add).new().nvals == 4  # the next call is clean
    calls = []

    def count(x, y):
        calls.append(1)
        raise ValueError("once")

    once = binary.register_anonymous(count, is_udt=True)
    with pytest.raises(ValueError, match="once"):
        FA.ewise_mult(FA, once).new()
    assert len(calls) == 1  # later elements do not call the function


@pyops
def test_python_ops_collect(F, FA):
    """Objects a Python op makes survive while stored, and intermediates are freed."""
    freed = []

    class Tracked:
        def __init__(self, x):
            self.x = x
            weakref.finalize(self, freed.append, x)

    wrap = unary.register_anonymous(lambda x: Tracked(x), is_udt=True)  # noqa: PLW0108
    unwrap = unary.register_anonymous(lambda t: t.x, is_udt=True)
    tmp = FA.apply(wrap).new()
    result = tmp.apply(unwrap).new()
    del tmp
    gc.collect()
    objects.collect()
    assert len(freed) == 4  # the Tracked intermediates
    assert result.isequal(FA)  # unwrap returns the stored Fractions, the same objects
    kept = FA.apply(wrap).new()
    objects.collect()
    assert len(freed) == 4
    assert [t.x for t in kept.to_coo()[2]] == FA.to_coo()[2].tolist()


@pyops
def test_python_ops_collect_during_call(monkeypatch):
    """A collection due during a Python-op call runs after it, and keeps its results."""
    monkeypatch.setattr(objects, "threshold", 10)
    v = Vector.from_coo(range(200), list(range(200)), dtype=OBJECT)
    before = objects.stats()["collections"]
    box = unary.register_anonymous(lambda x: [x], is_udt=True)
    w = v.apply(box).new()  # 200 allocations inside one call
    assert objects.stats()["collections"] > before
    assert [x[0] for x in w.to_coo()[1]] == list(range(200))


@pyops
def test_python_ops_cffi_fallback(F, FA, monkeypatch):
    """Without the C module or Numba, Python ops run as cffi callbacks, with the same results."""
    from graphblas.core.operator import _cshim, _numba_capi, pyobject

    monkeypatch.setattr(_cshim, "_mod", False)
    monkeypatch.setattr(_numba_capi, "_impl", False)
    add = binary.register_anonymous(lambda x, y: x + y, is_udt=True)
    C = FA.ewise_add(FA.T, add).new()
    assert pyobject._callbacks[id(add[OBJECT])][0] == "cffi"
    assert C.get(0, 0) == F(1)
    bad = binary.register_anonymous(lambda x, y: 1 / 0, is_udt=True)
    with pytest.raises(ZeroDivisionError):
        FA.ewise_mult(FA, bad).new()


@pyops
@pytest.mark.parametrize("module", ["_cshim", "_numba_capi"])
def test_python_ops_c_pool_reuse(FA, module):
    """A trampoline is reused once its op is garbage, so making ops in a loop does not run out."""
    from graphblas.core.operator import _cshim, _numba_capi, pyobject

    if module == "_cshim" and _cshim.load() is None:
        pytest.skip("no C compiler")
    if module == "_numba_capi" and (_cshim.load() is not None or not _numba_capi.load()):
        pytest.skip("Numba trampolines are used only without a C compiler, with Numba")
    mod = _cshim if module == "_cshim" else _numba_capi
    for i in range(mod.NPOOL + 10):
        op = unary.register_anonymous(lambda x, i=i: (x, i), is_udt=True)
        assert pyobject._callbacks[id(op[OBJECT])][0] == "c"
        assert FA.apply(op).new().get(2, 2)[1] == i
        del op
        if i % 100 == 0:
            gc.collect()  # an op and its typed ops form a cycle


@pyops
def test_python_ops_numba_capi(F, FA):
    """Without a C compiler, Python ops run in Numba-compiled trampolines that call the C API."""
    from graphblas.core.operator import _cshim, _numba_capi

    if _cshim.load() is not None or not _numba_capi.load():
        pytest.skip("Numba trampolines are used only without a C compiler, with Numba")
    add = binary.register_anonymous(lambda x, y: x + y, is_udt=True)
    C = FA.ewise_add(FA.T, add).new()
    assert any(f is add.orig_func for f in _numba_capi._funcs.values())
    assert C.get(0, 0) == F(1)
    for value, dtype in [
        (1.5, "FP64"),
        (0.5, "FP32"),
        (True, "BOOL"),
        (255, "UINT8"),
        (-3, "INT8"),
    ]:
        right = Scalar.from_value(value, dtype=dtype)  # each built-in input type
        assert FA.apply(add, right=right).new().get(2, 2) == F(5) + value
    bad = binary.register_anonymous(lambda x, y: 1 / 0, is_udt=True)
    with pytest.raises(ZeroDivisionError):
        FA.ewise_mult(FA, bad).new()
    # A stale handle (imported after its object was freed) fails in the trampoline
    v = Vector.from_coo([0, 1], [Thing(0), Thing(1)], dtype=OBJECT)
    stale = v.ss.export(raw=True)["values"].copy()
    del v
    objects.collect()
    w = Vector.from_coo([0, 1], [Thing(2), Thing(3)], dtype=OBJECT)
    parts = w.ss.export(raw=True)
    parts["values"] = stale
    same = unary.register_anonymous(lambda x: x, is_udt=True)
    with pytest.raises(objects.InvalidHandle):
        Vector.ss.import_any(**parts).apply(same).new()


@pyops
def test_python_ops_any_callable(F, FA):
    """With is_udt=True, builtins, partials, classes and other callables are UDFs on OBJECT."""
    import functools
    import operator

    add = binary.register_anonymous(operator.add, is_udt=True)
    assert FA.ewise_mult(FA, add).new().get(2, 2) == F(10)
    least = binary.register_anonymous(min, is_udt=True)
    assert FA.ewise_mult(FA, least).new().get(0, 1) is FA.get(0, 1)
    plus3 = unary.register_anonymous(functools.partial(operator.add, 3), is_udt=True)
    assert FA.apply(plus3).new().get(2, 2) == F(8)
    to_str = unary.register_anonymous(str, is_udt=True)
    assert FA.apply(to_str).new().get(1, 1) == "2/3"
    plus = monoid.register_anonymous(add, F(0))
    assert FA.reduce_scalar(plus).new().value == F(13, 2)
    if supports_udfs:
        with pytest.raises(TypeError, match="must be a function"):
            binary.register_anonymous(operator.add)  # Numba needs a Python function


@pyops
@pytest.mark.skipif("not suitesparse")
def test_empty_iso_value_is_kept():
    """A collection keeps the iso value of an empty iso Matrix: SuiteSparse calls ops on it."""
    A = Matrix(OBJECT, 1, 0)
    A[:, :] = Thing("iso")
    objects.collect()
    second = binary.register_anonymous(lambda x, y: y, is_udt=True)
    if A.ss.is_iso:
        assert A.ewise_add(A, second).new().nvals == 0


@pyops
def test_python_op_inside_python_op(F, FA):
    """A Python op may run a GraphBLAS call with a Python op on its own thread."""
    neg = unary.register_anonymous(lambda x: -x, is_udt=True)

    def outer(x):
        v = Vector.from_coo([0], [x], dtype=OBJECT)
        return v.apply(neg).new().get(0)

    op = unary.register_anonymous(outer, is_udt=True)
    if not suitesparse:  # it cannot ask for one thread there
        with pytest.raises(RuntimeError, match="several threads"):
            FA.apply(op).new()
        return
    # One thread for the outer call (without the GIL, the default is several)
    assert FA.apply(op).new(nthreads=1).get(2, 2) == F(-5)


# Review fixes: each test fails without its fix

_FT = not getattr(__import__("sys"), "_is_gil_enabled", lambda: True)()


@pyops
@pytest.mark.skipif(
    not _FT, reason="a call runs Python ops on several threads only without the GIL"
)
def test_nested_python_op_on_several_threads():
    """A Python op that runs a Python-op call inside a multithreaded call raises, not hangs."""
    neg = unary.register_anonymous(lambda x: -x, is_udt=True)

    def outer(x):
        return Vector.from_coo([0], [x], dtype=OBJECT).apply(neg).new().get(0)

    op = unary.register_anonymous(outer, is_udt=True)
    v = Vector.from_coo(range(200_000), list(range(200_000)), dtype=OBJECT)
    v[:1].new().apply(op).new(nthreads=1)  # type the op
    from graphblas.core.operator import pyobject

    if pyobject._callbacks[id(op[OBJECT])][0] == "cffi":
        pytest.skip("a cffi call always runs on one thread")
    result = []

    def work():
        try:
            v.apply(op).new(nthreads=8)
            result.append("ok")
        except RuntimeError as exc:
            result.append(exc)

    t = threading.Thread(target=work, daemon=True)
    t.start()
    t.join(60)
    assert result, "hung"
    assert isinstance(result[0], RuntimeError)


@pyops
@pytest.mark.parametrize("cshim", [True, False])
def test_handle_zero_in_nonempty_input(cshim, monkeypatch):
    """Handle 0 left by a failed ``<<`` raises when a Python op reads it."""
    from graphblas.core.operator import _cshim, _numba_capi

    if not cshim:
        monkeypatch.setattr(_cshim, "_mod", False)
        monkeypatch.setattr(_numba_capi, "_impl", False)
    C = Vector.from_coo([0, 1, 2], [1, 2, 3], dtype=OBJECT)
    v = Vector.from_coo([0, 1, 2], [1, 0, 2], dtype=OBJECT)
    with pytest.raises(ZeroDivisionError):
        C << v.apply(unary.register_anonymous(lambda x: 1 / x, is_udt=True))
    keep = select.register_anonymous(lambda x, i, j, t: True, is_udt=True)
    with pytest.raises(objects.InvalidHandle):
        C.select(keep, 0).new()
    with pytest.raises(objects.InvalidHandle):
        C.apply(unary.register_anonymous(lambda x: x, is_udt=True)).new()


@pyops
@pytest.mark.parametrize("cshim", [True, False])
def test_python_op_entries_are_freed(cshim, monkeypatch):
    """Dropping a Python op frees its function and what it closes over."""
    from graphblas.core.operator import _cshim, _numba_capi, pyobject

    if not cshim:
        monkeypatch.setattr(_cshim, "_mod", False)
        monkeypatch.setattr(_numba_capi, "_impl", False)
    freed = []
    held = Thing("closed over")
    weakref.finalize(held, freed.append, 1)

    def f(x, held=held):
        return x

    op = unary.register_anonymous(f, is_udt=True)
    Vector.from_coo([0], [1], dtype=OBJECT).apply(op).new()
    key = id(op[OBJECT])
    assert key in pyobject._callbacks
    del op, f, held
    gc.collect()
    assert freed == [1]
    assert key not in pyobject._callbacks


@pyops
@pytest.mark.parametrize("exc_type", [SystemExit, GeneratorExit])
def test_cffi_base_exceptions(exc_type, monkeypatch):
    """SystemExit and GeneratorExit from a function on the cffi path are raised after the call."""
    from graphblas.core.operator import _cshim, _numba_capi

    monkeypatch.setattr(_cshim, "_mod", False)
    monkeypatch.setattr(_numba_capi, "_impl", False)

    def f(x):
        raise exc_type

    op = unary.register_anonymous(f, is_udt=True)
    with pytest.raises(exc_type):
        Vector.from_coo([0, 1], [1, 2], dtype=OBJECT).apply(op).new()


@pyops
def test_sigint_goes_to_the_error_slot():
    """Ctrl-C during a Python-op call is raised after the call, whichever frame it lands in."""
    import signal

    if threading.current_thread() is not threading.main_thread():
        pytest.skip("signals go to the main thread")
    calls = []

    def f(x):
        calls.append(x)
        if len(calls) == 2:
            handler = signal.getsignal(signal.SIGINT)
            assert handler is not signal.default_int_handler
            handler(signal.SIGINT, None)  # as the interpreter would, at any later point
        return x

    op = unary.register_anonymous(f, is_udt=True)
    with pytest.raises(KeyboardInterrupt):
        Vector.from_coo(range(50), list(range(50)), dtype=OBJECT).apply(op).new()
    assert len(calls) == 2
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler


def test_from_coo_one_object():
    """For OBJECT, a value that is not a list, tuple or array is one object for every entry."""
    from fractions import Fraction

    v = Vector.from_coo([0, 1, 2], "abc", dtype=OBJECT)
    assert v.to_coo()[1].tolist() == ["abc"] * 3
    half = Fraction(1, 2)
    A = Matrix.from_coo([0, 1], [1, 0], half, dtype=OBJECT, nrows=2, ncols=2)
    assert all(x is half for x in A.to_coo()[2])
    w = Vector.from_coo([0, 1], (1, 2), dtype=OBJECT)  # a tuple is a sequence, as in numpy
    assert w.to_coo()[1].tolist() == [1, 2]


@pyops
@pytest.mark.skipif("not suitesparse")
def test_one_thread_keeps_descriptor_options():
    """With the GIL, a Python-op call with descriptor options still runs on one thread."""
    from graphblas.core.descriptor import lookup
    from graphblas.core.operator import pyobject

    desc = lookup(axb_method="dot", mask_structure=True)
    one = pyobject._one_thread(desc)
    config = pyobject._desc_config(one)
    assert config["nthreads"] == 1
    assert config["axb_method"] == "dot"
    assert config["mask_structure"]


@pyops
def test_build_with_python_dup_op():
    """Build and from_coo take a Python dup_op."""
    first = binary.register_anonymous(lambda x, y: x, is_udt=True)
    a, b = Thing("a"), Thing("b")
    v = Vector.from_coo([0, 0, 1], [a, b, b], dtype=OBJECT, dup_op=first)
    assert v.get(0) in {a, b}
    assert v.get(1) is b


def test_subclass_takes_the_lock():
    """A call on a subclass of Vector with OBJECT values holds the collector's lock."""

    class MyVector(Vector):
        __slots__ = ()

    v = MyVector.from_coo([0], [Thing("x")], dtype=OBJECT)
    assert type(v) is MyVector
    assert objects.touches_object([v])


@pyops
def test_python_op_calls_from_several_threads():
    """Python-op calls in several Python threads at once take turns; none raises."""
    add = binary.register_anonymous(lambda x, y: x + y, is_udt=True)
    errors = []

    def work(k):
        try:
            v = Vector.from_coo(range(1000), list(range(1000)), dtype=OBJECT)
            for _ in range(20):
                assert v.ewise_mult(v, add).new().get(999) == 1998
        except Exception as exc:  # pragma: no cover (failure)
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(k,)) for k in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors


@pyops
def test_python_op_waiting_on_another_threads_call(monkeypatch):
    """A Python op that waits for another thread's Python-op call raises instead of hanging."""
    from concurrent.futures import ThreadPoolExecutor

    from graphblas.core.operator import pyobject

    monkeypatch.setattr(pyobject, "wait_timeout", 1.0)
    neg = unary.register_anonymous(lambda x: -x, is_udt=True)
    pool = ThreadPoolExecutor(1)

    def inner(x):
        return Vector.from_coo([0], [x], dtype=OBJECT).apply(neg).new().get(0)

    op = unary.register_anonymous(lambda x: pool.submit(inner, x).result(), is_udt=True)
    v = Vector.from_coo([0, 1], [10, 20], dtype=OBJECT)
    result = []

    def work():
        try:
            v.apply(op).new(**({"nthreads": 1} if suitesparse else {}))
            result.append("ok")
        except RuntimeError as exc:
            result.append(exc)

    t = threading.Thread(target=work, daemon=True)
    t.start()
    t.join(30)
    pool.shutdown(wait=False)
    assert result, "hung"
    assert isinstance(result[0], RuntimeError)
    assert "another thread" in str(result[0])


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (np.array([1, 2, 3]), [1, 2, 3]),
        (np.array([0.5, 1.5, 2.5]), [0.5, 1.5, 2.5]),
        (range(3), [0, 1, 2]),
        (__import__("collections").deque(["a", "b", "c"]), ["a", "b", "c"]),
        ("abc", ["abc"] * 3),
        (b"xy", [b"xy"] * 3),
    ],
)
def test_from_coo_values_kinds(values, expected):
    """For OBJECT, a numeric array and any sequence but str or bytes give one value per index."""
    v = Vector.from_coo([0, 1, 2], values, dtype=OBJECT)
    assert v.to_coo()[1].tolist() == expected


@pytest.mark.parametrize("values", [{1, 2, 3}, {0: 1, 1: 2, 2: 3}, (x for x in range(3))])
def test_from_coo_values_unordered(values):
    """A set, dict or generator as OBJECT values raises: one value per index or one object?"""
    with pytest.raises(TypeError, match="list"):
        Vector.from_coo([0, 1, 2], values, dtype=OBJECT)


def test_from_coo_none_value():
    """None as the one value of every entry."""
    v = Vector.from_coo([0, 2], None, dtype=OBJECT, size=3)
    assert v.nvals == 2
    assert v.to_coo()[1].tolist() == [None, None]


def test_sigint_while_reporting_an_error():
    """Ctrl-C landing while an error is being recorded does not deadlock."""
    from graphblas.core.operator import _cshim, pyobject

    done = []

    def work():
        with pyobject._error_lock:  # as a trampoline reporting on the main thread
            pyobject._on_sigint(2, None)
        done.append(1)

    t = threading.Thread(target=work, daemon=True)
    t.start()
    t.join(10)
    pyobject._error = None
    _cshim.exit_call(_cshim.enter_call(False))  # clear the C flag it set
    if _cshim._mod:
        _cshim._mod.lib.HEAP.errflag = 0
    assert done, "deadlocked"


def test_empty_iso_subclass_argument():
    """An empty iso input of a subclass type counts, even after another OBJECT argument."""

    class MyMatrix(Matrix):
        __slots__ = ()

    A = Matrix.from_coo([0], [0], [Thing("a")], dtype=OBJECT, nrows=1, ncols=1)
    E = Matrix(OBJECT, 1, 0)
    E[:, :] = Thing("e")
    if suitesparse and not E.ss.is_iso:
        pytest.skip("not iso")
    sub = MyMatrix._from_obj(E.gb_obj, OBJECT, 1, 0, parent=E)
    assert objects.has_empty_iso([A, sub])


@pyops
def test_collect_loop_does_not_starve_calls():
    """A thread calling collect() in a loop lets other threads' calls take turns."""
    import time

    neg = unary.register_anonymous(lambda x: -x, is_udt=True)
    v = Vector.from_coo(range(100), list(range(100)), dtype=OBJECT)
    v.apply(neg).new()  # make the op first (the first C op attaches the heap)
    stop = time.perf_counter() + 2

    def collector():
        while time.perf_counter() < stop:
            objects.collect()

    t = threading.Thread(target=collector)
    t.start()
    rounds = 0
    while time.perf_counter() < stop:
        v.apply(neg).new()
        rounds += 1
    t.join()
    assert rounds >= 20


@pytest.mark.parametrize("kind", ["series", "dict_values", "custom"])
def test_from_coo_values_array_likes(kind):
    """An array-like gives a value per index; another non-sequence iterable raises."""
    if kind == "series":
        pd = pytest.importorskip("pandas")
        v = Vector.from_coo([0, 1, 2], pd.Series(["a", "b", "c"]), dtype=OBJECT)
        assert v.to_coo()[1].tolist() == ["a", "b", "c"]
        return

    class Bag:
        def __iter__(self):
            return iter([1, 2, 3])

        def __len__(self):
            return 3

    values = {0: 1, 1: 2, 2: 3}.values() if kind == "dict_values" else Bag()
    with pytest.raises(TypeError, match="list"):
        Vector.from_coo([0, 1, 2], values, dtype=OBJECT)


@pyops
def test_python_op_call_waits_for_a_long_call(monkeypatch):
    """A call waits for another thread's long call while that call makes progress."""
    import time

    from graphblas.core.operator import pyobject

    monkeypatch.setattr(pyobject, "wait_timeout", 0.5)

    def slow(x):
        time.sleep(0.002)  # releases the GIL, so the other thread runs meanwhile
        return (x,)  # a new object: the Numba trampolines show progress only by claims

    slow_op = unary.register_anonymous(slow, is_udt=True)
    neg = unary.register_anonymous(lambda x: -x, is_udt=True)
    big = Vector.from_coo(range(1000), list(range(1000)), dtype=OBJECT)  # about 2 s
    small = Vector.from_coo([0], [1], dtype=OBJECT)
    small.apply(neg).new()
    started = threading.Event()
    results = []

    def long_call():
        started.set()
        big.apply(slow_op).new(**({"nthreads": 1} if suitesparse else {}))

    t = threading.Thread(target=long_call)
    t.start()
    started.wait()
    time.sleep(0.2)
    try:
        results.append(small.apply(neg).new().get(0))
    except RuntimeError as exc:  # pragma: no cover (failure)
        results.append(exc)
    t.join()
    assert results == [-1]
