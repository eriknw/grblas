import gc
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


@pytest.mark.skipif(not supports_udfs, reason="first, second and any on a UDT need Numba")
def test_refuses_udf(A):
    with pytest.raises(KeyError, match="OBJECT"):
        A.apply(unary.register_anonymous(lambda x: x)).new()


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
    """Storing one object twice gives one handle."""
    t = Thing("x")
    v = Vector.from_coo([0, 1, 2], [t, t, Thing("y")], dtype=OBJECT)
    handles = objects._sorted_coo(v)[1]
    assert handles[0] == handles[1] != handles[2]


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
