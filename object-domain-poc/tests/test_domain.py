import gc
import weakref

import numpy as np
import pytest

import graphblas as gb
from object_domain_poc import ObjectDomain, ObjectMatrix, ObjectVector


class Payload:
    def __init__(self, value):
        self.value = value


def collect():
    gc.collect()
    gc.collect()


def test_domain_interns_by_identity_with_nonzero_uint64_handles():
    domain = ObjectDomain()
    first = Payload("equal")
    second = Payload("equal")

    first_handle = domain.intern(first)

    assert first_handle == 1
    assert domain.intern(first) == first_handle
    assert domain.intern(second) != first_handle
    assert domain.resolve(np.uint64(first_handle)) is first
    assert domain.contains_handle(first_handle)
    assert not domain.contains_handle(0)
    with pytest.raises(KeyError, match="does not belong"):
        domain.resolve(0)


def test_matrix_uses_native_uint64_storage_and_supports_none():
    domain = ObjectDomain()
    value = Payload("value")
    matrix = ObjectMatrix.from_coo(
        domain,
        [2, 0],
        [1, 3],
        [value, None],
        nrows=3,
        ncols=4,
    )

    assert type(matrix.native) is gb.Matrix
    assert matrix.native.dtype == gb.dtypes.UINT64
    assert matrix.shape == matrix.native.shape == (3, 4)
    assert matrix.nvals == matrix.native.nvals == 2
    assert matrix[2, 1] is value
    assert matrix[0, 3] is None
    assert (0, 3) in matrix
    assert matrix.get(1, 1, "absent") == "absent"
    with pytest.raises(KeyError):
        matrix[1, 1]
    rows, columns, values = matrix.to_coo()
    extracted = {(int(row), int(column)): item for row, column, item in zip(
        rows, columns, values, strict=True
    )}
    assert extracted[(0, 3)] is None
    assert extracted[(2, 1)] is value


def test_matrix_stores_string_edge_labels_as_domain_handles():
    domain = ObjectDomain()
    matrix = ObjectMatrix.from_coo(
        domain,
        rows=[0, 0, 1, 2],
        columns=[1, 2, 2, 0],
        values=["follows", "likes", "blocks", "mentions"],
        nrows=3,
        ncols=3,
    )

    assert matrix[0, 1] == "follows"
    assert matrix[0, 2] == "likes"
    assert matrix[1, 2] == "blocks"
    assert matrix[2, 0] == "mentions"

    rows, columns, strings = matrix.to_coo()
    extracted = dict(zip(zip(rows.tolist(), columns.tolist(), strict=True), strings, strict=True))
    assert extracted == {
        (0, 1): "follows",
        (0, 2): "likes",
        (1, 2): "blocks",
        (2, 0): "mentions",
    }

    _, _, handles = matrix.native.to_coo()
    assert handles.dtype == np.dtype(np.uint64)
    assert all(domain.resolve(handle) == extracted[key] for key, handle in zip(
        zip(rows.tolist(), columns.tolist(), strict=True), handles, strict=True
    ))


def test_string_matrix_assignment_and_native_duplication():
    domain = ObjectDomain()
    matrix = ObjectMatrix.from_coo(domain, [], [], nrows=2, ncols=2)

    matrix[0, 0] = "New York"
    matrix[0, 1] = "San Francisco"
    matrix[1, 0] = ""
    duplicate = matrix.dup()
    matrix[0, 0] = "Boston"

    assert matrix[0, 0] == "Boston"
    assert duplicate[0, 0] == "New York"
    assert duplicate[0, 1] == "San Francisco"
    assert duplicate[1, 0] == ""
    assert duplicate.domain is domain
    assert duplicate.native.dtype == gb.dtypes.UINT64


def test_vector_uses_native_uint64_storage():
    domain = ObjectDomain()
    value = Payload("value")
    vector = ObjectVector.from_coo(domain, [2], [value], size=5)

    assert type(vector.native) is gb.Vector
    assert vector.native.dtype == gb.dtypes.UINT64
    assert vector.size == vector.native.size == 5
    assert vector.shape == (5,)
    assert vector.nvals == 1
    indices, values = vector.to_coo()
    assert indices.tolist() == [2]
    assert values == [value]


def test_native_matrix_is_authoritative():
    domain = ObjectDomain()
    first = Payload("first")
    second = Payload("second")
    matrix = ObjectMatrix.from_coo(domain, [0], [0], [first], nrows=2, ncols=2)
    second_handle = domain.intern(second)

    matrix.native[1, 1] = second_handle
    del matrix.native[0, 0]

    assert matrix.shape == (2, 2)
    assert matrix.nvals == 1
    assert (0, 0) not in matrix
    assert matrix[1, 1] is second
    assert not hasattr(matrix, "_data")


def test_native_vector_is_authoritative():
    domain = ObjectDomain()
    value = Payload("value")
    vector = ObjectVector.from_coo(domain, [], [], size=3)

    vector.native[1] = domain.intern(value)

    assert vector.nvals == 1
    assert vector[1] is value
    assert not hasattr(vector, "_data")


def test_assignment_deletion_clear_and_resize_defer_to_native_objects():
    domain = ObjectDomain()
    matrix = ObjectMatrix.from_coo(domain, [], [], nrows=3, ncols=3)
    vector = ObjectVector.from_coo(domain, [], [], size=3)

    matrix[2, 2] = "matrix"
    vector[2] = "vector"
    assert matrix.native.nvals == vector.native.nvals == 1
    del matrix[2, 2]
    del vector[2]
    assert matrix.native.nvals == vector.native.nvals == 0

    matrix[2, 2] = "matrix"
    vector[2] = "vector"
    matrix.resize(2, 2)
    vector.resize(2)
    assert matrix.native.shape == (2, 2)
    assert vector.native.size == 2
    assert matrix.nvals == vector.nvals == 0

    matrix[0, 0] = "matrix"
    vector[0] = "vector"
    matrix.clear()
    vector.clear()
    assert matrix.native.nvals == vector.native.nvals == 0


def test_dup_uses_native_copy_and_shares_domain():
    domain = ObjectDomain()
    value = Payload("value")
    matrix = ObjectMatrix.from_coo(domain, [0], [1], [value], nrows=2, ncols=2)
    vector = ObjectVector.from_coo(domain, [1], [value], size=2)

    matrix_copy = matrix.dup()
    vector_copy = vector.dup()

    assert matrix_copy.domain is domain
    assert vector_copy.domain is domain
    assert matrix_copy.native is not matrix.native
    assert vector_copy.native is not vector.native
    assert matrix_copy[0, 1] is value
    assert vector_copy[1] is value


def test_wrapping_existing_native_objects_validates_dtype_and_handles():
    domain = ObjectDomain()
    value = Payload("value")
    handle = domain.intern(value)
    matrix = gb.Matrix.from_coo([0], [0], [handle], dtype=gb.dtypes.UINT64)
    vector = gb.Vector.from_coo([0], [handle], dtype=gb.dtypes.UINT64)

    assert ObjectMatrix(domain, matrix)[0, 0] is value
    assert ObjectVector(domain, vector)[0] is value

    with pytest.raises(TypeError, match="UINT64"):
        ObjectMatrix(domain, gb.Matrix(gb.dtypes.INT64, 1, 1))
    with pytest.raises(TypeError, match="UINT64"):
        ObjectVector(domain, gb.Vector(gb.dtypes.INT64, 1))
    with pytest.raises(KeyError, match="does not belong"):
        ObjectMatrix(
            domain,
            gb.Matrix.from_coo([0], [0], [handle + 1], dtype=gb.dtypes.UINT64),
        )


def test_unknown_handle_inserted_through_native_escape_hatch_fails_clearly():
    domain = ObjectDomain()
    vector = ObjectVector.from_coo(domain, [], [], size=1)
    vector.native[0] = 42

    with pytest.raises(KeyError, match="handle 42"):
        vector[0]
    with pytest.raises(KeyError, match="handle 42"):
        vector.to_coo()


def test_duplicate_coordinates_are_rejected_before_native_construction():
    domain = ObjectDomain()
    with pytest.raises(ValueError, match="duplicate coordinates"):
        ObjectMatrix.from_coo(domain, [0, 0], [0, 0], ["a", "b"], nrows=1, ncols=1)
    with pytest.raises(ValueError, match="duplicate indices"):
        ObjectVector.from_coo(domain, [0, 0], ["a", "b"], size=1)


def test_domain_retains_removed_objects_until_domain_teardown():
    domain = ObjectDomain()
    value = Payload("value")
    value_ref = weakref.ref(value)
    vector = ObjectVector.from_coo(domain, [0], [value], size=1)
    del value
    del vector[0]

    collect()
    assert value_ref() is not None
    del vector
    collect()
    assert value_ref() is not None
    del domain
    collect()
    assert value_ref() is None


def test_native_duplicate_keeps_domain_and_objects_alive():
    domain = ObjectDomain()
    value = Payload("value")
    value_ref = weakref.ref(value)
    vector = ObjectVector.from_coo(domain, [0], [value], size=1)
    duplicate = vector.dup()
    del value
    del vector
    del domain

    collect()
    assert value_ref() is not None
    assert duplicate[0] is value_ref()
    del duplicate
    collect()
    assert value_ref() is None


def test_wrapper_domain_object_cycles_are_collectable():
    domain = ObjectDomain()
    value = Payload("cycle")
    vector = ObjectVector.from_coo(domain, [0], [value], size=1)
    value.vector = vector
    domain_ref = weakref.ref(domain)
    value_ref = weakref.ref(value)
    vector_ref = weakref.ref(vector)
    del domain
    del value
    del vector

    collect()
    assert domain_ref() is None
    assert value_ref() is None
    assert vector_ref() is None
