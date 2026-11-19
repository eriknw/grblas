import gc
import weakref

import pytest

from object_domain_poc import ObjectDomain


class Payload:
    def __init__(self, value):
        self.value = value


def collect():
    gc.collect()
    gc.collect()


def test_matrix_construction_extraction_and_none():
    domain = ObjectDomain()
    first = object()
    matrix = domain.matrix(
        3,
        4,
        rows=[2, 0],
        columns=[1, 3],
        values=[first, None],
    )

    assert matrix.shape == (3, 4)
    assert matrix.nvals == 2
    assert matrix[2, 1] is first
    assert matrix[0, 3] is None
    assert (0, 3) in matrix
    assert (1, 1) not in matrix
    with pytest.raises(KeyError):
        matrix[1, 1]
    assert matrix.to_coo() == ([0, 2], [3, 1], [None, first])


def test_vector_construction_assignment_deletion_and_bounds():
    domain = ObjectDomain()
    vector = domain.vector(3, indices=[1], values=["one"])

    vector[2] = "two"
    assert vector.to_coo() == ([1, 2], ["one", "two"])
    del vector[1]
    assert vector.to_coo() == ([2], ["two"])
    with pytest.raises(IndexError):
        vector[3] = "outside"


def test_duplicates_require_an_explicit_reducer():
    domain = ObjectDomain()
    with pytest.raises(ValueError, match="duplicate coordinate"):
        domain.matrix(1, 1, rows=[0, 0], columns=[0, 0], values=["a", "b"])

    matrix = domain.matrix(
        1,
        1,
        rows=[0, 0, 0],
        columns=[0, 0, 0],
        values=["a", "b", "c"],
        dup=lambda left, right: f"({left}+{right})",
    )
    assert matrix[0, 0] == "((a+b)+c)"


def test_elementwise_operations_are_eager_and_preserve_identity_for_lone_values():
    domain = ObjectDomain()
    lone_left = object()
    lone_right = object()
    left = domain.vector(4, indices=[0, 1], values=[lone_left, "L"])
    right = domain.vector(4, indices=[1, 2], values=["R", lone_right])
    calls = []

    def combine(x, y):
        calls.append((x, y))
        return x + y

    intersection = left.ewise_intersect(right, combine)
    union = left.ewise_union(right, combine)

    assert intersection.to_coo() == ([1], ["LR"])
    indices, values = union.to_coo()
    assert indices == [0, 1, 2]
    assert values[0] is lone_left
    assert values[1] == "LR"
    assert values[2] is lone_right
    assert calls == [("L", "R"), ("L", "R")]


def test_reduction_is_deterministic_and_empty_uses_identity():
    domain = ObjectDomain()
    matrix = domain.matrix(
        2,
        2,
        rows=[1, 0, 1],
        columns=[0, 1, 1],
        values=["c", "b", "d"],
    )
    concatenate = domain.monoid(lambda left, right: left + right, "a")

    assert matrix.reduce(concatenate) == "abcd"
    assert domain.vector(4).reduce(concatenate) == "a"


def test_operator_exception_is_raised_without_a_partial_result():
    domain = ObjectDomain()
    left = domain.vector(3, indices=[0, 1], values=["ok", "boom"])
    right = domain.vector(3, indices=[0, 1], values=["value", "value"])

    def fail_on_boom(x, y):
        if x == "boom":
            raise MarkerError("operator failed")
        return Payload((x, y))

    with pytest.raises(MarkerError, match="operator failed"):
        left.ewise_intersect(right, fail_on_boom)


class MarkerError(Exception):
    pass


def test_values_are_released_on_replace_delete_and_clear():
    domain = ObjectDomain()
    original = Payload("original")
    replacement = Payload("replacement")
    original_ref = weakref.ref(original)
    replacement_ref = weakref.ref(replacement)
    vector = domain.vector(1, indices=[0], values=[original])
    del original

    collect()
    assert original_ref() is not None
    vector[0] = replacement
    collect()
    assert original_ref() is None

    del replacement
    del vector[0]
    collect()
    assert replacement_ref() is None


def test_copies_and_results_own_independent_strong_references():
    domain = ObjectDomain()
    value = Payload("value")
    value_ref = weakref.ref(value)
    source = domain.vector(2, indices=[0], values=[value])
    copied = source.copy()
    result = source.ewise_union(domain.vector(2), lambda x, y: None)
    del value
    del source
    del copied

    collect()
    assert value_ref() is not None
    del result
    collect()
    assert value_ref() is None


def test_close_releases_values_and_invalidates_all_containers():
    domain = ObjectDomain()
    value = Payload("value")
    value_ref = weakref.ref(value)
    matrix = domain.matrix(1, 1, rows=[0], columns=[0], values=[value])
    del value

    domain.close()
    domain.close()
    collect()

    assert value_ref() is None
    assert domain.closed
    with pytest.raises(RuntimeError, match="closed"):
        matrix.to_coo()
    with pytest.raises(RuntimeError, match="closed"):
        domain.vector(1)


def test_container_cycles_are_collectable():
    domain = ObjectDomain()
    payload = Payload("cycle")
    matrix = domain.matrix(1, 1, rows=[0], columns=[0], values=[payload])
    payload.matrix = matrix
    payload_ref = weakref.ref(payload)
    matrix_ref = weakref.ref(matrix)
    del payload
    del matrix

    collect()
    assert payload_ref() is None
    assert matrix_ref() is None
