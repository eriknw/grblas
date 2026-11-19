from __future__ import annotations

import operator
from collections.abc import Iterable
from typing import Any

import numpy as np

import graphblas as gb


_ABSENT = object()
_MAX_HANDLE = np.iinfo(np.uint64).max


class ObjectDomain:
    """Strong-reference owner for opaque handles stored by SuiteSparse."""

    __slots__ = ("_handles_by_identity", "_objects", "_next_handle", "__weakref__")

    def __init__(self) -> None:
        self._objects: dict[int, object] = {}
        self._handles_by_identity: dict[int, int] = {}
        self._next_handle = 1

    def __len__(self) -> int:
        return len(self._objects)

    def intern(self, value: object) -> int:
        """Return the stable domain-local handle for *value*'s identity."""
        identity = id(value)
        if (handle := self._handles_by_identity.get(identity)) is not None:
            if self._objects[handle] is value:
                return handle
            raise RuntimeError("object identity index is inconsistent")
        if self._next_handle > _MAX_HANDLE:
            raise OverflowError("object domain exhausted UINT64 handles")
        handle = self._next_handle
        self._next_handle += 1
        self._objects[handle] = value
        self._handles_by_identity[identity] = handle
        return handle

    def resolve(self, handle: object) -> object:
        """Resolve a handle or raise when it does not belong to this domain."""
        try:
            handle = operator.index(handle)
        except TypeError:
            raise TypeError("object handle must be an integer") from None
        try:
            return self._objects[handle]
        except KeyError:
            raise KeyError(f"handle {handle} does not belong to this object domain") from None

    def contains_handle(self, handle: object) -> bool:
        try:
            handle = operator.index(handle)
        except TypeError:
            return False
        return handle in self._objects


class ObjectMatrix:
    """Object view of a native SuiteSparse UINT64 Matrix."""

    __slots__ = ("domain", "_matrix", "__weakref__")

    def __init__(self, domain: ObjectDomain, matrix: gb.Matrix, *, validate: bool = True):
        _require_suitesparse()
        if not isinstance(domain, ObjectDomain):
            raise TypeError("domain must be an ObjectDomain")
        if type(matrix) is not gb.Matrix:
            raise TypeError("matrix must be a concrete graphblas.Matrix")
        if matrix.dtype != gb.dtypes.UINT64:
            raise TypeError("native object matrix must have UINT64 dtype")
        self.domain = domain
        self._matrix = matrix
        if validate:
            self._validate_handles()

    @classmethod
    def from_coo(
        cls,
        domain: ObjectDomain,
        rows: Iterable[int],
        columns: Iterable[int],
        values: Iterable[object],
        *,
        nrows: int,
        ncols: int,
        name: str | None = None,
    ) -> ObjectMatrix:
        _require_suitesparse()
        if not isinstance(domain, ObjectDomain):
            raise TypeError("domain must be an ObjectDomain")
        row_list = list(rows)
        column_list = list(columns)
        value_list = list(values)
        if not (len(row_list) == len(column_list) == len(value_list)):
            raise ValueError("rows, columns, and values must have the same length")
        coordinates = list(zip(row_list, column_list, strict=True))
        if len(set(coordinates)) != len(coordinates):
            raise ValueError("duplicate coordinates are not supported")
        handles = np.fromiter(
            (domain.intern(value) for value in value_list),
            dtype=np.uint64,
            count=len(value_list),
        )
        native = gb.Matrix.from_coo(
            row_list,
            column_list,
            handles,
            dtype=gb.dtypes.UINT64,
            nrows=nrows,
            ncols=ncols,
            name=name,
        )
        return cls(domain, native, validate=False)

    @property
    def native(self) -> gb.Matrix:
        return self._matrix

    @property
    def shape(self) -> tuple[int, int]:
        return self._matrix.shape

    @property
    def nrows(self) -> int:
        return self._matrix.nrows

    @property
    def ncols(self) -> int:
        return self._matrix.ncols

    @property
    def nvals(self) -> int:
        return self._matrix.nvals

    @property
    def S(self):
        """The structural mask of the native matrix."""
        return self._matrix.S

    @property
    def format(self) -> str:
        return self._matrix.ss.format

    def __contains__(self, key: object) -> bool:
        return key in self._matrix

    def __getitem__(self, key: tuple[int, int]) -> object:
        row, column = _matrix_key(key)
        handle = self._matrix.get(row, column, _ABSENT)
        if handle is _ABSENT:
            raise KeyError(key)
        return self.domain.resolve(handle)

    def get(self, row: int, column: int, default: object = None) -> object:
        handle = self._matrix.get(row, column, _ABSENT)
        if handle is _ABSENT:
            return default
        return self.domain.resolve(handle)

    def __setitem__(self, key: tuple[int, int], value: object) -> None:
        row, column = _matrix_key(key)
        self._matrix[row, column] = self.domain.intern(value)

    def __delitem__(self, key: tuple[int, int]) -> None:
        row, column = _matrix_key(key)
        del self._matrix[row, column]

    def clear(self) -> None:
        self._matrix.clear()

    def resize(self, nrows: int, ncols: int) -> None:
        self._matrix.resize(nrows, ncols)

    def dup(self, *, name: str | None = None) -> ObjectMatrix:
        return type(self)(self.domain, self._matrix.dup(name=name), validate=False)

    def to_coo(self, *, sort: bool = True) -> tuple[np.ndarray, np.ndarray, list[object]]:
        rows, columns, handles = self._matrix.to_coo(dtype=gb.dtypes.UINT64, sort=sort)
        return rows, columns, [self.domain.resolve(handle) for handle in handles]

    def _validate_handles(self) -> None:
        _, _, handles = self._matrix.to_coo(
            rows=False,
            columns=False,
            dtype=gb.dtypes.UINT64,
        )
        for handle in handles:
            self.domain.resolve(handle)


class ObjectVector:
    """Object view of a native SuiteSparse UINT64 Vector."""

    __slots__ = ("domain", "_vector", "__weakref__")

    def __init__(self, domain: ObjectDomain, vector: gb.Vector, *, validate: bool = True):
        _require_suitesparse()
        if not isinstance(domain, ObjectDomain):
            raise TypeError("domain must be an ObjectDomain")
        if type(vector) is not gb.Vector:
            raise TypeError("vector must be a concrete graphblas.Vector")
        if vector.dtype != gb.dtypes.UINT64:
            raise TypeError("native object vector must have UINT64 dtype")
        self.domain = domain
        self._vector = vector
        if validate:
            self._validate_handles()

    @classmethod
    def from_coo(
        cls,
        domain: ObjectDomain,
        indices: Iterable[int],
        values: Iterable[object],
        *,
        size: int,
        name: str | None = None,
    ) -> ObjectVector:
        _require_suitesparse()
        if not isinstance(domain, ObjectDomain):
            raise TypeError("domain must be an ObjectDomain")
        index_list = list(indices)
        value_list = list(values)
        if len(index_list) != len(value_list):
            raise ValueError("indices and values must have the same length")
        if len(set(index_list)) != len(index_list):
            raise ValueError("duplicate indices are not supported")
        handles = np.fromiter(
            (domain.intern(value) for value in value_list),
            dtype=np.uint64,
            count=len(value_list),
        )
        native = gb.Vector.from_coo(
            index_list,
            handles,
            dtype=gb.dtypes.UINT64,
            size=size,
            name=name,
        )
        return cls(domain, native, validate=False)

    @property
    def native(self) -> gb.Vector:
        return self._vector

    @property
    def size(self) -> int:
        return self._vector.size

    @property
    def shape(self) -> tuple[int, ...]:
        return self._vector.shape

    @property
    def nvals(self) -> int:
        return self._vector.nvals

    @property
    def S(self):
        """The structural mask of the native vector."""
        return self._vector.S

    @property
    def format(self) -> str:
        return self._vector.ss.format

    def __contains__(self, index: object) -> bool:
        return index in self._vector

    def __getitem__(self, index: int) -> object:
        handle = self._vector.get(index, _ABSENT)
        if handle is _ABSENT:
            raise KeyError(index)
        return self.domain.resolve(handle)

    def get(self, index: int, default: object = None) -> object:
        handle = self._vector.get(index, _ABSENT)
        if handle is _ABSENT:
            return default
        return self.domain.resolve(handle)

    def __setitem__(self, index: int, value: object) -> None:
        self._vector[index] = self.domain.intern(value)

    def __delitem__(self, index: int) -> None:
        del self._vector[index]

    def clear(self) -> None:
        self._vector.clear()

    def resize(self, size: int) -> None:
        self._vector.resize(size)

    def dup(self, *, name: str | None = None) -> ObjectVector:
        return type(self)(self.domain, self._vector.dup(name=name), validate=False)

    def to_coo(self, *, sort: bool = True) -> tuple[np.ndarray, list[object]]:
        indices, handles = self._vector.to_coo(dtype=gb.dtypes.UINT64, sort=sort)
        return indices, [self.domain.resolve(handle) for handle in handles]

    def _validate_handles(self) -> None:
        _, handles = self._vector.to_coo(indices=False, dtype=gb.dtypes.UINT64)
        for handle in handles:
            self.domain.resolve(handle)


def _require_suitesparse() -> None:
    # Accessing Matrix initializes the default backend when needed.
    _ = gb.Matrix
    if gb.backend != "suitesparse":
        raise RuntimeError("the object-domain prototype requires the suitesparse backend")


def _matrix_key(key: object) -> tuple[Any, Any]:
    if not isinstance(key, tuple) or len(key) != 2:
        raise TypeError("matrix key must be a (row, column) tuple")
    return key
