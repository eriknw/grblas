from __future__ import annotations

import operator
import weakref
from collections.abc import Callable, Iterable, Iterator
from typing import Any


class ObjectBinaryOp:
    """An eager Python binary operation for object-domain values."""

    def __init__(self, function: Callable[[object, object], object], *, name: str | None = None):
        if not callable(function):
            raise TypeError("function must be callable")
        self.function = function
        self.name = name or getattr(function, "__name__", type(function).__name__)

    def __call__(self, left: object, right: object) -> object:
        return self.function(left, right)

    def __repr__(self) -> str:
        return f"ObjectBinaryOp({self.name!r})"


class ObjectMonoid:
    """A binary operation and identity used for deterministic reduction."""

    def __init__(
        self,
        function: ObjectBinaryOp | Callable[[object, object], object],
        identity: object,
        *,
        name: str | None = None,
    ):
        self.binary = _as_binary(function)
        self.identity = identity
        self.name = name or self.binary.name

    def __repr__(self) -> str:
        return f"ObjectMonoid({self.name!r}, identity={self.identity!r})"


class ObjectDomain:
    """Owner and validity boundary for Python-managed sparse object containers."""

    def __init__(self) -> None:
        self._closed = False
        self._containers: weakref.WeakSet[_ObjectSparse] = weakref.WeakSet()

    @property
    def closed(self) -> bool:
        return self._closed

    def matrix(
        self,
        nrows: int,
        ncols: int,
        *,
        rows: Iterable[int] = (),
        columns: Iterable[int] = (),
        values: Iterable[object] = (),
        dup: ObjectBinaryOp | Callable[[object, object], object] | None = None,
    ) -> ObjectMatrix:
        self._check_open()
        shape = (_dimension(nrows, "nrows"), _dimension(ncols, "ncols"))
        row_list = list(rows)
        column_list = list(columns)
        value_list = list(values)
        if not (len(row_list) == len(column_list) == len(value_list)):
            raise ValueError("rows, columns, and values must have the same length")
        data: dict[tuple[int, int], object] = {}
        duplicate = None if dup is None else _as_binary(dup)
        for row, column, value in zip(row_list, column_list, value_list, strict=True):
            key = (_index(row, shape[0], "row"), _index(column, shape[1], "column"))
            if key in data:
                if duplicate is None:
                    raise ValueError(f"duplicate coordinate: {key!r}")
                value = duplicate(data[key], value)
            data[key] = value
        return ObjectMatrix(self, shape, data)

    def vector(
        self,
        size: int,
        *,
        indices: Iterable[int] = (),
        values: Iterable[object] = (),
        dup: ObjectBinaryOp | Callable[[object, object], object] | None = None,
    ) -> ObjectVector:
        self._check_open()
        size = _dimension(size, "size")
        index_list = list(indices)
        value_list = list(values)
        if len(index_list) != len(value_list):
            raise ValueError("indices and values must have the same length")
        data: dict[int, object] = {}
        duplicate = None if dup is None else _as_binary(dup)
        for index, value in zip(index_list, value_list, strict=True):
            key = _index(index, size, "index")
            if key in data:
                if duplicate is None:
                    raise ValueError(f"duplicate index: {key!r}")
                value = duplicate(data[key], value)
            data[key] = value
        return ObjectVector(self, size, data)

    def binary(
        self, function: Callable[[object, object], object], *, name: str | None = None
    ) -> ObjectBinaryOp:
        self._check_open()
        return ObjectBinaryOp(function, name=name)

    def monoid(
        self,
        function: ObjectBinaryOp | Callable[[object, object], object],
        identity: object,
        *,
        name: str | None = None,
    ) -> ObjectMonoid:
        self._check_open()
        return ObjectMonoid(function, identity, name=name)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for container in list(self._containers):
            container._data.clear()

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("object domain is closed")

    def _register(self, container: _ObjectSparse) -> None:
        self._containers.add(container)

    def __enter__(self) -> ObjectDomain:
        self._check_open()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class _ObjectSparse:
    _domain: ObjectDomain
    _data: dict[Any, object]
    _extent: Any

    def __init__(self, domain: ObjectDomain, extent: Any, data: dict[Any, object]):
        domain._check_open()
        self._domain = domain
        self._extent = extent
        self._data = data
        domain._register(self)

    @property
    def nvals(self) -> int:
        self._check_open()
        return len(self._data)

    def __len__(self) -> int:
        return self.nvals

    def __contains__(self, key: object) -> bool:
        self._check_open()
        try:
            normalized = self._normalize_key(key)
        except (IndexError, TypeError):
            return False
        return normalized in self._data

    def __getitem__(self, key: object) -> object:
        self._check_open()
        return self._data[self._normalize_key(key)]

    def get(self, key: object, default: object = None) -> object:
        self._check_open()
        return self._data.get(self._normalize_key(key), default)

    def __setitem__(self, key: object, value: object) -> None:
        self._check_open()
        self._data[self._normalize_key(key)] = value

    def __delitem__(self, key: object) -> None:
        self._check_open()
        del self._data[self._normalize_key(key)]

    def clear(self) -> None:
        self._check_open()
        self._data.clear()

    def copy(self):
        self._check_open()
        return type(self)(self._domain, self._extent, self._data.copy())

    def items(self) -> Iterator[tuple[Any, object]]:
        self._check_open()
        yield from ((key, self._data[key]) for key in sorted(self._data))

    def ewise_intersect(
        self, other: _ObjectSparse, op: ObjectBinaryOp | Callable[[object, object], object]
    ):
        """Apply *op* where both inputs have a stored value."""
        self._check_compatible(other)
        binary = _as_binary(op)
        data = {
            key: binary(self._data[key], other._data[key])
            for key in sorted(self._data.keys() & other._data.keys())
        }
        return type(self)(self._domain, self._extent, data)

    def ewise_union(
        self, other: _ObjectSparse, op: ObjectBinaryOp | Callable[[object, object], object]
    ):
        """Copy lone values and apply *op* where both inputs have a value."""
        self._check_compatible(other)
        binary = _as_binary(op)
        data: dict[Any, object] = {}
        for key in sorted(self._data.keys() | other._data.keys()):
            if key in self._data and key in other._data:
                data[key] = binary(self._data[key], other._data[key])
            elif key in self._data:
                data[key] = self._data[key]
            else:
                data[key] = other._data[key]
        return type(self)(self._domain, self._extent, data)

    def reduce(self, monoid: ObjectMonoid) -> object:
        """Reduce stored values in deterministic coordinate order."""
        self._check_open()
        if not isinstance(monoid, ObjectMonoid):
            raise TypeError("monoid must be an ObjectMonoid")
        result = monoid.identity
        for key in sorted(self._data):
            result = monoid.binary(result, self._data[key])
        return result

    def _check_open(self) -> None:
        self._domain._check_open()

    def _check_compatible(self, other: _ObjectSparse) -> None:
        self._check_open()
        if not isinstance(other, type(self)):
            raise TypeError(f"expected {type(self).__name__}, got {type(other).__name__}")
        other._check_open()
        if self._domain is not other._domain:
            raise ValueError("containers belong to different object domains")
        if self._extent != other._extent:
            raise ValueError("container dimensions do not match")

    def _normalize_key(self, key: object) -> Any:
        raise NotImplementedError


class ObjectMatrix(_ObjectSparse):
    def __init__(
        self,
        domain: ObjectDomain,
        shape: tuple[int, int],
        data: dict[tuple[int, int], object],
    ):
        super().__init__(domain, shape, data)

    @property
    def shape(self) -> tuple[int, int]:
        return self._extent

    @property
    def nrows(self) -> int:
        return self._extent[0]

    @property
    def ncols(self) -> int:
        return self._extent[1]

    def to_coo(self) -> tuple[list[int], list[int], list[object]]:
        self._check_open()
        rows: list[int] = []
        columns: list[int] = []
        values: list[object] = []
        for (row, column), value in self.items():
            rows.append(row)
            columns.append(column)
            values.append(value)
        return rows, columns, values

    def _normalize_key(self, key: object) -> tuple[int, int]:
        if not isinstance(key, tuple) or len(key) != 2:
            raise TypeError("matrix key must be a (row, column) tuple")
        return (
            _index(key[0], self.nrows, "row"),
            _index(key[1], self.ncols, "column"),
        )


class ObjectVector(_ObjectSparse):
    def __init__(self, domain: ObjectDomain, size: int, data: dict[int, object]):
        super().__init__(domain, size, data)

    @property
    def size(self) -> int:
        return self._extent

    def to_coo(self) -> tuple[list[int], list[object]]:
        self._check_open()
        indices: list[int] = []
        values: list[object] = []
        for index, value in self.items():
            indices.append(index)
            values.append(value)
        return indices, values

    def _normalize_key(self, key: object) -> int:
        return _index(key, self.size, "index")


def _as_binary(
    function: ObjectBinaryOp | Callable[[object, object], object],
) -> ObjectBinaryOp:
    if isinstance(function, ObjectBinaryOp):
        return function
    return ObjectBinaryOp(function)


def _dimension(value: object, name: str) -> int:
    try:
        value = operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer") from None
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _index(value: object, size: int, name: str) -> int:
    try:
        value = operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer") from None
    if value < 0 or value >= size:
        raise IndexError(f"{name} {value} is out of bounds for size {size}")
    return value
