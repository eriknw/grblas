# Native object-domain proof of concept

This isolated stage-one prototype stores object handles in real
SuiteSparse-backed `python-graphblas` matrices and vectors. It does not maintain
a Python sparse representation and does not modify the `graphblas` package.

`ObjectDomain` owns strong references to Python objects and assigns each object
an opaque, nonzero `UINT64` handle. `ObjectMatrix` and `ObjectVector` each hold
only a domain and a native `graphblas.Matrix` or `graphblas.Vector`. Coordinates,
dimensions, formats, and handles are authoritative in the native object.

The initial API includes:

- native COO construction and extraction
- scalar presence, extraction, assignment, and deletion
- native duplication with a shared domain
- shape, size, `nvals`, structural mask, and SuiteSparse format delegation
- explicit wrapping and validation of existing native handle containers
- a `.native` escape hatch for inspecting the underlying GraphBLAS object

The native escape hatch must preserve opaque-handle semantics. Numeric
operations on handles are not object operations, and inserting an integer that
does not belong to the domain makes object extraction fail clearly.

Run the tests from the repository root in a configured development environment:

```console
python -m pytest -c /dev/null object-domain-poc/tests
```
