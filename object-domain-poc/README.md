# Object domain proof of concept

This directory contains the isolated stage-one prototype described in
`ai-notes/pyobject-udt.md`. It does not import or modify `python-graphblas` and
does not use a GraphBLAS backend. Sparse values are ordinary strong Python
references stored in coordinate dictionaries.

The prototype provides:

- `ObjectDomain` factories for sparse matrices and vectors
- scalar extraction, assignment, deletion, copying, and COO export
- eager elementwise intersection and union operations
- deterministic row-major (matrix) or index-order (vector) monoid reduction
- explicit domain closure, which invalidates its containers and releases values

Absent entries are distinct from stored `None`. Duplicate input coordinates
are rejected unless a Python duplicate reducer is supplied.

Run its tests from the repository root:

```console
python -m pytest object-domain-poc/tests
```

This is intentionally not installed as part of `python-graphblas`. Later stages
can use it as the semantic reference while experimenting with arena handles and
structural GraphBLAS storage.
