# OBJECT dtype: demos and checks

Scripts that show and test Python objects stored in GraphBLAS containers (`dtype=object`).
Each one says in its docstring what it checks, how to run it, and what output means success;
`--help` prints that. The defaults finish in a few seconds. The design, and the results these
scripts back, are in [`../object-dtype.md`](../object-dtype.md).

## Running

Running a script by its path puts the script's directory first on `sys.path`, not the
repository root, so `import graphblas` would find an installed python-graphblas. Put this
checkout first on `PYTHONPATH`. Each script prints the `graphblas` it imported.

```bash
REPO=/path/to/python-graphblas  # a checkout of objdtype-core
PYTHONPATH=$REPO python $REPO/ai-notes/object-dtype/demos/run_all.py
```

- Runs at the same time need their own `GRAPHBLAS_CACHE_PATH` (two processes that compile
  the same new JIT kernels fail) and `GRAPHBLAS_CSHIM_DIR` (where the C trampolines are built
  on first use).
- Python ops run in C trampolines by default. `GRAPHBLAS_CSHIM=0` uses the Numba trampolines,
  and `GRAPHBLAS_CSHIM=0 GRAPHBLAS_NUMBA_CAPI=0` uses cffi callbacks. The checks print the
  one used as `paths=...`.
- On a free-threaded Python without networkx, two demos run unchecked. networkx is pure
  Python, so a source checkout works: `PYTHONPATH=$REPO:/path/to/networkx-checkout`.

## Scripts

| Script                     | Shows or checks                                                                                               | Success                          |
| -------------------------- | ------------------------------------------------------------------------------------------------------------- | -------------------------------- |
| `demos/run_all.py`         | The five demos below, each against an independent reference; `--large` adds a bigger size, `--csv` saves rows | last line `OK: ...`              |
| `demos/walks.py`           | Exact walk counts with big ints (`mxv`, Python `+` monoid)                                                    | last line `OK: ...`              |
| `demos/rational_sssp.py`   | Exact Bellman-Ford with `Fraction` weights, against networkx                                                  | last line `OK: ...`              |
| `demos/bfs_paths.py`       | BFS carrying path tuples as values, against networkx                                                          | last line `OK: ...`              |
| `demos/decimal_groupby.py` | Group-by sums of `Decimal`, against a dict loop, numpy and pandas                                             | last line `OK: ...`              |
| `demos/provenance.py`      | Why-provenance with frozensets, against pure Python                                                           | last line `OK: ...`              |
| `checks/fuzz.py`           | Random ops and collections against a pure Python model, by identity; no object leaks or is freed early        | `PASS seed=...`, exit 0          |
| `checks/thread_stress.py`  | Threads mutate, read and collect at once; identity per step, no leaks at the end                              | `PASS ... errors=0 leaked=0`     |
| `checks/safety_probe.py`   | SuiteSparse calls Python ops only on live stored objects, across formats, iso, masks and mxm methods          | `PASS ... bad=0 wrong=0`         |
| `checks/bench_calls.py`    | Nanoseconds per Python-op call, and the same work in pure Python and numpy (`--baselines`)                    | JSON lines, then `{"done": true` |

The fuzz runs one seed per process. For many seeds, loop in the shell:

```bash
for s in 41 42 43 44 45; do
  PYTHONPATH=$REPO python $REPO/ai-notes/object-dtype/checks/fuzz.py --seed $s --threshold 0 || break
done
```
