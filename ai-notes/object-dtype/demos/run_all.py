# ruff: noqa: INP001, T201
"""Run every demo and check each result against its reference.

The demos store Python objects (big ints, Fractions, tuples, Decimals, frozensets) in
OBJECT matrices and vectors and run Python functions as GraphBLAS operators. Each
compares its result with an independent computation (pure Python, networkx, numpy or
pandas) and reports both run times. Run it under a GIL build and a free-threaded build to
compare: with the GIL, Python ops run on one thread; without it, the C and Numba
trampolines use SuiteSparse's threads (cffi callbacks still use one). Without networkx,
two demos run unchecked; without pandas, one baseline is skipped.

Run: PYTHONPATH=<worktree> python run_all.py [--large] [--reps N] [--csv PATH]
Success: every printed row has 'correct': True, and the last line starts with "OK:".
"""

import sys
import time

import bfs_paths
import decimal_groupby
import provenance
import rational_sssp
import walks
from _common import finish, parse_args, start


def main():
    args = parse_args(__doc__)
    start()
    rows = []
    for mod in (walks, rational_sssp, bfs_paths, decimal_groupby, provenance):
        t0 = time.perf_counter()
        rows.extend(mod.main(mod.SIZES if args.large else mod.SIZES[:1], reps=args.reps))
        print(f"{mod.__name__} done in {time.perf_counter() - t0:.1f}s", flush=True)
    return finish(rows, args.csv)


if __name__ == "__main__":
    sys.exit(main())
