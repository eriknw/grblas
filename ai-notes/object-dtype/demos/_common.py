# ruff: noqa: INP001, T201
"""Helpers shared by the demos: the command line, timing, a random graph, and line counts."""

import argparse
import csv
import inspect
import sys
import sysconfig
import time
from pathlib import Path

import numpy as np

import graphblas as gb


def build():
    """``"FT"`` on a free-threaded Python running without the GIL, else ``"GIL"``."""
    ft = bool(sysconfig.get_config_var("Py_GIL_DISABLED")) and not sys._is_gil_enabled()
    return "FT" if ft else "GIL"


def parse_args(doc):
    parser = argparse.ArgumentParser(
        description=doc, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--large", action="store_true", help="also run the larger size (several seconds per demo)"
    )
    parser.add_argument("--reps", type=int, default=3, help="best of this many runs (default 3)")
    parser.add_argument("--csv", metavar="PATH", help="append one row per demo and size to PATH")
    return parser.parse_args()


def start():
    """Print where graphblas comes from; exit with a message if it has no OBJECT dtype."""
    if not hasattr(gb.dtypes, "OBJECT"):
        sys.exit(f"{gb.__file__} has no OBJECT dtype: put objdtype-core on PYTHONPATH")
    print(f"graphblas {gb.__file__}; Python {sys.version.split()[0]} {build()}", flush=True)


def finish(rows, csv_path=None):
    """Write the rows to ``csv_path`` if given, print the verdict, and return the exit code.

    A row's ``correct`` is None when its reference library is not installed.
    """
    for row in rows:
        row["build"] = build()
        base = row["baseline_s"]
        row["ratio_graphblas_over_baseline"] = round(row["graphblas_s"] / base, 2) if base else None
    if csv_path:
        path = Path(csv_path)
        new = not path.exists()
        with path.open("a", newline="") as f:
            writer = csv.DictWriter(f, list(rows[0]))
            if new:
                writer.writeheader()
            writer.writerows(rows)
    wrong = [f"{row['demo']} ({row['size']})" for row in rows if row["correct"] is False]
    unchecked = sum(row["correct"] is None for row in rows)
    if wrong:
        print(f"FAIL: {len(wrong)} of {len(rows)} results differ from the reference: {wrong}")
        return 1
    msg = f"OK: {len(rows) - unchecked} of {len(rows)} results match the reference"
    if unchecked:
        msg += f"; {unchecked} unchecked (reference library not installed)"
    print(msg)
    return 0


def run(main, sizes, doc):
    """Run one demo from the command line: the first size, or every size with ``--large``."""
    args = parse_args(doc)
    start()
    rows = main(sizes if args.large else sizes[:1], reps=args.reps)
    sys.exit(finish(rows, args.csv))


def best_of(fn, *args, reps=3):
    """Best wall time of ``reps`` calls of ``fn(*args)``, and the last result."""
    best = None
    out = None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn(*args)
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)
    return best, out


def random_digraph(n, m, seed):
    """``m`` distinct directed edges (no self loops) on ``n`` nodes, sorted."""
    rng = np.random.default_rng(seed)
    idx = np.unique(rng.integers(0, n * n, int(m * 1.2)))
    rng.shuffle(idx)
    idx = idx[(idx // n) != (idx % n)][:m]
    idx.sort()
    return idx // n, idx % n


def loc(fn):
    """Lines of code of a function, without its docstring and blank lines."""
    src = inspect.getsource(fn).splitlines()
    body = [ln for ln in src if ln.strip() and not ln.strip().startswith("#")]
    doc = fn.__doc__
    ndoc = len(doc.strip().splitlines()) if doc else 0
    return len(body) - ndoc
