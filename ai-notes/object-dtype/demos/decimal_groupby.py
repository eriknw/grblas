# ruff: noqa: INP001, T201
"""Group-by sums of decimal.Decimal money amounts (exact cents).

Rows (transactions) go in a matrix as (group, row) entries; ``reduce_rowwise`` with a
Python + monoid sums each group. Baselines: a dict accumulation loop, numpy object
arrays with ``np.add.reduceat`` after a stable sort by group, and pandas groupby.
Reference: all of them must agree exactly, and with the sums computed in integer cents.
Without pandas, its baseline is skipped.

Run: PYTHONPATH=<worktree> python decimal_groupby.py [--large] [--reps N] [--csv PATH]
Success: each printed row has 'correct': True, and the last line starts with "OK:".
"""

import operator
from decimal import Decimal

import numpy as np
from _common import best_of, loc, run

import graphblas as gb

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None

SIZES = ((200_000, 1000), (1_000_000, 10_000))


def groupby_graphblas(groups, amounts, ngroups):
    """{group: Decimal sum} with reduce_rowwise."""
    n = len(amounts)
    M = gb.Matrix.from_coo(groups, np.arange(n), amounts, dtype=object, nrows=ngroups, ncols=n)
    plus = gb.monoid.register_anonymous(
        gb.binary.register_anonymous(operator.add, is_udt=True), Decimal(0)
    )
    idx, vals = M.reduce_rowwise(plus).new().to_coo()
    return dict(zip(idx.tolist(), vals, strict=True))


def groupby_dict(groups, amounts):
    """The plain Python loop."""
    acc = {}
    for g, x in zip(groups.tolist(), amounts, strict=True):
        acc[g] = acc[g] + x if g in acc else x
    return acc


def groupby_numpy(groups, amounts):
    order = np.argsort(groups, kind="stable")
    g = groups[order]
    vals = np.empty(len(amounts), dtype=object)
    vals[:] = amounts
    vals = vals[order]
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    return dict(zip(g[starts].tolist(), np.add.reduceat(vals, starts).tolist(), strict=True))


def groupby_pandas(groups, amounts):
    s = pd.Series(amounts, dtype=object).groupby(groups, sort=True).sum()
    return dict(zip(s.index.tolist(), s.tolist(), strict=True))


def main(sizes=SIZES, reps=3):
    if pd is None:
        print("pandas is not installed: skipping the pandas baseline")
    out = []
    for n, ngroups in sizes:
        rng = np.random.default_rng(4)
        groups = rng.integers(0, ngroups, n)
        cents = rng.integers(1, 10**7, n)
        amounts = [Decimal(int(c)).scaleb(-2) for c in cents]
        t_gb, got = best_of(groupby_graphblas, groups, amounts, ngroups, reps=reps)
        t_d, ref = best_of(groupby_dict, groups, amounts, reps=reps)
        t_np, r_np = best_of(groupby_numpy, groups, amounts, reps=reps)
        if pd is None:
            pd_note, r_pd = "skipped (not installed)", ref
        else:
            t_pd, r_pd = best_of(groupby_pandas, groups, amounts, reps=reps)
            pd_note = f"{t_pd:.3f}s"
        exact_cents = {g: Decimal(int(c)) for g, c in groupby_dict(groups, list(cents)).items()}
        ok = got == ref == r_np == r_pd and all(got[g] == exact_cents[g].scaleb(-2) for g in got)
        f64 = np.bincount(groups, weights=cents / 100.0, minlength=ngroups)
        f_wrong = sum(Decimal(repr(float(f64[g]))) != v for g, v in ref.items())
        out.append(
            {
                "demo": "Decimal group-by sum",
                "size": f"rows={n} groups={ngroups}",
                "graphblas_s": t_gb,
                "baseline": "dict loop [numpy reduceat, pandas]",
                "baseline_s": t_d,
                "correct": ok,
                "note": (
                    f"numpy object reduceat {t_np:.3f}s, pandas groupby {pd_note}; "
                    f"FP64 bincount sums differ from exact in {f_wrong}/{len(ref)} groups"
                ),
                "user_loc": loc(groupby_graphblas),
                "baseline_loc": loc(groupby_dict),
            }
        )
        print(out[-1], flush=True)
    return out


if __name__ == "__main__":
    run(main, SIZES, __doc__)
