# ruff: noqa: INP001, T201, RUF005
"""Breadth-first search that carries the actual paths (tuples) as values.

Edge (u, v) stores the object v. The multiply appends it to the path (``path + (v,)``);
the add keeps the shorter path, ties broken lexicographically (so the result is
deterministic). One masked ``vxm`` per BFS level. Reference: networkx
``single_source_shortest_path``, which also returns a path per reached node; the check
compares the reached nodes and path lengths, and that each path follows edges from the
source. Without networkx the demo still runs, and its row is unchecked.

Run: PYTHONPATH=<worktree> python bfs_paths.py [--large] [--reps N] [--csv PATH]
Success: each printed row has 'correct': True, and the last line starts with "OK:".
"""

from _common import best_of, loc, random_digraph, run

import graphblas as gb

try:
    import networkx as nx
except ModuleNotFoundError:
    nx = None

SIZES = ((20000, 100000), (100000, 500000))


def shorter(p, q):
    return p if (len(p), p) <= (len(q), q) else q


def bfs_paths_graphblas(rows, cols, n, source):
    """{node: path tuple} for every node reachable from ``source``."""
    A = gb.Matrix.from_coo(rows, cols, [int(c) for c in cols], dtype=object, nrows=n, ncols=n)
    keep = gb.monoid.register_anonymous(
        gb.binary.register_anonymous(shorter, is_udt=True), (None,) * n
    )
    extend = gb.binary.register_anonymous(lambda path, v: path + (v,), is_udt=True)
    semi = gb.semiring.register_anonymous(keep, extend)
    paths = gb.Vector.from_coo([source], [(source,)], dtype=object, size=n)
    frontier = paths
    while frontier.nvals:
        frontier = frontier.vxm(A, semi).new(mask=~paths.S)
        paths(frontier.S) << frontier
    idx, vals = paths.to_coo()
    return dict(zip(idx.tolist(), vals, strict=True))


def bfs_paths_networkx(G, source):
    """Networkx BFS with paths."""
    return nx.single_source_shortest_path(G, source)


def main(sizes=SIZES, reps=3):
    if nx is None:
        print("networkx is not installed: no networkx baseline, and the rows are unchecked")
    out = []
    for n, m in sizes:
        r, c = random_digraph(n, m, seed=5)
        t_gb, got = best_of(bfs_paths_graphblas, r, c, n, 0, reps=reps)
        if nx is None:
            t_nx = ok = None
        else:
            G = nx.DiGraph()
            G.add_nodes_from(range(n))
            G.add_edges_from(zip(r.tolist(), c.tolist(), strict=True))
            t_nx, ref = best_of(bfs_paths_networkx, G, 0, reps=reps)
            edges = set(zip(r.tolist(), c.tolist(), strict=True))
            ok = got.keys() == ref.keys() and all(
                len(got[v]) == len(ref[v])
                and got[v][0] == 0
                and got[v][-1] == v
                and all((a, b) in edges for a, b in zip(got[v], got[v][1:], strict=False))
                for v in got
            )
        out.append(
            {
                "demo": "BFS with paths (tuples)",
                "size": f"n={n} m={m}",
                "graphblas_s": t_gb,
                "baseline": "networkx single_source_shortest_path",
                "baseline_s": t_nx,
                "correct": ok,
                "note": f"reached {len(got)}; depth {max(map(len, got.values())) - 1}",
                "user_loc": loc(bfs_paths_graphblas) + loc(shorter),
                "baseline_loc": loc(bfs_paths_networkx),
            }
        )
        print(out[-1], flush=True)
    return out


if __name__ == "__main__":
    run(main, SIZES, __doc__)
