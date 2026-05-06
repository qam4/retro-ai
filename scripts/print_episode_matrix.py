#!/usr/bin/env python3
"""Print the start_level × reached_level transition matrix from episodes.csv.

Complements the live ``success=[...]`` metric in the training log, which
only tells you "fraction of episodes starting at N that reached ≥ N+1" —
it collapses "reached N+1" and "reached N+4" into the same number.

Usage
-----

::

    python scripts/print_episode_matrix.py \\
        output/.../run/episodes.csv \\
        [--last-fraction 0.2]

With ``--last-fraction`` 0.2 (default), the matrix uses only the last
20% of episodes — the agent's behavior after training has converged.
Pass 1.0 to use all episodes.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("episodes_csv", help="Path to episodes.csv.")
    p.add_argument(
        "--last-fraction",
        type=float,
        default=0.2,
        help=(
            "Fraction of the most recent episodes to include (default 0.2). "
            "Pass 1.0 to include everything."
        ),
    )
    args = p.parse_args()

    with open(args.episodes_csv) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("No rows.")
        return

    n = len(rows)
    take = max(1, int(args.last_fraction * n))
    subset = rows[n - take :]

    trans: Counter = Counter()
    for r in subset:
        s = int(r["start_level"])
        e = int(r["reached_level"])
        trans[(s, e)] += 1

    starts = sorted({s for (s, _) in trans})
    reacheds = sorted({e for (_, e) in trans})

    print(
        f"{n} total episodes, showing last {take} "
        f"({args.last_fraction*100:.0f}%)"
    )
    print()
    # Header
    header_label = "start \\ reached"
    header = f"{header_label:>16s}  " + "  ".join(f"{r:>6d}" for r in reacheds)
    print(header + "     n")
    for s in starts:
        total = sum(c for (s2, _), c in trans.items() if s2 == s)
        row_cells = []
        for e in reacheds:
            c = trans.get((s, e), 0)
            pct = 100 * c / total if total else 0
            row_cells.append(f"{pct:5.1f}%" if c else "   -  ")
        print(
            f"{'start=' + str(s):>16s}  "
            + "  ".join(row_cells)
            + f"   {total:>5d}"
        )

    # Extra: fraction of reset episodes that reach each level
    print()
    from_reset = [r for r in subset if int(r["start_level"]) == 0]
    if from_reset:
        print("from reset chain: fraction reaching ≥ N")
        max_r = max(int(r["reached_level"]) for r in subset)
        for L in range(1, max_r + 1):
            hit = sum(1 for r in from_reset if int(r["reached_level"]) >= L)
            pct = 100 * hit / len(from_reset)
            print(f"  ≥{L}: {pct:5.1f}% ({hit}/{len(from_reset)})")


if __name__ == "__main__":
    main()
