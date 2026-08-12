#!/usr/bin/env python3
"""Render the start x reached MATRIX (and slices) from a run's episodes.csv.

The matrix is ~N^2 for N route points, so it can never live in a log line —
the log carries only fixed-size 1-D projections (reach from reset, order-free
progress; see CheckpointManager.route_table). The raw material for every other
question is two columns in episodes.csv:

  start_key       the TRUE start source ("0" = real game reset, else a CP level
                  or a waypoint id). NOTE: ``start_level`` cannot be used for
                  this — it is derived from fruits-remaining, so a WP-seeded
                  episode reports 0, identical to a reset.
  reached_points  ';'-joined route points that episode reached.

So pairwise "does X -> Y work?" is a QUERY over the real training distribution,
instead of a new log field or an emulator-booting probe.

Examples:
  # from-reset chain (the composition question)
  route_report.py --run <dir> --from 0
  # every start's health, last 20% of the run
  route_report.py --run <dir> --tail 0.2
  # full matrix
  route_report.py --run <dir> --matrix
"""
from __future__ import annotations

import argparse
import collections
import csv
import os

from retro_ai.training.yeti_map import get_level_map


def load(path, tail=1.0):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if "start_key" not in (rows[0] if rows else {}):
        raise SystemExit(
            "episodes.csv predates the start_key/reached_points columns "
            "(pre-log-rework run) — from-reset analysis is not possible on it: "
            "WP-seeded episodes are indistinguishable from resets."
        )
    if tail < 1.0:
        rows = rows[int(len(rows) * (1.0 - tail)) :]
    return rows


def points(row):
    raw = (row.get("reached_points") or "").strip()
    return [p for p in raw.split(";") if p]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="training output dir")
    ap.add_argument("--from", dest="src", help="only this start_key (e.g. 0)")
    ap.add_argument(
        "--to",
        dest="dst",
        help="with --from: the PAIRWISE number P(reach this point | started "
        "there). Use this when comparing against a specific link measurement — "
        "the default `progress` column is a WEAKER, order-free condition "
        "(reached >=1 new point) and the two are NOT interchangeable.",
    )
    ap.add_argument("--matrix", action="store_true", help="full start x reached")
    ap.add_argument("--tail", type=float, default=1.0, help="fraction at the end")
    ap.add_argument("--level", type=int, default=3)
    args = ap.parse_args()

    rows = load(os.path.join(args.run, "episodes.csv"), args.tail)
    order = list(getattr(get_level_map(args.level), "route_order", None) or [])
    by_start = collections.defaultdict(list)
    for r in rows:
        by_start[r["start_key"]].append(r)
    print(f"{len(rows)} episodes; {len(by_start)} distinct starts (tail={args.tail})")

    def col_order(seen):
        return order + sorted(seen - set(order))

    if args.src is not None:
        rs = by_start.get(args.src, [])
        if not rs:
            raise SystemExit(f"no episodes with start_key={args.src!r}")
        if args.dst:
            hit = sum(1 for r in rs if args.dst in points(r))
            print(
                f"\nPAIRWISE {args.src!r} -> {args.dst!r}: "
                f"{hit}/{len(rs)} = {100*hit/len(rs):.1f}%"
            )
            return
        cnt = collections.Counter(p for r in rs for p in points(r))
        seen = set(cnt)
        print(f"\nfrom {args.src!r}: {len(rs)} episodes")
        print(f"  {'point':<18s} {'reach':>7s}")
        for p in col_order(seen):
            if p in cnt:
                print(f"  {p:<18s} {100*cnt[p]/len(rs):6.1f}%")
        return

    if args.matrix:
        seen = {p for r in rows for p in points(r)}
        cols = col_order(seen)
        starts = order + sorted(set(by_start) - set(order))
        print("\nmatrix: rows = start, cols = reached (%)")
        print(
            "  "
            + "start".ljust(16)
            + "n".rjust(6)
            + "".join(c[:6].rjust(7) for c in cols)
        )
        for s in starts:
            rs = by_start.get(s)
            if not rs:
                continue
            cnt = collections.Counter(p for r in rs for p in points(r))
            cells = "".join(f"{100*cnt.get(c,0)/len(rs):6.0f} " for c in cols)
            print(f"  {str(s):<16s}{len(rs):6d}{cells}")
        return

    # Default: per-start health (the order-free projection).
    # CAVEAT: offline we cannot subtract the seed's INHERITED points (they are
    # not in the CSV), so re-touching a point BELOW the seed counts here, unlike
    # the in-training `prog` EMA which does subtract them. Treat this as an
    # upper bound, and use --from/--to for a specific link.
    print(f"\n  {'start':<18s} {'n':>7s} {'progress':>9s}  (reached >=1 new point;")
    print("                                            upper bound, see --to)")
    starts = order + sorted(set(by_start) - set(order))
    for s in starts:
        rs = by_start.get(s)
        if not rs:
            continue
        prog = sum(1 for r in rs if set(points(r)) - {s})
        print(f"  {str(s):<18s} {len(rs):7d} {100*prog/len(rs):8.1f}%")


if __name__ == "__main__":
    main()
