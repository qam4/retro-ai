#!/usr/bin/env python3
"""One-time migration: backfill the reached-MILESTONE set on existing seeds.

Seeds captured before the milestone-restore fix carry an EMPTY reached set, so
a seeded episode still treats every milestone behind it as pending and the
potential pays it to RETREAT (see experiments/003-yeti/level3_notes.md). Those
pools are too valuable to throw away (they hold the whole L3 ascent bootstrap),
so instead of discarding them we infer the set from each pool's ROUTE POSITION:
a seed in pool P has necessarily passed every milestone at-or-below P.

This is safe because it is exactly what the capture path now records live; it
only reconstructs history for pre-fix files. Route order is supplied per level
(display/semantics-free ordering, see curriculum_cp_wp_model.md).

Usage:
  python scripts/mo5/yeti/backfill_seed_milestones.py \
      --in  <run>/checkpoints.pkl --out <dir>/checkpoints.pkl
"""
from __future__ import annotations

import argparse
import pickle

from retro_ai.training.yeti_map import get_level_map

# L3 route order (bottom -> top). Launch pads sit just before their landing.
L3_ROUTE = [
    "Lgoat_a_top",
    "Lgoat_b_top",
    "Lesc_top",
    "Ldown_bot",
    "Lsc1_top",
    "Lsc2_top",
    "Lsc3_top",
    "Lsc4_top",
    "A1_launch",
    "A1",
    "A2_launch",
    "A2",
    "A3_launch",
    "A3",
    "A4_launch",
    "A4",
    "A5_launch",
    "A5",
    "Lprincess_top",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--level", type=int, default=3)
    args = ap.parse_args()

    m = get_level_map(args.level)
    idx = {wid: i for i, wid in enumerate(L3_ROUTE)}
    # Record ALL route ids at-or-below a pool, not just the milestone ones: the
    # reward resolves which of them are milestones (and it accepts either naming
    # scheme — the curriculum's "A1" or the graph's "J10_11_b"), so we don't have
    # to replicate that mapping here.
    _ = m

    with open(args.src, "rb") as f:
        data = pickle.load(f)

    changed = 0
    for wp_id, payload in data.get("waypoints", {}).items():
        states, goal_score = payload
        if wp_id not in idx:
            print(f"  ! {wp_id}: not in route order, left empty")
            continue
        # Every route point at or below this pool's position was passed.
        behind = {w for w in L3_ROUTE if idx[w] <= idx[wp_id]}
        new_states = []
        for s in states:
            s = tuple(s)
            stack = s[3] if len(s) >= 4 else None
            new_states.append(
                (int(s[0]), int(s[1]), bytes(s[2]), stack, frozenset(behind))
            )
            changed += 1
        data["waypoints"][wp_id] = (new_states, goal_score)
        print(f"  {wp_id:14s} <- {len(behind)} milestones, {len(new_states)} seeds")

    # Fruit CP pools: a collected-fruit state is above the whole ascent, so it
    # has passed every milestone except the princess ladder.
    for lvl, states in enumerate(data.get("checkpoints", [])):
        if not states or lvl == 0:
            continue
        behind = {w for w in L3_ROUTE if w != "Lprincess_top"}
        new_states = []
        for s in states:
            s = tuple(s)
            stack = s[3] if len(s) >= 4 else None
            new_states.append(
                (int(s[0]), int(s[1]), bytes(s[2]), stack, frozenset(behind))
            )
            changed += 1
        data["checkpoints"][lvl] = new_states
        print(f"  cp{lvl:<12d} <- {len(behind)} milestones, {len(new_states)} seeds")

    with open(args.dst, "wb") as f:
        pickle.dump(data, f)
    print(f"\nbackfilled {changed} seeds -> {args.dst}")


if __name__ == "__main__":
    main()
