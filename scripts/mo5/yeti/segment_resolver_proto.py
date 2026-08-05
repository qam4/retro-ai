#!/usr/bin/env python3
"""Prototype: resolve which GRAPH SEGMENT (horizontal floor or vertical ladder)
the agent is on from (pixel_x, pixel_y), and compute path-distance from a point
ON that segment. Standalone (no library edits) so we can evaluate the idea and
its ambiguity before wiring it into the reward.

Segment precedence:
  1. HORIZONTAL floor: exactly the current agent_floor_from_pixel_xy rule
     (platform extent + y tol, or y-only on L1/L2). If it resolves -> ("floor").
  2. VERTICAL ladder: x within X_TOL of a ladder centre AND y strictly between
     its endpoints -> ("ladder", ...). Nearest-x ladder wins.
Otherwise None.

Distance from a point:
  floor  -> today's rule (min over same-floor nodes of |x-node.x| + dist).
  ladder -> min(|y - y_top| + dist[top->target], |y - y_bot| + dist[bot->target]),
            i.e. how far along the vertical edge to each endpoint, then graph.
"""
from __future__ import annotations

from retro_ai.training.yeti_map import (
    agent_floor_from_pixel_xy,
    build_navigation_map,
    get_level_map,
)

X_TOL = 6  # px — real ladder climbs sit at cx exactly; escalator is ~4px off


def resolve_segment(pix_x, pix_y, level):
    m = get_level_map(level)
    floor = agent_floor_from_pixel_xy(pix_x, pix_y, level)
    if floor is not None:
        return ("floor", floor, None, None)
    best = None
    for name, top_f, bot_f, cx in m.ladders:
        y_top, y_bot = m.floor_top_y[top_f], m.floor_top_y[bot_f]
        lo, hi = min(y_top, y_bot), max(y_top, y_bot)
        if abs(pix_x - cx) <= X_TOL and lo - 2 <= pix_y <= hi + 2:
            dx = abs(pix_x - cx)
            if best is None or dx < best[1]:
                best = (("ladder", name, y_top, y_bot), dx)
    return best[0] if best else None


def seg_distance(nav, seg, pix_x, pix_y, target):
    kind = seg[0]
    if kind == "floor":
        return nav.path_distance_from_agent(seg[1], pix_x, target)
    _, name, y_top, y_bot = seg
    ti = nav.node_by_ident[target]
    top_i = nav.node_by_ident[f"{name}_top"]
    bot_i = nav.node_by_ident[f"{name}_bot"]
    return min(
        abs(pix_y - y_top) + nav.dist[top_i][ti],
        abs(pix_y - y_bot) + nav.dist[bot_i][ti],
    )


def demo_escalator():
    print(
        "=== L3 escalator ride: OLD (floor) vs NEW (segment) distance to "
        "Ldown_bot ===",
        flush=True,
    )
    nav = build_navigation_map(3)
    pix_x = 32 * 4 + 8  # agent ax=32 -> pixel centre 136 (Lesc cx=140)
    print("   y   seg           OLD_floor OLD_dist  NEW_dist", flush=True)
    for y in range(94, 162, 4):
        seg = resolve_segment(pix_x, y, 3)
        old_floor = agent_floor_from_pixel_xy(pix_x, y, 3)
        old_dist = (
            nav.path_distance_from_agent(old_floor, pix_x, "Ldown_bot")
            if old_floor is not None
            else None
        )
        new_dist = seg_distance(nav, seg, pix_x, y, "Ldown_bot") if seg else None
        segstr = seg[0] if seg else "None"
        if seg and seg[0] == "ladder":
            segstr = f"ladder:{seg[1]}"
        print(
            f"  {y:3d}  {segstr:14} {str(old_floor):>7}  "
            f"{str(old_dist):>7}  {str(new_dist):>7}",
            flush=True,
        )


def demo_ladder(level, target):
    print(
        f"\n=== L{level} ladder sweep: NEW resolves mid-ladder (OLD holds) ===",
        flush=True,
    )
    m = get_level_map(level)
    nav = build_navigation_map(level)
    name, top_f, bot_f, cx = m.ladders[0]
    y_top, y_bot = m.floor_top_y[top_f], m.floor_top_y[bot_f]
    print(f"  ladder {name}: x={cx} y {y_bot}->{y_top}; target={target}", flush=True)
    for y in range(max(y_top, y_bot), min(y_top, y_bot) - 1, -3):
        seg = resolve_segment(cx, y, level)
        old_floor = agent_floor_from_pixel_xy(cx, y, level)
        new_dist = seg_distance(nav, seg, cx, y, target) if seg else None
        segstr = seg[0] if seg else "None"
        if seg and seg[0] == "ladder":
            segstr = f"ladder:{seg[1]}"
        print(
            f"  y{y:3d} seg={segstr:12} old_floor={str(old_floor):>4} "
            f"new_dist={new_dist}",
            flush=True,
        )


def ambiguity_scan():
    print("\n=== ambiguity scan (x px 0..319, y px 20..190) ===", flush=True)
    for level in (1, 2, 3):
        m = get_level_map(level)
        both = 0  # matches a floor AND a ladder
        multi = 0  # matches >= 2 ladders
        floor_ex = ladder_ex = None
        for px in range(0, 320, 4):
            for py in range(20, 192, 2):
                floor = agent_floor_from_pixel_xy(px, py, level)
                lad = []
                for name, tf, bf, cx in m.ladders:
                    yt, yb = m.floor_top_y[tf], m.floor_top_y[bf]
                    lo, hi = min(yt, yb), max(yt, yb)
                    if abs(px - cx) <= X_TOL and lo - 2 <= py <= hi + 2:
                        lad.append(name)
                if floor is not None and lad:
                    both += 1
                    floor_ex = floor_ex or (px, py, floor, lad)
                if len(lad) >= 2:
                    multi += 1
                    ladder_ex = ladder_ex or (px, py, lad)
        print(
            f"  L{level}: floor&ladder overlap cells={both} "
            f"(e.g. {floor_ex}); multi-ladder cells={multi} "
            f"(e.g. {ladder_ex})",
            flush=True,
        )


if __name__ == "__main__":
    demo_escalator()
    demo_ladder(1, "princess")
    demo_ladder(2, "princess")
    ambiguity_scan()
