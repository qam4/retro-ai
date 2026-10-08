#!/usr/bin/env python3
"""Validate the described L4 route against the geometry extracted from RAM.

Authoring a LevelMap by hand from a verbal route is exactly where a transposed
digit silently becomes a wrong reward target, so every step is checked here:
ladders must actually connect the two platforms claimed, and jumps must be
plausible (adjacent rows, a real gap, a reachable span). Anything that fails is
printed as a QUESTION rather than quietly corrected.
"""
from __future__ import annotations

import json

# Tracked copy (output/ is gitignored); regenerate with extract_level_map.py.
D = json.load(open("experiments/003-yeti/data/level_maps/level4_map_raw.json"))
SURF = {s["n"]: s for s in D["surface"]}
LADDERS = D["ladders"]
PLAT = {p["id"]: p for p in D["platforms"]}

# The route as described, in order. ("ladder", top_tile) | ("jump", a, b)
# | ("rope", a, b) | ("spring", a, b) | ("fruit",) | ("princess",)
ROUTE = [
    ("phase", "1. fetch the fruit (bottom right, through the kangaroo corridor)"),
    ("ladder", 128),
    ("jump", 131, 132),
    ("jump", 135, 123),
    ("fruit",),
    ("phase", "2. return to the start: 2 jumps back, then down ladder 128"),
    ("jump", 123, 135),
    ("jump", 132, 131),
    ("ladder", 128),
    ("phase", "3. climb the left side"),
    ("ladder", 116),
    ("ladder", 89),
    ("rope", 92, 93),
    ("ladder", 71),
    ("spring", 74, 75),
    ("jump", 78, 79),
    ("ladder", 62),
    ("phase", "4a. LOW route"),
    ("jump", 60, 59),  # confirmed: P12's left lip -> P11's right lip
    ("rope", 54, 53),
    ("phase", "4b. HIGH route"),
    ("ladder", 34),
    ("jump", 32, 31),
    ("jump", 30, 21),
    ("jump", 20, 16),
    ("jump", 15, 8),
    ("jump", 7, 14),
    ("ladder", 11),
    ("phase", "5. common finish"),
    ("ladder", 26),
    ("princess",),
]

PRINCESS = {"cols": (0, 1), "rows": (6, 7), "stands_on": "P7", "centre_px": (8, 64)}
SNOWBALLS = {"P7": (28, 29), "P10": (46, 53)}


def ladder_of(top_tile):
    s = SURF.get(top_tile)
    if s is None or s["kind"] != "ladder_top":
        return None, s
    for m in LADDERS:
        if m["top_row"] == s["row"] and m["col0"] <= s["col"] <= m["col1"]:
            return m, s
    return None, s


def platform_at_row(row, col_lo, col_hi):
    for p in D["platforms"]:
        if p["row"] == row and not (p["col1"] < col_lo or p["col0"] > col_hi):
            return p["id"]
    return None


problems = []
print(f"{'step':<26}{'detail':<58}{'check'}")
print("-" * 100)
for item in ROUTE:
    kind = item[0]
    if kind == "phase":
        print(f"\n--- {item[1]} ---")
        continue
    if kind == "fruit":
        b = D["blocks"]["L4-style"]
        detail = "block rows {} cols {}".format(b["rows"], b["cols"])
        print(f"{'collect fruit':<26}{detail:<58}on P21")
        continue
    if kind == "princess":
        detail = "cols {} rows {}".format(PRINCESS["cols"], PRINCESS["rows"])
        print(f"{'reach princess':<26}{detail:<58}stands on {PRINCESS['stands_on']}")
        continue
    if kind == "ladder":
        m, s = ladder_of(item[1])
        if m is None:
            problems.append(f"tile {item[1]} is not a ladder top")
            print(f"{'ladder ' + str(item[1]):<26}{'NOT A LADDER TOP':<58}FAIL")
            continue
        top_p = platform_at_row(m["top_row"], m["col0"], m["col1"])
        bot_p = platform_at_row(m["bot_row"], m["col0"], m["col1"])
        detail = (
            f"cols {m['col0']}-{m['col1']} body rows {m['row0']}-{m['row1']}  "
            f"{top_p or 'ground/none'} (row {m['top_row']}) <-> "
            f"{bot_p or 'ground/none'} (row {m['bot_row']})"
        )
        print(f"{'ladder ' + str(item[1]):<26}{detail:<58}ok")
        if bot_p is None and top_p is None:
            problems.append(f"ladder {item[1]} connects nothing recognised")
        continue
    a, b = item[1], item[2]
    sa, sb = SURF.get(a), SURF.get(b)
    if sa is None or sb is None:
        problems.append(f"{kind} {a}->{b}: unknown tile")
        print(f"{kind + ' ' + str(a) + '->' + str(b):<26}{'UNKNOWN TILE':<58}FAIL")
        continue
    drow = sb["row"] - sa["row"]
    dcol = sb["col"] - sa["col"]
    gap = abs(dcol) - 1
    detail = (
        f"{sa['platform']} r{sa['row']}c{sa['col']} -> {sb['platform']} "
        f"r{sb['row']}c{sb['col']}   drow {drow:+d} dcol {dcol:+d} gap {gap}"
    )
    verdict = "ok"
    if sa["platform"] == sb["platform"]:
        verdict = "SAME PLATFORM?"
        problems.append(f"{kind} {a}->{b}: both on {sa['platform']}")
    elif kind == "jump" and (abs(drow) > 1 or gap > 4):
        verdict = "IMPLAUSIBLE"
        problems.append(
            f"jump {a}->{b}: drow {drow:+d}, gap {gap} — too far for a jump"
        )
    print(f"{kind + ' ' + str(a) + '->' + str(b):<26}{detail:<58}{verdict}")

print("\n=== snowball platforms (from your note; motion map agrees) ===")
for p, (lo, hi) in SNOWBALLS.items():
    print(
        f"  {p}: tiles {lo}-{hi}  rows {PLAT[p]['row']}  "
        f"cols {PLAT[p]['col0']}-{PLAT[p]['col1']}"
    )

print("\n=== QUESTIONS ===" if problems else "\nno inconsistencies found")
for p in problems:
    print(f"  - {p}")
