"""Yeti snowballs, read from RAM: where they are and which way they go.

Decoded 2026-10-13 on L1 (experiments/003-yeti/snowball_rng_notes.md); drawn over real
frames to confirm. Up to six snowballs, one 8-byte slot each at ``SLOT_BASE + 8*k``:

    +0  y (px; a snowball rolling on a floor sits at that floor's standing y + 10)
    +1  x (RAM units; px = x*4 + 8, the same rule as the player)
    +4  active (non-zero while the slot holds a snowball)

Where each snowball goes at a ladder or a platform edge is pseudo-random. ``RNG_BYTE``
is part of the generator: writing another value there at a level start changes every
route and no spawn time. It is not the whole generator (L1's value written into L5's
start does not give L1's routes).

The analysis works on ``rows``: per game step, RAM ``WINDOW_LO .. WINDOW_LO + 255``
(see :func:`hazard_row`), so slot offsets below are relative to ``WINDOW_LO``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

WINDOW_LO = 0x2B00
WINDOW_LEN = 0x100
SLOT_BASE = 0x2B60
SLOT_STRIDE = 8
N_SLOTS = 6
RNG_BYTE = 0x2B24
# A rolling snowball's y is the standing y of the floor it rolls on plus this (measured
# on L1: snowballs roll at 64/96/128/160/192 on floors standing at 54/86/118/150/182).
ROLL_DY = 10


def hazard_row(iface) -> np.ndarray:
    """RAM ``WINDOW_LO .. +255`` now: the slots, generator byte and spawn clock."""
    ram = np.frombuffer(bytes(iface.read_ram()), dtype=np.uint8)
    return ram[WINDOW_LO : WINDOW_LO + WINDOW_LEN].copy()


def snowballs(rows: np.ndarray) -> List[dict]:
    """Every snowball in ``rows``, in spawn order.

    A snowball is a run of one slot from its spawn: the slot turning active, or its y
    jumping back up while active (a re-spawn at the top). A slot can read inactive for
    one or two steps mid-roll (seen on L5), so a gap of up to 2 steps that resumes
    within 8 px is the same snowball.

    Each entry: ``t0`` (spawn step), ``slot``, ``path`` [(y, x_px)] per step, ``route``
    [(y, x_px)] = for each floor it rolled on for at least 3 steps, the px where it
    left it (the last floor is dropped while the snowball is still alive at the end of
    ``rows``, because it has not left it yet), and ``alive``.
    """
    out = []
    for k in range(N_SLOTS):
        b = SLOT_BASE - WINDOW_LO + SLOT_STRIDE * k
        act = rows[:, b + 4]
        ys = rows[:, b].astype(int)
        xs = rows[:, b + 1].astype(int) * 4 + 8
        segs: List[list] = []
        cur, gap = None, 0
        for t in range(len(rows)):
            if not act[t]:
                gap += 1
                if gap > 2:
                    cur = None
                continue
            cont = (
                cur is not None
                and cur[1]
                and abs(ys[t] - cur[1][-1][0]) <= 8
                and abs(xs[t] - cur[1][-1][1]) <= 8 * (gap + 1)
            )
            if cur is None or (gap and not cont) or (t and ys[t] < ys[t - 1] - 8):
                cur = [t, []]
                segs.append(cur)
            gap = 0
            cur[1].append((int(ys[t]), int(xs[t])))
        for t0, path in segs:
            floors: List[Tuple[int, int]] = []
            i = 0
            while i < len(path):
                j = i
                while j < len(path) and path[j][0] == path[i][0]:
                    j += 1
                if j - i >= 3:
                    if floors and floors[-1][0] == path[i][0]:
                        floors[-1] = (path[i][0], path[j - 1][1])
                    else:
                        floors.append((path[i][0], path[j - 1][1]))
                i = j
            alive = t0 + len(path) >= len(rows)
            out.append(
                {
                    "t0": t0,
                    "slot": k,
                    "path": path,
                    "route": floors[:-1] if alive else floors,
                    "alive": alive,
                }
            )
    out.sort(key=lambda s: (s["t0"], s["slot"]))
    return out


def _floor_ladders(layout: int):
    from retro_ai.training.yeti_map import get_level_map

    lvl = get_level_map(layout)
    floor_of_y = {y + ROLL_DY: f for f, y in lvl.floor_top_y.items()}
    down: Dict[int, List[Tuple[str, int]]] = {}
    for name, top, _bot, px in lvl.ladders:
        down.setdefault(top, []).append((name, px))
    return floor_of_y, down


def choices(rows: np.ndarray, layout: int = 1) -> List[List[str]]:
    """Per snowball, the choice it made on each floor it left: ``"f<floor>:<ladder>"``,
    or ``"f<floor>:edgeL"`` / ``"edgeR"`` for rolling off the left / right end.

    Measured on L1's layout only (layout 1: L1 and L5). Platform-heavy layouts would
    need their own floor model.
    """
    if layout != 1:
        raise ValueError(
            f"snowball choices are only modelled for layout 1, not {layout}"
        )
    floor_of_y, down = _floor_ladders(layout)
    out = []
    for sb in snowballs(rows):
        seq = []
        for y, x in sb["route"]:
            fl = floor_of_y.get(y)
            if fl is None:
                continue  # the yeti's ledge at the top is not a floor of the map
            lad = [n for n, px in down.get(fl, []) if abs(px - x) <= 8]
            seq.append(f"f{fl}:{lad[0] if lad else ('edgeL' if x < 160 else 'edgeR')}")
        out.append(seq)
    return out


def compare(a: Sequence[Sequence[str]], b: Sequence[Sequence[str]]) -> Tuple[int, int]:
    """``(same, different)`` choices between two sequences, over what both saw."""
    same = diff = 0
    for i in range(min(len(a), len(b))):
        for j in range(min(len(a[i]), len(b[i]))):
            if a[i][j] == b[i][j]:
                same += 1
            else:
                diff += 1
    return same, diff


def group_sequences(seqs: Sequence[Sequence[Sequence[str]]]) -> List[int]:
    """A group id per sequence: greedy, in order; a sequence joins the first group
    all of whose members it never disagrees with. Two recordings of one sequence can
    only be told apart by a choice both saw, so a short recording joins any group it
    fits."""
    groups: List[List[int]] = []
    gid: List[Optional[int]] = [None] * len(seqs)
    for i, s in enumerate(seqs):
        for g, members in enumerate(groups):
            if all(compare(s, seqs[m])[1] == 0 for m in members):
                members.append(i)
                gid[i] = g
                break
        else:
            groups.append([i])
            gid[i] = len(groups) - 1
    return [int(g) for g in gid]


def route_choices(rows: np.ndarray, x_step: int = 8) -> List[List[str]]:
    """:func:`choices` without a floor model, for any layout: per hazard, where it left
    each surface it rolled or walked on, ``"<y>:<x // x_step>"``.

    It needs no ladder or platform list, so it works on L2-L4, whose layouts
    :func:`choices` does not model. Checked against :func:`choices` on L1's 299 l1_r1
    respawn recordings: with ``x_step`` 8 (or 16, or 32) :func:`group_sequences` gives
    the same 31 groups, and no pair of recordings is grouped differently. At 4 it
    splits recordings :func:`choices` joins, because where a snowball leaves a floor
    wobbles by a few pixels.

    What it does not see is TIMING: two starts whose hazards take the same routes at
    different moments are one group. On L2, whose goats walk rather than choose, that is
    most of what varies between starts (``tests/python/test_yeti_hazards.py``).
    """
    return [[f"{y}:{x // x_step}" for y, x in sb["route"]] for sb in snowballs(rows)]


__all__ = [
    "WINDOW_LO",
    "SLOT_BASE",
    "SLOT_STRIDE",
    "N_SLOTS",
    "RNG_BYTE",
    "ROLL_DY",
    "hazard_row",
    "snowballs",
    "choices",
    "compare",
    "group_sequences",
    "route_choices",
]
