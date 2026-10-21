"""Snowball decoding on synthetic RAM rows (no emulator needed)."""

from __future__ import annotations

import numpy as np
from retro_ai.games import yeti_hazards as hz


def _rows(track, n=None, slot=0):
    """Rows with one slot following ``track`` = [(y, x_px) or None per step]."""
    n = n or len(track)
    rows = np.zeros((n, hz.WINDOW_LEN), dtype=np.uint8)
    b = hz.SLOT_BASE - hz.WINDOW_LO + hz.SLOT_STRIDE * slot
    for t, p in enumerate(track):
        if p is None:
            continue
        y, px = p
        rows[t, b], rows[t, b + 1], rows[t, b + 4] = y, (px - 8) // 4, 1
    return rows


def _roll(y, x0, x1, step=4):
    d = step if x1 >= x0 else -step
    return [(y, x) for x in range(x0, x1 + d, d)]


def test_a_route_names_the_ladder_or_edge_left_on_each_floor():
    # spawn on the yeti ledge, drop to floor 5, roll left PAST L45 (px 208) to the
    # left end, drop to floor 4, roll right and go down L34 (px 176), reach floor 3.
    track = (
        _roll(32, 272, 264)
        + _roll(64, 264, 152)
        + [(80, 152)]
        + _roll(96, 152, 176)
        + [(112, 176)]
        + _roll(128, 176, 164)
    )
    sb = hz.snowballs(_rows(track))
    assert len(sb) == 1 and sb[0]["t0"] == 0 and sb[0]["alive"]
    assert hz.choices(_rows(track)) == [["f5:edgeL", "f4:L34"]]


def test_a_short_inactive_gap_is_the_same_snowball():
    """Seen on L5: slot 5 read inactive for one step mid-roll."""
    track = _roll(64, 200, 172)
    track = track[:5] + [None] + track[6:]
    assert len(hz.snowballs(_rows(track))) == 1


def test_a_respawn_at_the_top_is_a_new_snowball():
    track = _roll(64, 264, 240) + _roll(32, 272, 264) + _roll(64, 264, 240)
    sb = hz.snowballs(_rows(track))
    assert [s["t0"] for s in sb] == [0, 7]


def test_grouping_needs_a_disagreement_to_split():
    a = [["f5:L45", "f4:edgeR"], ["f5:edgeL"]]
    b = [["f5:L45"]]  # shorter recording of the same sequence
    c = [["f5:edgeL", "f4:L34"]]  # differs on snowball 0, floors 5 and 4
    assert hz.compare(a, b) == (1, 0)
    assert hz.compare(a, c) == (0, 2)
    assert hz.group_sequences([a, b, c, a]) == [0, 0, 1, 0]


def test_route_choices_needs_no_floor_model():
    """Same track as above, on a layout `choices` cannot read: where it left each
    surface, in y and 8-px columns, and nothing about which ladder that was."""
    track = (
        _roll(32, 272, 264)
        + _roll(64, 264, 152)
        + [(80, 152)]
        + _roll(96, 152, 176)
        + [(112, 176)]
        + _roll(128, 176, 164)
    )
    assert hz.route_choices(_rows(track)) == [["32:33", "64:19", "96:22"]]


def test_route_choices_groups_by_route_not_by_timing():
    """Two starts whose snowball takes the same route a few steps apart are one group;
    a different route is another. That is the limit noted in route_choices."""
    route = _roll(64, 264, 152) + [(80, 152)] + _roll(96, 152, 160)
    late = [None] * 5 + route
    other = _roll(64, 264, 208) + [(80, 208)] + _roll(96, 208, 216)
    seqs = [hz.route_choices(_rows(t)) for t in (route, late, other)]
    assert hz.group_sequences(seqs) == [0, 0, 1]
