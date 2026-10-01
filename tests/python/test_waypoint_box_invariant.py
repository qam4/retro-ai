"""Jump detection boxes reach OUTSIDE their platform, and that is correct. Here is why.

This module used to assert the opposite. It enforced "a detection box must fit inside
the platform's standable span", which drove a change narrowing jump tolerances from
+-24 px to +-8 px and moving 20 anchors to make them fit. That change caused a measured
regression and was reverted; these tests now pin the reverted state and record the
reasoning, so the same idea is not re-derived from first principles a third time.

WHY A NARROW BOX IS WRONG FOR A JUMP

**A jump's landing position depends on the POLICY that jumped.** Measured on floor 7,
from the same seed pool, with two policies:

    v6 champion   lands at px 116                     box 112..128 covered 83%
    later policy  lands at px 108, then JUMPS to 136  the same box covered 1.8%

1.8% is also the figure `train_checkpoint_curriculum.py`'s tolerance comment recorded
from L4 v1, where `Rope1` read 1.8% while `Lclimb2_top` -- reachable only THROUGH
Rope1's platform -- read 87%. Narrowing reproduced a documented failure exactly.

Cost, in a controlled run (v6's warm start, v6's seed, only the code differing): `Low1`
and `Low2_launch` went 0.61 -> 0.00. The agent stopped reaching the rope-2 launch pad.

THE DISTINCTION THAT MATTERS

A box reaching into the VOID is harmless: detection is pose-gated, so no grounded frame
can occur there. A box covering a platform's LETHAL EDGE -- where the agent reads
grounded and falls on the next step -- poisons that waypoint's seed POOL, which is what
happened to `Low2_launch` (100 seeds, all unrecoverable). Those are different problems.
The fix for the second belongs in CAPTURE admission, not in detection width: a pool
needs states you can play from, while a reach metric must tolerate policy variance.
Not implemented yet.
"""

from __future__ import annotations

from retro_ai.games import yeti
from retro_ai.training.yeti_map import (
    get_level_map,
    jump_waypoints,
    standable_span,
    waypoint_tolerance,
)

LADDER_TOL, JUMP_TOL = 2, 6


def _boxes(level):
    """(wp_id, tol, box_lo, box_hi, span, is_jump) per positional waypoint."""
    lvl = get_level_map(level)
    if not lvl.platforms:
        return []
    jump = set(jump_waypoints(lvl))
    out = []
    for wid, (x, _y, floor) in yeti.waypoints(level).items():
        span = standable_span(lvl, floor)
        if span is None:
            continue
        tol = JUMP_TOL if wid in jump else LADDER_TOL
        c = x * 4 + 8
        out.append((wid, tol, c - 4 * tol, c + 4 * tol, span, wid in jump))
    return out


def test_jump_tolerance_is_flat_and_wide():
    """The reverted state: jump waypoints get a flat +-24 px box, ladders +-8 px.
    `waypoint_tolerance` still exists as a measurement helper but must NOT be applied to
    detection -- that application is what regressed."""
    assert JUMP_TOL == 6
    lvl = get_level_map(4)
    # the helper would narrow most jump boxes; detection must not use it
    narrowed = [
        wid
        for wid, (x, _y, f) in yeti.waypoints(4).items()
        if wid in set(jump_waypoints(lvl))
        and waypoint_tolerance(lvl, f, x, JUMP_TOL) != JUMP_TOL
    ]
    assert len(narrowed) >= 15, "helper unexpectedly agrees with the flat tolerance"


def test_l4_anchors_come_from_a_multi_policy_census():
    """Six measured overrides, from TWO different kinds of evidence. The 20-anchor
    geometric derivation was reverted; neither basis here is geometric.

    Per-episode WORST-POLICY detection score (debug/l4_anchor_recommend.py):

        Rope1  27 -> 25   1.00 both ways; 25 centres the modal landing instead of
                          putting it on the box edge (anchor 28 scored 1.00 on one
                          policy and 0.00 on another, for exactly that reason)
        Spring 48 -> 51   worst-policy 0.12 -> 0.80
        Step   60 -> 66   worst-policy 0.00 -> 0.92, i.e. it never marked at all

    Direct STANDABILITY measurement -- walk there, hold NOOP 30 frames, both approach
    directions, 8 seeds each. Floor 12's usable span is px 188..224:

        Low2_launch 44 -> 45   px 184 falls 0/8, px 188 stands 8/8. Re-landed after
                               e667206's wholesale revert dropped it while its comment
                               kept claiming it.
        Low1        56 -> 54   px 232 falls 0/8, px 224 stands 8/8. Inert in practice
                               (that pool is healthy) but the potential should not aim
                               at a pixel the agent can never occupy.
    """
    lvl = get_level_map(4)
    assert lvl.jump_waypoint_pos == {
        "Fr1": (64, 158),
        "Rope1": (25, 118),
        "Spring": (51, 94),
        "Step": (66, 102),
        "Low2_launch": (45, 70),
        "Low1": (54, 70),
    }


def test_floor12_anchors_are_inside_the_MEASURED_usable_span():
    """The span is px 188..224, measured directly, not derived from the tile extent.

    Floor 12's recorded extent is [184, 232) and the naive reading -- x_min+4 .. x_max-4
    = 188..228 -- is what `standable_span`'s docstring used to say. px 228 actually
    FALLS
    (0/8, both approach directions), so the right end is 224. This test exists because
    two anchors sat outside the span for weeks: `Low2_launch` on px 184 (which poisoned
    81 of its 100 seeds) and `Low1` on px 232 (which happened to be harmless).

    Falls back to the ANCHOR SOURCE for a waypoint that is no longer emitted.
    `Low2_launch` joined `jump_waypoint_skip` on 2026-10-01, to train the rope-2 crossing
    from `Low1` instead, so it is gone from `yeti.waypoints(4)`. The skip is a curriculum
    decision while this guard is about the MEASUREMENT, which outlives it -- reading only
    the emitted set would have let the guard vanish silently with the waypoint.
    """
    lo, hi = 188, 224
    emitted = yeti.waypoints(4)
    lvl = get_level_map(4)
    skipped = set(lvl.jump_waypoint_skip or ())
    for wid in ("Low2_launch", "Low1"):
        if wid in emitted:
            x, _y, floor = emitted[wid]
            assert floor == 12, f"{wid} is on floor {floor}, not 12"
        else:
            assert wid in skipped, f"{wid} is neither emitted nor deliberately skipped"
            x, y = lvl.jump_waypoint_pos[wid]
            assert y == lvl.floor_top_y[12], f"{wid} anchor y {y} is not floor 12's"
        px = x * 4 + 8
        assert (
            lo <= px <= hi
        ), f"{wid} anchor px {px} outside the measured span {lo}..{hi}"


def test_ladder_boxes_stay_inside_their_platform():
    """Ladders are exact positions on wide platforms, so their tight boxes fit. This is
    the one place the fit-inside property does hold, and it is why ladder waypoints were
    never the ones misbehaving."""
    for wid, _tol, lo, hi, (slo, shi), is_jump in _boxes(4):
        if not is_jump:
            assert lo >= slo and hi <= shi, f"L4 ladder {wid} overflows"


def test_jump_boxes_DO_overflow_and_that_is_intended():
    """Recorded, not deplored. 20 of 29 L4 boxes reach outside the standable span, and
    all 20 are jump waypoints. Making this number 0 is what broke the level.

    Was 21 of 30 until 2026-10-01, when `Low2_launch` joined `jump_waypoint_skip`. The
    count is a census of the current universe, not an invariant -- it is here so that a
    change in WHICH boxes overflow has to be acknowledged, and dropping a waypoint is an
    acknowledged change.
    """
    bad = [w for w, _t, lo, hi, (slo, shi), _j in _boxes(4) if lo < slo or hi > shi]
    jump = set(jump_waypoints(get_level_map(4)))
    assert len(bad) == 20
    assert set(bad) <= jump


def test_rope1_box_covers_both_measured_landings():
    """One box must cover landings from DIFFERENT policies: v6's champion lands at
    px 116, a later policy at px 108. The tol-2 reward box (fixed at 2 regardless of the
    curriculum's tolerance) must contain both, and the wide curriculum box must too."""
    x, _y, floor = yeti.waypoints(4)["Rope1"]
    c = x * 4 + 8
    lo, hi = c - 4 * JUMP_TOL, c + 4 * JUMP_TOL
    for landing in (108, 116):
        assert (
            lo <= landing <= hi
        ), f"px {landing} outside Rope1 curriculum box {lo}..{hi}"
    # The REWARD box is tol 2 whatever the curriculum uses. Rope1 is mandatory, so a box
    # that misses the landing leaves a distance term switched on for the whole episode.
    for landing in (108, 116):
        assert c - 8 <= landing <= c + 8, f"Rope1 reward box misses px {landing}"
    assert floor == 7


def test_mandatory_reward_boxes_cover_a_measured_grounded_position():
    """The `Fr1` / `Spring` / `Step` defect class: a mandatory milestone whose tol-2
    reward box contains no position the agent is ever grounded at can never be marked,
    so its distance term never switches off. Modal grounded positions, measured.
    """
    modes = {"Rope1": (7, [108, 116]), "Spring": (9, [212]), "Step": (10, [256, 272])}
    for wid, (floor, obs) in modes.items():
        x, _y, f = yeti.waypoints(4)[wid]
        assert f == floor
        c = x * 4 + 8
        assert any(
            c - 8 <= o <= c + 8 for o in obs
        ), f"{wid} reward box {c - 8}..{c + 8} contains none of {obs}"


def test_standable_span_still_available_as_a_measurement_tool():
    """Kept for debug/l4_edge_limit.py and debug/yeti_standable_audit.py. Its per-floor
    numbers were measured by walking to the edge and confirming the stance survives a
    NOOP hold, which is useful -- the APPLICATION to detection was the wrong part."""
    lvl = get_level_map(4)
    for floor, (lo_m, hi_m) in {12: (188, 228), 3: (256, 276), 9: (208, 232)}.items():
        lo, hi = standable_span(lvl, floor)
        assert lo >= lo_m and hi <= hi_m


def test_l3_untouched():
    """L3 is out of scope and must not have moved."""
    assert get_level_map(3).jump_waypoint_pos in (None, {})
