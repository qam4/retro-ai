"""A detection box must fit inside its platform's standable span.

Every waypoint defect found on this game is the same violation of that one rule:

* ``Low2_launch`` was anchored at floor 12's tile edge and its box straddled the brink,
  so its seed pool filled with states that read as grounded and fell one step later --
  all 100 of them. The pool meant to teach the rope-2 crossing taught falling.
* ``Low1_launch``'s +-24 px box reached a ladder 24 px away and reported a stalled climb
  as an arrival: reach 0.36 against 0.03 for the same event.
* ``Fr1`` sat on a platform extremity the agent never occupies, so the reward could
  never mark it and its distance term never switched off.

These tests pin the rule and RECORD the current violations, so the scale of the problem
is a number in CI rather than something rediscovered per run. They are written to pass
against today's map: the expected-violation counts are asserted, so fixing anchors will
fail these tests and force the counts down deliberately.
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
    """(wp_id, requested_tol, box_lo, box_hi, span, is_jump) per positional waypoint."""
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


def test_standable_span_is_inside_every_measured_limit():
    """The span is [x_min+8, x_max-8], chosen to be conservative rather than exact.
    These are the limits measured per floor by the NOOP-confirmed edge probe; the span
    must be a SUBSET of each, or a box that "fits" could still sit off the platform."""
    lvl = get_level_map(4)
    measured = {12: (188, 228), 3: (256, 276), 9: (208, 232)}
    for floor, (lo_m, hi_m) in measured.items():
        lo, hi = standable_span(lvl, floor)
        assert lo >= lo_m, f"floor {floor}: span starts left of the measured limit"
        assert hi <= hi_m, f"floor {floor}: span ends right of the measured limit"
    # floor 13's left probe stalled, so only its right limit is known (124).
    assert standable_span(lvl, 13)[1] <= 124


def test_l4_ladder_boxes_all_fit():
    """Ladders are exact positions on wide platforms, which is why they are the
    waypoints that work. All nine L4 ladders pass while all 21 L4 jump waypoints fail --
    that contrast is the evidence the defect is in how jump anchors are derived, not in
    waypoints generally."""
    for wid, _tol, lo, hi, (slo, shi), is_jump in _boxes(4):
        if not is_jump:
            assert lo >= slo and hi <= shi, f"L4 ladder {wid} overflows"


def test_l3_has_three_bad_LADDER_anchors():
    """L3 is worse than L4: three of its ladder anchors also overflow, on the goat
    platform (safe span 80..104).

    `Lesc_top` is the extreme case at px 132..148 -- 28 px clear of the platform, which
    matches the "0 standable positions" already recorded in level3_notes.md. These are a
    DIFFERENT bug from the jump-anchor derivation and need their own measurement; they
    are pinned here so they are not lost again."""
    bad = [
        w
        for w, _t, lo, hi, (slo, shi), is_jump in _boxes(3)
        if not is_jump and (lo < slo or hi > shi)
    ]
    assert sorted(bad) == ["Lesc_top", "Lgoat_a_top", "Lgoat_b_top"]


HI_CHAIN = frozenset(
    {
        "Hi1",
        "Hi1_launch",
        "Hi2",
        "Hi2_launch",
        "Hi3",
        "Hi3_launch",
        "Hi4",
        "Hi4_launch",
        "Hi5",
        "Hi5_launch",
    }
)


def _violations(level, tol_ceiling=2):
    """Waypoints whose DERIVED box still reaches outside their standable span."""
    lvl = get_level_map(level)
    if not lvl.platforms:
        return []
    bad = []
    for wid, (x, _y, floor) in yeti.waypoints(level).items():
        span = standable_span(lvl, floor)
        if span is None:
            continue
        tol = waypoint_tolerance(lvl, floor, x, tol_ceiling)
        c = x * 4 + 8
        if c - 4 * tol < span[0] or c + 4 * tol > span[1]:
            bad.append(wid)
    return sorted(bad)


def test_l4_has_no_violations_left_at_all():
    """THE GOAL, and the guard against regressing it. Every one of L4's 30 waypoints has
    a box entirely inside its platform's standable span. Was 21 violations.

    The fix was to DERIVE each jump anchor: place it against the edge facing the other
    platform, which is where the agent departs from or arrives at, but pulled inside far
    enough that the whole box fits -- and derive the tolerance per platform rather than
    using a flat +-24 px."""
    assert _violations(4) == []


def test_hi_chain_detects_at_tolerance_zero():
    """Route B's platforms are 16 px wide, so the span is a SINGLE centre and the
    derived tolerance is 0. That still detects: x advances one 4-px unit per step,
    so an exact value cannot be skipped.

    The anchors had to move for this to hold. At tol 0 on the old edge-derived
    anchor the box sat OFF the platform, so narrowing the tolerance without
    re-anchoring would turn "detects occasionally" into "detects never"."""
    lvl = get_level_map(4)
    for wid in HI_CHAIN:
        x, _y, floor = yeti.waypoints(4)[wid]
        lo, hi = standable_span(lvl, floor)
        c = x * 4 + 8
        assert lo <= c <= hi, f"{wid} anchor px {c} outside span {lo}..{hi}"


def test_l3_violations_recorded_out_of_scope():
    """L3 is explicitly out of scope for now; recorded so a later fix has a baseline.

    13 boxes overflowed with the old flat tolerance; deriving the tolerance rescues
    two of them (`Lgoat_a_top` and `Lgoat_b_top`, whose anchors DO sit inside their span
    and just needed a narrower box), leaving 11. Of those, `Lesc_top` is a bad ANCHOR,
    28 px clear of its platform, matching the "0 standable positions" already in
    level3_notes.md; A1..A5 are the ascent chain, which needs the same anchor derivation
    L4 just got."""
    bad = _violations(3, tol_ceiling=6)
    assert len(bad) == 11
    assert "Lesc_top" in bad
    assert {f"A{i}" for i in range(1, 6)} <= set(bad)


def test_the_defect_was_structural_not_a_list_of_mistakes():
    """All 21 original L4 violations were jump waypoints and all nine ladders passed,
    because `jump_waypoints` derives anchors from platform EDGES while the standable run
    stops short of them, and the flat +-24 px jump tolerance needs a 48 px span most of
    these platforms do not have. One cause, 21 symptoms."""
    lvl = get_level_map(4)
    jump = set(jump_waypoints(lvl))
    with_old_flat_tolerance = [
        w for w, _t, lo, hi, (slo, shi), _j in _boxes(4) if lo < slo or hi > shi
    ]
    assert set(with_old_flat_tolerance) <= jump


def test_waypoint_tolerance_only_ever_narrows_and_never_overflows():
    lvl = get_level_map(4)
    for wid, (x, _y, floor) in yeti.waypoints(4).items():
        for want in (0, 1, 2, 6):
            got = waypoint_tolerance(lvl, floor, x, want)
            assert 0 <= got <= want
            span = standable_span(lvl, floor)
            c = x * 4 + 8
            if span[0] <= c <= span[1]:
                # anchor inside the span => the returned box must fit
                assert c - 4 * got >= span[0] and c + 4 * got <= span[1]


def test_hi_chain_platforms_admit_no_tolerance_at_all():
    """Two-tile platforms have a single safe centre, so the constraint is genuinely
    per-platform and no global tolerance can be correct."""
    lvl = get_level_map(4)
    for floor in (15, 16, 17, 18):
        lo, hi = standable_span(lvl, floor)
        assert lo == hi, f"floor {floor} span {lo}..{hi} unexpectedly wide"
