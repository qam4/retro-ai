"""Sprite-overlap reach test: does the agent's SPRITE contain the anchor POINT?

The inverse of the historical test, which asks whether the agent's POSITION falls inside
a tolerance box. Motivation, measured on L4 floor 7: the agent lands at px 108, is
grounded for TWO FRAMES, then is airborne (pose 9) all the way to the ladder at px 144.
With a 4 px y-window plus a grounded allowlist, a waypoint there can only fire in those
two frames, so an anchor 4 px away detects nothing -- anchor 28 scored 1.00 against one
policy and 0.00 against another that landed 4 px further left.

These tests pin the geometry and the opt-in default. They do NOT assert that sprite mode
trains better; that is a training question.
"""

import pytest
from retro_ai.games import yeti
from retro_ai.training.targets import (
    SPRITE_H,
    SPRITE_W,
    reaches,
    sprite_overlaps,
    within_tol,
)

# Rope1 on L4: anchor y is 118, the standing sprite TOP on floor 7.
ROPE1_Y = 118
GOOD = 25  # px 108, the modal landing
BROKE = 28  # px 120, the anchor that scored 0.00 against a later policy


def test_measured_sprite_extents():
    """14 px wide, 18 px tall, measured from frames at known RAM positions."""
    assert (SPRITE_W, SPRITE_H) == (14, 18)


def test_x_axis_is_slightly_TIGHTER_than_the_old_box_at_tol_2():
    """In x the two tests are close but NOT equal, and the difference matters.

    ``tol=2`` is +-8 px. The sprite spans centre-7..centre+6 (14 is even, so the span is
    asymmetric). So sprite mode is one x_ram unit tighter on each side: it agrees for
    |dx| <= 1 and rejects |dx| = 2, which the box accepts. Sprite mode is therefore NOT
    a pure loosening -- it trades x tightness for a much larger y window.
    """
    for dx in (-1, 0, 1):
        agent = GOOD + dx
        assert within_tol((GOOD, ROPE1_Y), agent, ROPE1_Y, 2)
        assert sprite_overlaps((GOOD, ROPE1_Y), agent, ROPE1_Y), f"dx={dx}"
    for dx in (-2, 2):
        agent = GOOD + dx
        assert within_tol((GOOD, ROPE1_Y), agent, ROPE1_Y, 2), f"box accepts dx={dx}"
        assert not sprite_overlaps(
            (GOOD, ROPE1_Y), agent, ROPE1_Y
        ), f"sprite rejects dx={dx}"


def test_y_window_is_the_whole_sprite_not_the_tolerance():
    """The point of the change. A grounded agent's sprite spans y..y+17, so an anchor at
    the standing line is contained for the whole descent of a jump arc."""
    for y in range(ROPE1_Y - SPRITE_H + 1, ROPE1_Y + 1):
        assert sprite_overlaps((GOOD, ROPE1_Y), GOOD, y), f"y={y} should overlap"
    # one pixel further and the sprite's bottom no longer reaches the anchor
    assert not sprite_overlaps((GOOD, ROPE1_Y), GOOD, ROPE1_Y - SPRITE_H)
    # below the line the sprite's top has passed it
    assert not sprite_overlaps((GOOD, ROPE1_Y), GOOD, ROPE1_Y + 1)


def test_airborne_mid_arc_fires_under_sprite_but_not_under_box():
    """The measured L4 case: airborne at px 112 (x_ram 26), y 110, sprite spans 110..127
    which contains floor 7's standing line at 118. The box test misses it on y."""
    assert sprite_overlaps((GOOD, ROPE1_Y), 26, 110)
    assert not within_tol((GOOD, ROPE1_Y), 26, 110, 2)


def test_grounded_landing_is_detected_by_both():
    """Whatever else changes, the actual landing must still register."""
    assert within_tol((GOOD, ROPE1_Y), GOOD, ROPE1_Y, 2)
    assert sprite_overlaps((GOOD, ROPE1_Y), GOOD, ROPE1_Y)


def test_the_broken_anchor_still_misses_the_landing_frame():
    """Sprite overlap is not magic: at the landing frame itself, anchor 28 is 12 px away
    in x and still does not fire. It recovers on the AIRBORNE frames that pass px 120,
    which is why the pose gate has to stop failing closed for this to help."""
    assert not within_tol((BROKE, ROPE1_Y), GOOD, ROPE1_Y, 2)
    assert not sprite_overlaps((BROKE, ROPE1_Y), GOOD, ROPE1_Y)
    # ... and DOES fire once the agent's sprite passes over px 120 mid-arc
    assert sprite_overlaps((BROKE, ROPE1_Y), 29, 112)


def test_dispatcher_defaults_to_box_so_nothing_changes_unless_opted_in():
    for a in (GOOD, BROKE):
        for x in range(20, 40):
            for y in range(100, 140):
                assert reaches((a, ROPE1_Y), x, y, 2) == within_tol(
                    (a, ROPE1_Y), x, y, 2
                )


def test_dispatcher_sprite_mode_ignores_tolerance_entirely():
    """The sprite's size IS the tolerance, so passing a different tol must not change
    the answer -- otherwise the two consumers could still drift via their tolerances."""
    for tol in (0, 1, 2, 6, 24):
        assert reaches((GOOD, ROPE1_Y), 26, 110, tol, mode="sprite") is True
        assert reaches((BROKE, ROPE1_Y), GOOD, ROPE1_Y, tol, mode="sprite") is False


def test_unknown_mode_is_rejected_loudly():
    with pytest.raises(ValueError, match="unknown reach mode"):
        reaches((GOOD, ROPE1_Y), GOOD, ROPE1_Y, 2, mode="whatever")


def test_pose_blocklist_fails_open_where_the_allowlist_failed_closed():
    """`SURFACE_POSES` omitted poses 6 and 7 -- half the leftward walk cycle -- for the
    project's whole history, and pose 15 went uncatalogued just as long. An allowlist
    turns every such omission into silent suppression; a blocklist only excludes what it
    names.

    This test used pose 15 as its live example of an uncatalogued code. It is no longer
    one: on 2026-09-11 it was identified as the LEFTWARD rope carry, i.e. the rope-2
    crossing itself -- which is exactly the point of the blocklist. Because the gate
    fails open, that unknown pose still counted for reach the whole time it was unnamed.
    Had reach been gated on the `SURFACE_POSES` allowlist instead, L4's wall manoeuvre
    would have been invisible to detection. The example is now a hypothetical code."""
    assert yeti.NON_TRAVERSAL_POSES == frozenset({11, 12})
    # the poses that were historically missing now pass a blocklist gate
    for p in (6, 7):
        assert p not in yeti.NON_TRAVERSAL_POSES
    # the once-unknown pose 15 is catalogued now, and still counts for reach
    assert 15 in yeti.KNOWN_POSES
    assert 15 not in yeti.NON_TRAVERSAL_POSES
    # an UNCATALOGUED pose passes the blocklist but would fail the allowlist
    unknown = 99
    assert unknown not in yeti.KNOWN_POSES
    assert unknown not in yeti.SURFACE_POSES  # allowlist: silently suppressed
    assert unknown not in yeti.NON_TRAVERSAL_POSES  # blocklist: counted
    # fall and death must never count as reaching a waypoint
    for p in (11, 12):
        assert p in yeti.NON_TRAVERSAL_POSES


def test_box_mode_is_exactly_the_old_expression():
    """ "box" mode must be byte-identical to what shipped, proven exhaustively.

    The trainer used to compute one value for both detection and capture::

        pose in SEED_POSES and within_tol((wx, wy), x, y, tol)

    and now computes detection as ``pose_ok and reaches(..., mode=...)`` with
    ``pose_ok = pose in SEED_POSES`` when the mode is "box". This asserts the two agree
    over every pose and a dense position grid, which a smoke run CANNOT establish: the
    threaded vec env is not reproducible across processes (the same 3000-step smoke gave
    reset_reach[9] = 0.10 twice and 0.11 once), so an equivalence claim has to be made
    against the expression, not against a training log.
    """
    seed_poses = frozenset(yeti.SURFACE_POSES | {13})
    for pose in range(0, 20):
        grounded = pose in seed_poses
        for tol in (2, 6):
            for x in range(20, 36):
                for y in range(104, 132):
                    old = grounded and within_tol((GOOD, ROPE1_Y), x, y, tol)
                    new = grounded and reaches((GOOD, ROPE1_Y), x, y, tol, mode="box")
                    assert old == new, (pose, tol, x, y)


def test_sprite_mode_fires_at_each_anchor_but_on_a_NARROWER_x_window():
    """Sprite mode is TIGHTER in x than any tol >= 2, and that needs pinning.

    The x window is the sprite half-width, ~7 px, i.e. under 2 x_ram units. A ladder
    waypoint's box is tol 2 (+-8 px) and a jump waypoint's is tol 6 (+-24 px), so sprite
    mode fires on strictly FEWER FRAMES than either.

    That is acceptable only because both consumers latch once per episode (the reach EMA
    and milestone marking), and the offline sweep confirmed per-episode rates hold: the
    agent transits its anchor even when it does not linger there. Measured example --
    `Step`'s anchor is px 272 while the agent's MODAL grounded position on floor 10 is
    px 260, outside the sprite window, yet per-episode detection is unchanged at 0.85
    because it walks through px 272 on the way to the ladder.

    So the invariant is "fires AT the anchor", not "fires everywhere the box did". If a
    frame COUNT ever starts mattering, revisit this.
    """
    cases = [
        (25, 118, 260),  # Rope1  floor 7,  anchor px 108
        (51, 94, 212),  # Spring floor 9,  anchor px 212
        (66, 102, 260),  # Step   floor 10, anchor px 272, modal grounded px 260
    ]
    for wx, wy, modal_px in cases:
        assert sprite_overlaps((wx, wy), wx, wy), f"anchor {wx} does not fire at itself"
        # genuinely narrower than tol 2, let alone the tol 6 used for jump waypoints
        assert not sprite_overlaps((wx, wy), wx + 2, wy)
        assert within_tol((wx, wy), wx + 2, wy, 2)
        # record the measured modal positions that fall OUTSIDE the sprite window
        modal_ram = (modal_px - 8) // 4
        if abs(modal_ram - wx) >= 2:
            assert not sprite_overlaps((wx, wy), modal_ram, wy)
