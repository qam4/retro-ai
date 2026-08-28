"""The sprite-pose catalogue must be complete, and gaps in it must be explicit.

An uncatalogued pose is a silent behaviour change. Every pose-gated decision --
waypoint reach detection, seed capture, reward milestone marking, floor crediting --
reads
an unrecognised code as "not on a surface", so whatever the agent did in that frame does
not count anywhere.

That is not hypothetical. The walk animation is a FOUR-pose cycle per direction, but
only
the rightward cycle (0-3) was ever listed; poses 6 and 7 are grounded leftward-walk
frames and were absent from ``SURFACE_POSES``. Measured on L4 floor 12: 54% of grounded
frames while walking left were discarded by the surface gate, versus 0% walking right.
L4's closing stretch is leftward (rope 2 crosses floor 12 -> 13 leftward, and floor 13
to the princess ladder is leftward), so that gap fell exactly on the level's wall.
"""

from __future__ import annotations

from retro_ai.games import yeti


def test_every_surface_pose_is_catalogued():
    assert not (yeti.SURFACE_POSES - yeti.KNOWN_POSES)


def test_walk_cycles_are_four_poses_each():
    """Measured: 0-3 step +4 px, 4-7 step -4 px. Both cycles are four frames long.
    If this fails the catalogue has been edited without re-measuring."""
    right = {p for p, n in yeti.POSE_NAMES.items() if n.startswith("walk-right")}
    left = {p for p, n in yeti.POSE_NAMES.items() if n.startswith("walk-left")}
    assert right == {0, 1, 2, 3}
    assert left == {4, 5, 6, 7}


def test_the_known_gap_is_named_not_just_commented():
    """The two grounded left-walk poses missing from the surface gate are exported, so
    callers and status lines can point at the gap instead of re-deriving it."""
    assert yeti.SURFACE_POSES_MISSING_LEFT == {6, 7}
    # they are catalogued as grounded...
    for p in yeti.SURFACE_POSES_MISSING_LEFT:
        assert p in yeti.KNOWN_POSES
        assert "grounded" in yeti.POSE_NAMES[p]
    # ...but deliberately still excluded from the surface gate. Admitting them changes
    # which frames can mark a reward milestone, so it invalidates existing champions as
    # warm-starts and has to be its own training lever.
    assert not (yeti.SURFACE_POSES_MISSING_LEFT & yeti.SURFACE_POSES)


def test_unknown_poses_detects_an_uncatalogued_code():
    assert yeti.unknown_poses([0, 8, 11]) == set()
    assert yeti.unknown_poses([0, 99]) == {99}
    # 16/17 are catalogued (trampoline). 15 is NOT: it was observed within minutes of
    # this check going in (4 occurrences in a 3000-step smoke run) and has not been
    # identified yet, so it must still report as unknown. When someone identifies it and
    # adds it to POSE_NAMES, this line is the one to update.
    assert yeti.unknown_poses([15, 16, 17]) == {15}


def test_unknown_poses_accepts_a_counter_or_any_iterable():
    from collections import Counter

    assert yeti.unknown_poses(Counter({0: 5, 42: 1})) == {42}
    assert yeti.unknown_poses({0, 42}) == {42}
    assert yeti.unknown_poses([]) == set()


def test_airborne_poses_are_not_surface():
    """Regression guard: a fall or a jump must never satisfy the surface gate, or the
    curriculum seeds mid-air states that inherit the fall on reload."""
    for p in (9, 10, 11, 12):
        assert p not in yeti.SURFACE_POSES
        assert p in yeti.KNOWN_POSES


def test_l4_rope_and_trampoline_poses_are_distinguished():
    """14 is the rope carry; 16/17 are the trampoline below the rope-2 gap. Conflating
    them produced a wrong diagnosis once -- pose 17 at the launch pad was read as
    "hanging on the rope" when the agent had already fallen and was bouncing back up."""
    assert "rope" in yeti.POSE_NAMES[14]
    assert "trampoline" in yeti.POSE_NAMES[16]
    assert "trampoline" in yeti.POSE_NAMES[17]
    for p in (14, 16, 17):
        assert p not in yeti.SURFACE_POSES
