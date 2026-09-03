"""The reach test must be ONE code path (targets.within_tol).

Two consumers decide "has the agent reached this waypoint": the curriculum
(reach EMAs, capture, seeding) and the reward (milestone marking). They used to
implement the comparison separately, and they still pass DIFFERENT tolerances --
which is how one anchor produced two answers on L4 ``Fr1`` and the reward
milestone became unmarkable. These tests pin the shared comparison and document
the remaining divergence so it cannot be lost again.
"""

from __future__ import annotations

import pytest
from retro_ai.training.targets import build_targets, targets_by_id, within_tol


def _old_inline(wx, wy, x, y, tol):
    """The pre-refactor comparison, duplicated at both call sites."""
    return abs(x - wx) <= tol and abs(y - wy) <= tol


@pytest.mark.parametrize("tol", [0, 1, 2, 3, 6])
def test_within_tol_matches_the_old_inline_logic(tol):
    """Pure refactor: identical results, so no run's behaviour changes."""
    for wx in range(0, 80, 7):
        for wy in range(0, 200, 11):
            for x in range(0, 80, 3):
                for y in range(0, 200, 6):
                    assert within_tol((wx, wy), x, y, tol) == _old_inline(
                        wx, wy, x, y, tol
                    )


def test_tol_y_defaults_to_tol_x_preserving_history():
    # one number for both axes is the historical behaviour
    assert within_tol((60, 158), 62, 160, 2) is True
    assert within_tol((60, 158), 62, 161, 2) is False


def test_per_axis_tolerance_is_available():
    """x is in 4 px units, y in pixels, so the axes need different numbers.
    Measured guidance (CurriculumConfig.waypoint_tolerance): tol_x 1, tol_y 2."""
    assert within_tol((60, 158), 61, 160, 1, 2) is True
    assert within_tol((60, 158), 63, 158, 1, 2) is False  # 3 units of x is too far
    assert within_tol((60, 158), 60, 156, 1, 2) is True  # 2 px of y is fine


def test_positional_targets_expose_reached():
    t = targets_by_id(4)["Fr1"]
    assert t.positional
    # Anchor is the MEASURED landing (64, 158), not floor 3's edge (60, 158).
    assert t.pos == (64, 158)
    assert t.reached(64, 158, 2) is True
    assert t.reached(60, 158, 2) is False


def test_non_positional_targets_refuse_a_position_test():
    """Fruits are RAM events and the princess is a flag; asking where they are is
    a category error, and silently answering would hide a real bug."""
    for level in (1, 2, 3, 4):
        for t in build_targets(level):
            if t.trigger == "position":
                continue
            with pytest.raises(ValueError):
                t.reached(0, 0, 2)


def test_l4_corrected_anchors_are_markable_where_the_agent_actually_stands():
    """The bug this module was written for, now pinned in its FIXED state.

    Before: ``Fr1`` was anchored at x_ram 60, floor 3's left extremity, while the
    agent occupies only 64..68 there (10 from-reset episodes, 120 grounded steps) and
    walking left to 61 is fatal 6/6. The reward's tol-2 box could not fire, so the
    milestone was never marked and its distance term never switched off -- worth
    +0.48/step of pull against +0.04 from the base route potential.

    ``Rope1`` had the same defect: anchored at floor 7's edge (24, 118) while the
    landing is (27, 118), visually confirmed.

    If this test fails, an anchor moved again. That is a REWARD change: L3 champions
    stop being valid warm-starts. See experiments/003-yeti/level4_notes.md.
    """
    t = targets_by_id(4)
    # every position the agent was measured standing at on floor 3 marks now
    assert all(t["Fr1"].reached(x, 158, 2) for x in (64, 65, 66))
    # and the old edge anchor's lethal neighbourhood no longer marks
    assert not t["Fr1"].reached(60, 158, 2)
    # Rope1 sits at x_ram 25 (px 108), the MODAL landing across two policies, chosen by
    # multi-policy census. A geometric derivation briefly put it at 28 (px 120): that
    # scored 1.00 against v6's champion and 0.00 against a later policy landing 4 px
    # further left, because the tol-2 reward box 112..128 sat on the jump arc. Centring
    # on the mode is what buys margin against the next policy's drift.
    assert t["Rope1"].pos == (25, 118)
    assert t["Rope1"].reached(25, 118, 2)  # px 108, the modal landing
    assert t["Rope1"].reached(27, 118, 2)  # px 116, the other policy's landing


def test_l4_redundant_launch_pads_are_gone():
    """Three launch pads shared a platform with a waypoint that already marked
    correctly, so they were a second, wider, misplaced box for the same traversal.
    `Low1_launch` in particular sat on floor 11 with `Lclimb3_top` and its tol-6 box
    reported a stalled climb 24 px away as an arrival."""
    ids = set(targets_by_id(4))
    for gone in ("Spring_launch", "Step_launch", "Low1_launch"):
        assert gone not in ids
    # the exact waypoints that replaced them are still there
    for kept in ("Lclimb2_top", "Spring", "Lclimb3_top"):
        assert kept in ids


def test_anchor_overrides_do_not_break_the_graph_alias():
    """A jump landing has two names for one point (curriculum `Fr1`, graph `J2_3_b`).
    That link used to be resolved by comparing COORDINATES, so the first anchor
    override silently severed it and dropped L4's mandatory set from 14 to 12 --
    shortening the progress ladder without any error. It is now derived from the
    jump-edge structure, so it survives an anchor move."""
    for level, expect in ((3, 14), (4, 14)):
        t = targets_by_id(level)
        assert sum(1 for x in t.values() if x.mandatory) == expect
    t = targets_by_id(4)
    assert t["Fr1"].node_ident == "J2_3_b" and t["Fr1"].mandatory
    assert t["Rope1"].node_ident == "J6_7_b" and t["Rope1"].mandatory
