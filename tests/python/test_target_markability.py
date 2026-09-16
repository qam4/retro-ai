"""Every positional target must be markable from somewhere the agent can stand.

WHY IT MATTERS
--------------
A target that cannot be marked is not a mild inefficiency. An unmarked group stays in
the potential's sum and keeps pulling toward itself forever, and on L4 floor 12 the
unmarked backward terms outweighed the forward ones and REVERSED the shaping gradient
(sum 984 at px 184 against 888 at px 232, i.e. away from the goal).

WHAT THIS CATCHES, AND WHAT IT DOES NOT
---------------------------------------
It catches STRUCTURAL unmarkability: an anchor with no standable position on its own
floor from which the reach test can fire, at either geometry. That is a real class --
it is how L3 `Lesc_top` was found to sit 28 px past floor 4's standable end (it is a
carried target, see CARRIED below).

It does NOT catch the failure that motivated the original script, and claiming otherwise
would be worse than not having the test. L4 `Fr1` was anchored at x_ram 60 with reward
tol 2, and never marked in eight runs because the agent only ever occupies 64..68 on
that platform. But floor 3's standable run at that height is 60..68, so x_ram 60 IS
standable and every check here passes it. Verified by injecting that anchor: sprite
markable True, box tol-2 markable True.

Separating "reachable in principle" from "somewhere the policy actually goes" needs
measured occupancy from a rollout, which needs the emulator and a policy, and a miss
there is evidence about THAT policy rather than proof of impossibility. That belongs in
scripts/mo5/yeti/diag/, not in CI. A pass here is necessary, not sufficient.

The two reach geometries are both covered because `waypoint_reach_mode` selects between
them at runtime, so a target must be markable under whichever one a run picks.
"""

from __future__ import annotations

import pytest
from retro_ai.training.targets import build_targets, sprite_overlaps, within_tol
from retro_ai.training.yeti_map import (
    agent_floor_from_pixel_xy,
    get_level_map,
    jump_waypoints,
)

LEVELS = (1, 2, 3, 4)
# The tolerances the two consumers actually pass today, in box mode.
REWARD_TOL = 2
CURRICULUM_LADDER_TOL = 2
CURRICULUM_JUMP_TOL = 6

# TARGETS THE AGENT IS CARRIED TO, not stood on.
#
# The invariant below assumes a target is reached from a position that resolves to the
# target's own floor. That is false where the level moves the agent: the L3 escalator
# carries it (pose 13, "escalator ride, a controlled vertical traversal") and L4's ropes
# and trampoline do the same (poses 14/15 rope carry, 16/17 trampoline rise). For those,
# the anchor marks a point on the CONVEYANCE, so no standable floor position exists and
# the reach test is expected to fire on a carried frame instead.
#
# Each entry must say which mechanism carries the agent. This is a list of known
# exceptions, NOT a way to silence a failure: a new unexplained entry here is a bug
# being hidden. Verified when added -- L3 Lesc_top sits at x_ram 33 while floor 4's
# standable run is 16..26, i.e. 7 units (28 px) past the platform's end, and
# scripts/mo5/yeti/diag/l3_lesc_boarding.py probes exactly "within tol of (33,94)".
CARRIED = {
    (3, "Lesc_top"): "escalator boarding point; agent rides (pose 13), never stands",
}
# x_ram sweep bound. The playfield is 320 px and px = x_ram * 4 + 8, so x_ram tops out
# near 78; -4..84 covers it with margin on both sides.
X_LO, X_HI = -4, 84


def positional_targets(level):
    """Targets whose reach test is geometric AND which the agent stands to reach."""
    return [
        t
        for t in build_targets(level)
        if t.trigger == "position"
        and t.pos is not None
        and t.floor is not None
        and (level, t.id) not in CARRIED
    ]


def standable(x_ram, y_px, floor, level):
    """Does (x_ram, y_px) resolve to ``floor``? The same helper the reward uses.

    Done in PIXELS deliberately: platform extents are tile-pixel boundaries while
    x_ram is the agent's reference point (px = x_ram * 4 + 8). Mixing the two units
    is wrong by 1-2 units on every platform.
    """
    return agent_floor_from_pixel_xy(x_ram * 4 + 8, y_px, level) == floor


def standable_run(t, level):
    """Every x_ram on the target's own floor at the anchor's height."""
    return [x for x in range(X_LO, X_HI) if standable(x, t.pos[1], t.floor, level)]


def _ids(level):
    return [t.id for t in positional_targets(level)]


@pytest.mark.parametrize("level", LEVELS)
def test_every_target_sits_on_a_floor_with_somewhere_to_stand(level):
    """Before asking about reach: the target's floor must be occupiable at all."""
    dead = [t.id for t in positional_targets(level) if not standable_run(t, level)]
    assert not dead, (
        f"L{level}: these targets' floors have NO standable x at the anchor's height, "
        f"so no reach test can ever fire there: {dead}"
    )


@pytest.mark.parametrize("level", LEVELS)
def test_every_target_is_markable_under_sprite_mode(level):
    """Sprite mode: some standable position's sprite must contain the anchor point."""
    unmarkable = []
    for t in positional_targets(level):
        run = standable_run(t, level)
        if not any(sprite_overlaps(t.pos, x, t.pos[1]) for x in run):
            unmarkable.append((t.id, t.pos, (min(run), max(run)) if run else None))
    assert not unmarkable, (
        f"L{level}: unmarkable under sprite overlap (id, anchor, standable x_ram run): "
        f"{unmarkable}"
    )


@pytest.mark.parametrize("level", LEVELS)
def test_every_target_is_markable_under_box_mode(level):
    """Box mode: some standable position must fall inside the tolerance box.

    Checked at BOTH tolerances in play, because the reward passes 2 while the
    curriculum passes 6 for jump waypoints -- a divergence that let a target be
    capturable as a seed but never markable as a reward milestone.
    """
    lvl = get_level_map(level)
    jump_ids = set(jump_waypoints(lvl))
    bad = []
    for t in positional_targets(level):
        run = standable_run(t, level)
        cur_tol = CURRICULUM_JUMP_TOL if t.id in jump_ids else CURRICULUM_LADDER_TOL
        for who, tol in (("reward", REWARD_TOL), ("curriculum", cur_tol)):
            if not any(within_tol(t.pos, x, t.pos[1], tol, tol) for x in run):
                bad.append((t.id, who, tol, t.pos))
    assert not bad, (
        f"L{level}: no standable position falls in the box (id, consumer, tol, anchor):"
        f" {bad}"
    )


def test_l4_mandatory_targets_are_all_covered():
    """Guard the guard: if L4's target list is re-keyed, these tests must not go quiet.

    An allowlist that silently matches nothing is how the pose-gate bug survived --
    `SURFACE_POSES` was missing two poses for the project's whole history and every
    pose-gated decision just quietly did less.
    """
    mandatory = [t for t in build_targets(4) if t.mandatory]
    assert len(mandatory) == 14, (
        f"L4 mandatory target count changed to {len(mandatory)}; it was 14. Update the "
        "reward_waypoints groups and the rung-index mapping in level4_notes.md "
        "together with this number."
    )
    # Only POSITIONAL mandatory targets are in scope. L4's `F1` is trigger="event" and
    # `princess` is trigger="flag": they carry a pos for map drawing but are not reached
    # by geometry, so demanding a standable box for them tests nothing real.
    positional = set(_ids(4))
    missed = [
        t.id
        for t in mandatory
        if t.trigger == "position"
        and (4, t.id) not in CARRIED
        and t.id not in positional
    ]
    assert not missed, f"positional mandatory targets not covered: {missed}"

    by_trigger = {}
    for t in mandatory:
        by_trigger[t.trigger] = by_trigger.get(t.trigger, 0) + 1
    assert by_trigger.get("position", 0) >= 12, (
        f"only {by_trigger.get('position', 0)} of L4's mandatory targets are "
        f"positional ({by_trigger}); if most became non-positional these geometry "
        "tests stopped covering the route and need rewriting, not relaxing"
    )
