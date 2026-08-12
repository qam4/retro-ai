"""The progress ladder: pools keyed by HOW MANY mandatory targets are done.

Reframing (replaces "pools keyed by fruits collected"): a progress pool holds
states where N mandatory TARGETS have been reached. Mandatory targets are the
fruits, the princess, and — on levels that define them — the waypoint milestones.

Why it matters: L3 has ONE fruit, so the old fruit-count ladder had a single
rung (0 -> 1). That is why `cp=[0, 100]` and `success=[0->1]` carried no
information all through v5-v13, and why "reached the next checkpoint" (used by
seed admission) could effectively never fire. Counting mandatory targets instead
gives L3 twelve intermediate rungs — the same curriculum granularity that
plausibly solved L2 when waypoints were introduced there.

The critical property, asserted here rather than assumed: on L1/L2 the ladder is
UNCHANGED, because those levels define no waypoint milestones, so the count of
mandatory targets reached is exactly the number of fruits collected. That is what
makes this reframing safe for two solved champions (L1 99.7%, L2 98.7%).
"""

from __future__ import annotations

import pytest
from retro_ai.games import yeti
from retro_ai.training.targets import build_targets
from retro_ai.training.yeti_map import get_level_map


def mandatory_ids(level: int) -> set:
    """Ids of every mandatory target, including graph aliases for jump landings
    (the curriculum calls one "A1", the graph calls it "J10_11_b")."""
    out = set()
    for t in build_targets(level):
        if not t.mandatory:
            continue
        out.add(t.id)
        if t.node_ident:
            out.add(t.node_ident)
    return out


def rung_of(level: int, reached: set) -> int:
    """The progress rung a state belongs to: how many mandatory targets it has
    behind it. This is the single keying rule for every level."""
    return len({r for r in reached if r in mandatory_ids(level)})


@pytest.mark.parametrize("level", [1, 2])
def test_ladder_unchanged_on_solved_levels(level):
    """L1/L2 have no waypoint milestones, so only fruits (and the princess) can
    advance a rung -> identical to the old fruits-collected keying."""
    lvl = get_level_map(level)
    assert not lvl.reward_waypoints, "precondition: no milestones on this level"
    mand = mandatory_ids(level)
    # every waypoint on these levels is NON-mandatory, so touching them cannot
    # move a state to a different rung (the property that keeps pools intact)
    for wid in yeti.waypoints(level):
        assert wid not in mand
    # a state that has touched every waypoint but no fruit is still rung 0
    assert rung_of(level, set(yeti.waypoints(level))) == 0
    # collecting fruits advances the rung one at a time
    fruit_ids = [f"F{i}" for i in sorted(lvl.fruit_centre_px)]
    for k in range(len(fruit_ids) + 1):
        assert rung_of(level, set(fruit_ids[:k])) == k


def test_l3_ladder_gains_intermediate_rungs():
    """The point of the reframing: L3 goes from one rung to many."""
    n_fruit_rungs = len(get_level_map(3).fruit_centre_px)
    assert n_fruit_rungs == 1, "L3 has a single fruit — the old ladder"
    mand = {t for t in build_targets(3) if t.mandatory and t.kind != "princess"}
    assert len(mand) >= 10, "expected the milestone-based ladder to be much finer"


def test_l3_rung_increases_along_the_route():
    """Walking the route must move a state monotonically UP the ladder, so
    "reached the next rung" is a meaningful admission signal on L3."""
    route = [
        "Lgoat_a_top",
        "Ldown_bot",
        "Lsc1_top",
        "Lsc2_top",
        "Lsc3_top",
        "Lsc4_top",
        "A1",
        "A2",
        "A3",
        "A5",
        "Lprincess_top",
    ]
    seen: set = set()
    rungs = []
    for wid in route:
        seen.add(wid)
        rungs.append(rung_of(3, set(seen)))
    assert rungs == sorted(rungs), f"rungs must not go down along the route: {rungs}"
    assert rungs[-1] > rungs[0], "the route must climb the ladder"
    # every step on this route is a milestone, so each one advances the rung
    assert rungs == list(range(1, len(route) + 1)), rungs


def test_non_milestone_waypoints_do_not_advance_the_ladder():
    """Launch pads and the escalator board point are seedable but NOT mandatory,
    so they must not inflate the rung (they are places, not progress)."""
    base = rung_of(3, {"Lsc4_top"})
    for extra in ("A1_launch", "A2_launch", "Lesc_top", "A4_launch"):
        assert rung_of(3, {"Lsc4_top", extra}) == base, extra


def test_existing_seed_reached_sets_map_onto_rungs():
    """The reached-set we already store on every seed is exactly what the ladder
    needs — no recapture or migration of state bytes required."""
    # A4 sits on the fruit platform and is deliberately not a milestone, so a
    # seed there is keyed by what it has actually banked.
    assert rung_of(3, {"Lgoat_a_top"}) == 1
    assert rung_of(3, {"Lgoat_a_top", "Ldown_bot", "Lsc1_top"}) == 3
    # graph-alias form must count the same as the curriculum name
    assert rung_of(3, {"J10_11_b"}) == rung_of(3, {"A1"}) == 1
