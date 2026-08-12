"""The unified Target model must reproduce today's fruits / waypoints / reward
milestones exactly — it is a shared vocabulary, not a behaviour change.

Guards the two alias/naming traps that already caused real bugs:
  * a jump landing has TWO names for one point (curriculum "A1" vs graph
    "J10_11_b"); mandatory-ness must resolve across both, or the milestone
    restore silently does nothing;
  * positional targets do NOT survive a save-state (unlike fruits, whose state
    is emulator RAM), which is exactly the distinction that made seeded episodes
    get paid to retreat.
"""

from __future__ import annotations

import pytest
from retro_ai.games import yeti
from retro_ai.training.targets import build_targets, targets_by_id
from retro_ai.training.yeti_map import get_level_map


@pytest.mark.parametrize("level", [1, 2, 3])
def test_targets_cover_fruits_waypoints_princess(level):
    lvl = get_level_map(level)
    t = targets_by_id(level)
    # every fruit
    for fid in lvl.fruit_centre_px:
        assert f"F{fid}" in t and t[f"F{fid}"].kind == "fruit"
    # every waypoint the seeder knows about
    for wid in yeti.waypoints(level):
        assert wid in t and t[wid].kind == "waypoint"
    # the princess
    assert t["princess"].kind == "princess" and t["princess"].trigger == "flag"
    # no duplicates
    ids = [x.id for x in build_targets(level)]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("level", [1, 2, 3])
def test_trigger_determines_save_state_survival(level):
    """Only POSITIONAL targets need restoring on a seed load; fruit/flag state
    lives in emulator RAM and comes back for free."""
    for t in build_targets(level):
        if t.kind == "waypoint":
            assert t.trigger == "position" and t.positional
        else:
            assert t.trigger in ("event", "flag") and not t.positional


@pytest.mark.parametrize("level", [1, 2, 3])
def test_mandatory_matches_reward_waypoints_plus_fruits_and_princess(level):
    lvl = get_level_map(level)
    reward_idents = {i for g in (lvl.reward_waypoints or []) for i in g}
    t = build_targets(level)
    mandatory = {x.id for x in t if x.mandatory}
    # fruits + princess are always mandatory
    for fid in lvl.fruit_centre_px:
        assert f"F{fid}" in mandatory
    assert "princess" in mandatory
    # waypoints are mandatory iff named in reward_waypoints (directly or via a
    # same-position graph alias)
    wp_mandatory = {x.id for x in t if x.kind == "waypoint" and x.mandatory}
    if not reward_idents:  # L1/L2 have none
        assert wp_mandatory == set()
    else:
        # direct names must all be present
        named = {i for i in reward_idents if i in {x.id for x in t}}
        assert named <= wp_mandatory


def test_l3_jump_landing_alias_resolves_to_graph_node():
    """A1..A5 are named by the curriculum but are graph nodes J*_b; mandatory-ness
    and shaping must follow the alias (the trap that disabled the restore)."""
    t = targets_by_id(3)
    for name, ident in (
        ("A1", "J10_11_b"),
        ("A2", "J11_12_b"),
        ("A3", "J12_13_b"),
        ("A5", "J14_15_b"),
    ):
        assert t[name].node_ident == ident, f"{name} should map to {ident}"
        assert t[name].mandatory, f"{name} is a reward milestone on L3"
    # A4 is deliberately NOT a milestone (the FRUIT sits on that platform)
    assert not t["A4"].mandatory
    # launch pads are seedable but never reward targets
    assert t["A1_launch"].seedable and not t["A1_launch"].mandatory


def test_princess_is_not_seedable():
    """Terminal: there is nothing to practise FROM the princess touch."""
    for level in (1, 2, 3):
        assert not targets_by_id(level)["princess"].seedable


def test_fruits_and_waypoints_are_seedable():
    t = build_targets(3)
    assert all(x.seedable for x in t if x.kind in ("fruit", "waypoint"))
