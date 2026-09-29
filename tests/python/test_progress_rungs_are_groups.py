"""A route STEP is a group of waypoints, so alternatives and aliases count once.

`rung_of` used to count distinct ids in a flat set, which double-counted two ways and
both were live on L4:

    reached Low2 (low route to floor 13)           -> 11   correct
    reached Lhi_down_bot (high route, same floor)  -> 11   correct
    reached BOTH                                  -> 12   WRONG
    reached Low2 and its own alias J12_13_b        -> 12   WRONG

Not hypothetical. `Lhi_down_bot` sits at px 104 and `Low2` landings at px 88-120 -- the
same floor, overlapping ground -- so an agent that crosses rope 2 and walks a few pixels
left registers both and banks two rungs for one step. And every jump landing carries two
names by design (the curriculum's `Low2`, the nav graph's `J12_13_b`), either of which a
seed may record.

The rung COUNT was already protected: `n_rungs` is passed in rather than derived from
the id set, with a comment saying the id set "would double-count". The function that
assigns a state TO a rung was not.

Grouping comes from the level map's own `reward_waypoints`, so the curriculum and
the reward cannot disagree about what one step is. Mandatory targets the reward does
not group -- the fruits, paid by the fruit term -- each become a group of one.

No emulator needed.
"""

import importlib.util
import pathlib

import pytest

_SRC = (
    pathlib.Path(__file__).resolve().parents[2]
    / "scripts"
    / "mo5"
    / "yeti"
    / "train_checkpoint_curriculum.py"
)


def _mod():
    spec = importlib.util.spec_from_file_location("_yeti_rungs", _SRC)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:  # pragma: no cover - native deps absent
        pytest.skip(f"trainer module not importable here: {exc}")
    return mod


def _mgr(mod, groups, n_rungs):
    return mod.CheckpointManager(
        max_states_per_checkpoint=10,
        min_states_to_advance=1,
        reset_fraction=0.0,
        frontier_fraction=0.0,
        earlier_fraction=0.0,
        mandatory_groups=groups,
        n_rungs=n_rungs,
    )


# L4's rung 11: either route to floor 13 satisfies it.
# `J12_13_b` is `Low2`'s graph name.
FLOOR13 = frozenset({"Low2", "J12_13_b", "Lhi_down_bot"})
EARLIER = [frozenset({"Lfruit_top"}), frozenset({"Fr1", "J2_3_b"})]


def test_either_member_of_an_or_group_counts_once():
    mod = _mod()
    m = _mgr(mod, EARLIER + [FLOOR13], 3)
    base = {"Lfruit_top", "Fr1"}
    assert m.rung_of(base) == 2
    assert m.rung_of(base | {"Low2"}) == 3
    assert m.rung_of(base | {"Lhi_down_bot"}) == 3


def test_reaching_BOTH_members_of_an_or_group_still_counts_once():
    """The low route lands on floor 13 a few pixels from where the high route
    arrives, so an agent can genuinely register both."""
    mod = _mod()
    m = _mgr(mod, EARLIER + [FLOOR13], 3)
    assert m.rung_of({"Lfruit_top", "Fr1", "Low2", "Lhi_down_bot"}) == 3


def test_a_graph_alias_does_not_add_a_rung():
    mod = _mod()
    m = _mgr(mod, EARLIER + [FLOOR13], 3)
    assert m.rung_of({"Lfruit_top", "Fr1", "Low2", "J12_13_b"}) == 3
    # the alias ALONE is enough, since a seed may have recorded either name
    assert m.rung_of({"Lfruit_top", "Fr1", "J12_13_b"}) == 3


def test_unknown_ids_are_ignored():
    mod = _mod()
    m = _mgr(mod, EARLIER + [FLOOR13], 3)
    assert m.rung_of({"Lfruit_top", "Low2_launch", "Hi3", "Fr1_launch"}) == 1


def test_empty_and_none():
    mod = _mod()
    m = _mgr(mod, EARLIER + [FLOOR13], 3)
    assert m.rung_of(None) == 0
    assert m.rung_of(set()) == 0


def test_the_real_L4_ladder_groups_the_two_routes_to_floor_13():
    """Built from the level map, not hand-written: the OR group must survive whatever
    `reward_waypoints` says, and every fruit must get its own rung."""
    mod = _mod()

    class _Cfg:
        class reward:
            params = {"level": 4}

    groups, n = mod._progress_ladder(_Cfg)
    assert n == len(groups)
    # exactly one group holds both floor-13 arrivals
    or_groups = [g for g in groups if "Low2" in g]
    assert len(or_groups) == 1
    assert "Lhi_down_bot" in or_groups[0], or_groups[0]
    assert "J12_13_b" in or_groups[0], or_groups[0]
    # the fruit is mandatory and ungrouped by the reward, so it is its own rung
    assert any("F1" in g for g in groups)
    # no name appears in two groups, or a state could count twice
    seen: set = set()
    for g in groups:
        assert not (g & seen), g & seen
        seen |= g


def test_the_L4_ladder_shrinks_because_the_or_pair_is_one_step():
    """13 mandatory targets, but `Low2` and `Lhi_down_bot` are one step, so 12 rungs.
    Pinned because every reach index and every reset_reach column shifts with it."""
    mod = _mod()

    class _Cfg:
        class reward:
            params = {"level": 4}

    groups, n = mod._progress_ladder(_Cfg)
    assert n == 12, f"expected 12 rungs, got {n}: {[sorted(g) for g in groups]}"
