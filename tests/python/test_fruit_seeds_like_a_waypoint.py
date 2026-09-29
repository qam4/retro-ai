"""A fruit is seeded exactly like any other route target.

A fruit and a waypoint are the same kind of thing: same `Target` class, same `mandatory`
and `seedable` flags, counted by the same `rung_of`. Only the TRIGGER differs -- the
game's presence byte says a fruit was collected, sprite overlap says a waypoint was
reached. Nothing about SEEDING follows from that, and for a long time it did:

    fruits    -> pool keyed by RUNG NUMBER, `self.checkpoints[rung]`
    waypoints -> pool keyed by NAME,        `self.waypoints[id]`

What it cost, on every L4 status line: `cp=` printed N_RUNGS+1 slots and could only
ever fill the one a fruit pickup landed in. L4 has one fruit, so one slot out of
fourteen, while `Lclimb3_top` -- mandatory, rung 9 -- reported zero forever because
its states went to a name-keyed pool. The rung-keyed structure was near-vestigial
too: of 25522 episodes in v23, 1880 started from the rung pool against 15438 from
named pools.

These tests pin the unified behaviour at the pool level, so a fruit cannot drift back
into a separate structure.

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
    spec = importlib.util.spec_from_file_location("_yeti_fruitseed", _SRC)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:  # pragma: no cover - native deps absent
        pytest.skip(f"trainer module not importable here: {exc}")
    return mod


GROUPS = [
    frozenset({"F1"}),
    frozenset({"Lfruit_top"}),
    frozenset({"Low2", "J12_13_b", "Lhi_down_bot"}),
]


def _mgr(mod):
    return mod.CheckpointManager(
        max_states_per_checkpoint=10,
        min_states_to_advance=1,
        reset_fraction=0.0,
        frontier_fraction=0.0,
        earlier_fraction=0.0,
        min_survival_steps=30,
        mandatory_groups=GROUPS,
        n_rungs=len(GROUPS),
        admit_requires_survival=True,
    )


WALK = 0


def test_a_fruit_gets_its_own_named_pool():
    mod = _mod()
    m = _mgr(mod)
    m.save_waypoint("F1", b"state", 90, False, end_pose=WALK)
    assert "F1" in m.waypoints
    assert len(m.waypoints["F1"]) == 1


def test_a_fruit_and_a_waypoint_go_through_the_same_call():
    """Same method, same gate, same pool structure -- only the key differs."""
    mod = _mod()
    m = _mgr(mod)
    m.save_waypoint("F1", b"a", 90, False, end_pose=WALK)
    m.save_waypoint("Low2", b"b", 90, False, end_pose=WALK)
    assert sorted(m.waypoints) == ["F1", "Low2"]
    assert len(m.waypoints["F1"]) == len(m.waypoints["Low2"]) == 1


def test_a_fruit_capture_is_refused_by_the_same_survival_gate():
    mod = _mod()
    m = _mgr(mod)
    m.save_waypoint("F1", b"doomed", 5, False, end_pose=WALK)
    assert "F1" not in m.waypoints
    assert m.wp_rejected_precarious.get("F1") == 1


def test_no_rung_keyed_pool_is_written_any_more():
    """The rung array survives only to hold `goal_score`; nothing may be inserted."""
    mod = _mod()
    m = _mgr(mod)
    for tid in ("F1", "Lfruit_top", "Low2"):
        m.save_waypoint(tid, b"s", 90, False, end_pose=WALK)
    assert all(len(p) == 0 for p in m.checkpoints), [len(p) for p in m.checkpoints]


def test_the_status_line_no_longer_prints_the_misleading_cp_slots():
    mod = _mod()
    m = _mgr(mod)
    m.save_waypoint("F1", b"a", 90, False, end_pose=WALK)
    m.save_waypoint("Low2", b"b", 90, False, end_pose=WALK)
    s = m.summary()
    assert "cp=[" not in s, s
    assert "seeds=2 in 2 pools" in s, s
