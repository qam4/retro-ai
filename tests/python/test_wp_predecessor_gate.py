"""The waypoint start-eligibility gate, and why the own-reach rule self-locks.

Measured on L4 (`l4_anchors_v2_1200k`, 7140 episodes): `Lclimb3_top`, `Low1` and
`Low2_launch` each held 100 usable seeds and were sampled as a start 0 times, while
`Step` -- the route point immediately before `Lclimb3_top` -- was reached 0.81 of the
time from reset. Seeded directly, the `Lclimb3_top` pool reaches floor 12 in 17% of
episodes, so the pool was not the problem: the gate never opened.

The own-reach rule cannot open it, because the frontier's own reach is ~0 by
definition. Gating on the PREDECESSOR's reach opens exactly one rung.
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


def _load_manager_cls():
    """Import the trainer module by path (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("_yeti_cc", _SRC)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:  # pragma: no cover - native/env deps absent
        pytest.skip(f"trainer module not importable here: {exc}")
    return mod.CheckpointManager


ROUTE = ["Spring", "Step", "Lclimb3_top", "Low1", "Low2_launch"]


def _mgr(**kw):
    cls = _load_manager_cls()
    m = cls(
        max_states_per_checkpoint=10,
        min_states_to_advance=1,
        reset_fraction=0.0,
        frontier_fraction=0.0,
        earlier_fraction=0.0,
        reach_threshold=0.15,
        **kw,
    )
    m.route_order = list(ROUTE)
    # The measured L4 situation: reliably reach Step, essentially never past it.
    m.wp_reach_ema = {
        "Spring": 0.84,
        "Step": 0.81,
        "Lclimb3_top": 0.02,
        "Low1": 0.00,
        "Low2_launch": 0.00,
    }
    return m


def test_gate_off_admits_everything():
    m = _mgr(gate_waypoints=False)
    for wid in ROUTE:
        assert m._wp_eligible(wid)


def test_own_reach_gate_locks_out_the_frontier():
    """The bug: Step is at 0.81 but nothing past it can ever be sampled."""
    m = _mgr(gate_waypoints=True)
    assert m._wp_eligible("Spring")
    assert m._wp_eligible("Step")
    assert not m._wp_eligible("Lclimb3_top")
    assert not m._wp_eligible("Low1")
    assert not m._wp_eligible("Low2_launch")


def test_predecessor_gate_opens_exactly_one_rung():
    """`Lclimb3_top` opens because `Step` is reached; `Low1` stays shut because
    `Lclimb3_top` is not. That is the frontier advancing one step at a time, which is
    the protection the gate was added for."""
    m = _mgr(gate_waypoints=True, gate_waypoints_by_predecessor=True)
    assert m._wp_eligible("Step")
    assert m._wp_eligible("Lclimb3_top"), "the frontier must become trainable"
    assert not m._wp_eligible("Low1"), "two rungs ahead must stay gated"
    assert not m._wp_eligible("Low2_launch")


def test_frontier_advances_when_the_rung_is_learned():
    """Once `Lclimb3_top` clears the threshold, `Low1` opens and `Low2_launch` does
    not. The gate walks forward instead of unlocking the whole tail at once."""
    m = _mgr(gate_waypoints=True, gate_waypoints_by_predecessor=True)
    m.wp_reach_ema["Lclimb3_top"] = 0.40
    assert m._wp_eligible("Low1")
    assert not m._wp_eligible("Low2_launch")


def test_first_route_point_needs_no_predecessor():
    m = _mgr(gate_waypoints=True, gate_waypoints_by_predecessor=True)
    m.wp_reach_ema["Spring"] = 0.0
    assert m._wp_eligible("Spring"), "reset reaches the first point by definition"


def test_waypoint_absent_from_route_order_is_not_crashed_on():
    """Route order is a display list that may omit points; absence must not raise."""
    m = _mgr(gate_waypoints=True, gate_waypoints_by_predecessor=True)
    m.wp_reach_ema["Mystery"] = 0.0
    assert m._wp_predecessor("Mystery") is None
    assert m._wp_eligible("Mystery") is False


def test_predecessor_gate_is_off_by_default():
    """L1/L2/L3 behaviour must be untouched."""
    m = _mgr(gate_waypoints=True)
    assert m.gate_waypoints_by_predecessor is False
    assert not m._wp_eligible("Lclimb3_top")
