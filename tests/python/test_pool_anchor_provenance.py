"""A waypoint pool must not be inherited across an ANCHOR MOVE.

A pool's states were selected by proximity to the waypoint's anchor, so moving that
anchor invalidates them: they are filed under a position they were not captured at.

The resume path already dropped pools for waypoints a level no longer DEFINES, but that
check is on the NAME, and an anchor move keeps the name. So a corrected anchor would
take effect for future captures while the old, bad states were carried forward -- which
is precisely the case the correction was meant to fix.

Measured on L4: `Low2_launch` was anchored at px 184, floor 12's tile edge. The agent
reads as grounded there for one frame (y still 70, walk pose) and then falls, so all 100
of its seeds were unrecoverable and the pool meant to teach the rope-2 crossing taught
falling instead. Correcting the anchor to px 188 fixes new captures; without the check
under test here, a warm start would have re-imported the 100 dead states.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "mo5" / "yeti"))

ccm = pytest.importorskip("train_checkpoint_curriculum")
CheckpointManager = ccm.CheckpointManager


def _mgr(anchors):
    m = CheckpointManager(
        max_states_per_checkpoint=8,
        min_states_to_advance=2,
        reset_fraction=1.0,
        frontier_fraction=0.0,
        earlier_fraction=0.0,
        min_survival_steps=30,
    )
    m.waypoint_anchors = dict(anchors)
    return m


def _write(tmp_path, pools, anchors):
    """A pool file as save_to_disk writes one."""
    path = tmp_path / "checkpoints.pkl"
    data = {
        "checkpoints": [[] for _ in range(5)],
        "stats": {},
        "waypoints": {w: (states, 0.5) for w, states in pools.items()},
    }
    if anchors is not None:
        data["waypoint_anchors"] = anchors
    with open(path, "wb") as f:
        pickle.dump(data, f)
    return str(path)


def _seed(tag=b"state"):
    # (source_cp, bonus, state_bytes, stack, reached)
    return (0, 100, tag, None, frozenset())


def test_pool_dropped_when_anchor_moved(tmp_path):
    """The L4 Low2_launch case: same name, anchor moved 184 -> 188 px."""
    path = _write(
        tmp_path,
        {"Low2_launch": [_seed()], "Low2": [_seed()]},
        {"Low2_launch": (44, 70), "Low2": (30, 70)},
    )
    # x_ram 44 -> 45 is px 184 -> 188; Low2 unchanged.
    m = _mgr({"Low2_launch": (45, 70), "Low2": (30, 70)})
    m.load_from_disk(path)
    assert "Low2_launch" not in m.waypoints, "pool survived an anchor move"
    assert "Low2" in m.waypoints, "an UNMOVED anchor's pool must still be inherited"


def test_pool_kept_when_anchor_identical(tmp_path):
    path = _write(tmp_path, {"Rope1": [_seed()]}, {"Rope1": (27, 118)})
    m = _mgr({"Rope1": (27, 118)})
    m.load_from_disk(path)
    assert "Rope1" in m.waypoints
    assert len(m.waypoints["Rope1"]) == 1


def test_a_one_unit_move_is_enough_to_drop(tmp_path):
    """x is in 4-PIXEL units, so a single unit is a 4 px move -- and 4 px is exactly
    the difference between floor 12's tile edge and its first standable centre."""
    path = _write(tmp_path, {"W": [_seed()]}, {"W": (44, 70)})
    m = _mgr({"W": (45, 70)})
    m.load_from_disk(path)
    assert "W" not in m.waypoints


def test_y_move_also_drops(tmp_path):
    path = _write(tmp_path, {"W": [_seed()]}, {"W": (44, 70)})
    m = _mgr({"W": (44, 78)})
    m.load_from_disk(path)
    assert "W" not in m.waypoints


def test_pre_provenance_file_is_still_loaded(tmp_path):
    """Files written before anchors were recorded carry no provenance. They keep the
    old behaviour (inherit) rather than silently discarding a run's pools; the loader
    reports that it could not verify them."""
    path = _write(tmp_path, {"W": [_seed()]}, None)
    m = _mgr({"W": (44, 70)})
    m.load_from_disk(path)
    assert "W" in m.waypoints


def test_unknown_waypoint_in_file_is_untouched_by_this_check(tmp_path):
    """A pool for a waypoint the level no longer defines has no current anchor, so this
    check must not decide its fate -- the separate name-based stale drop in main() owns
    that, and it must keep working."""
    path = _write(tmp_path, {"Gone": [_seed()]}, {"Gone": (10, 20)})
    m = _mgr({"W": (44, 70)})  # 'Gone' absent from the current level
    m.load_from_disk(path)
    assert "Gone" in m.waypoints


def test_anchors_round_trip_through_save(tmp_path):
    """save_to_disk must persist the provenance, or the next run cannot check it."""
    m = _mgr({"W": (45, 70), "V": (27, 118)})
    m.save_waypoint("W", b"s", survived_steps=999, reached_next=True)
    out = tmp_path / "out.pkl"
    m.save_to_disk(str(out))
    with open(out, "rb") as f:
        data = pickle.load(f)
    assert data["waypoint_anchors"] == {"W": (45, 70), "V": (27, 118)}


def test_saved_anchors_let_a_later_load_drop_the_pool(tmp_path):
    """End to end: capture under one anchor, reload under another, pool is gone."""
    m1 = _mgr({"W": (44, 70)})
    m1.save_waypoint("W", b"s", survived_steps=999, reached_next=True)
    assert len(m1.waypoints["W"]) == 1
    out = str(tmp_path / "out.pkl")
    m1.save_to_disk(out)

    m2 = _mgr({"W": (45, 70)})
    m2.load_from_disk(out)
    assert "W" not in m2.waypoints
