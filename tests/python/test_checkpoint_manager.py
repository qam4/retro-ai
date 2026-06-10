"""Tests for the CheckpointManager seed-pool logic in
scripts/train_checkpoint_curriculum.py.

Covers (approach 30):
- the deferred play-based admission filter (survival / reached-next),
- reset-origin retention (evict highest source_cp first, bonus
  tiebreak),
- success-weighted across-CP start selection with a CP0 floor.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

# Importing the training script pulls SB3/gym; skip cleanly if absent.
ccm = pytest.importorskip("train_checkpoint_curriculum")
CheckpointManager = ccm.CheckpointManager


def _mgr(**overrides):
    kwargs = dict(
        max_states_per_checkpoint=3,
        min_states_to_advance=2,
        reset_fraction=1.0,  # = cp0_floor
        frontier_fraction=0.0,
        earlier_fraction=0.0,
        min_survival_frames=30,
    )
    kwargs.update(overrides)
    return CheckpointManager(**kwargs)


def _pool_source_cps(mgr, level):
    return sorted(src for src, _b, _s in mgr.checkpoints[level])


# ---------------------------------------------------------------------------
# Admission filter
# ---------------------------------------------------------------------------


def test_reached_next_is_admitted_regardless_of_survival():
    mgr = _mgr()
    mgr.save_scored(
        1, b"state", survived_frames=5, reached_next=True, bonus=100, source_cp=0
    )
    assert len(mgr.checkpoints[1]) == 1
    assert mgr.stats["rejected_precarious"][1] == 0


def test_short_survival_no_next_is_rejected():
    mgr = _mgr()
    mgr.save_scored(
        1, b"state", survived_frames=5, reached_next=False, bonus=100, source_cp=0
    )
    assert len(mgr.checkpoints[1]) == 0
    assert mgr.stats["rejected_precarious"][1] == 1


def test_long_survival_admitted_even_without_next():
    mgr = _mgr()
    mgr.save_scored(
        1, b"state", survived_frames=200, reached_next=False, bonus=100, source_cp=0
    )
    assert len(mgr.checkpoints[1]) == 1


# ---------------------------------------------------------------------------
# Reset-origin retention (source_cp priority)
# ---------------------------------------------------------------------------


def test_eviction_prefers_reset_origin_states():
    mgr = _mgr(max_states_per_checkpoint=3)
    # Fill CP3 pool with three artificial states (source_cp=3).
    for i in range(3):
        mgr.save_scored(3, f"art{i}".encode(), 100, True, bonus=10, source_cp=3)
    assert _pool_source_cps(mgr, 3) == [3, 3, 3]

    # A reset-origin state (source_cp=0) should evict an artificial one.
    mgr.save_scored(3, b"reset0", 100, True, bonus=10, source_cp=0)
    assert _pool_source_cps(mgr, 3) == [0, 3, 3]

    # Another reset-origin state evicts another artificial one.
    mgr.save_scored(3, b"reset1", 100, True, bonus=10, source_cp=0)
    assert _pool_source_cps(mgr, 3) == [0, 0, 3]

    # An artificial newcomer (source_cp=3) is now no better than the
    # worst (also 3) -> dropped.
    mgr.save_scored(3, b"art_late", 100, True, bonus=999, source_cp=3)
    assert _pool_source_cps(mgr, 3) == [0, 0, 3]


def test_eviction_bonus_tiebreak_within_same_source_cp():
    mgr = _mgr(max_states_per_checkpoint=2)
    # Two same-origin states, bonuses 10 and 20.
    mgr.save_scored(2, b"a", 100, True, bonus=10, source_cp=2)
    mgr.save_scored(2, b"b", 100, True, bonus=20, source_cp=2)
    # Higher-bonus same-origin newcomer evicts the lowest-bonus (10).
    mgr.save_scored(2, b"c", 100, True, bonus=30, source_cp=2)
    bonuses = sorted(b for _src, b, _s in mgr.checkpoints[2])
    assert bonuses == [20, 30]


# ---------------------------------------------------------------------------
# Start selection
# ---------------------------------------------------------------------------


def test_pick_start_reset_returns_none_at_full_floor():
    mgr = _mgr(reset_fraction=1.0)  # cp0_floor = 1.0
    mgr.save_scored(2, b"cp2_state", 100, True, bonus=10, source_cp=0)
    for _ in range(20):
        level, state = mgr.pick_start()
        assert level == 0
        assert state is None


def test_pick_start_picks_only_nonempty_level():
    mgr = _mgr(reset_fraction=0.0)  # never forced reset
    mgr.save_scored(2, b"cp2_state", 100, True, bonus=10, source_cp=0)
    for _ in range(20):
        level, state = mgr.pick_start()
        assert level == 2
        assert state == b"cp2_state"


def test_pick_start_weights_toward_failing_segment():
    mgr = _mgr(reset_fraction=0.0, max_states_per_checkpoint=10)
    # Two levels available, each with a state.
    mgr.save_scored(1, b"cp1", 100, True, bonus=10, source_cp=0)
    mgr.save_scored(2, b"cp2", 100, True, bonus=10, source_cp=0)
    # CP1 is "solved" (high success), CP2 is "failing" (low success).
    mgr.segment_attempts[1] = 100
    mgr.segment_successes[1] = 95  # 95% -> weight 0.05
    mgr.segment_attempts[2] = 100
    mgr.segment_successes[2] = 5  # 5% -> weight 0.95
    counts = {1: 0, 2: 0}
    import random

    random.seed(0)
    for _ in range(2000):
        level, _ = mgr.pick_start()
        counts[level] += 1
    # The failing segment (CP2) should be sampled far more often.
    assert counts[2] > counts[1] * 3


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_save_checkpoint_seed_archive_defaults():
    mgr = _mgr()
    mgr.save_checkpoint(1, b"seed")  # defaults source_cp=0, bonus=0
    assert len(mgr.checkpoints[1]) == 1
    assert mgr.checkpoints[1][0] == (0, 0, b"seed")


def test_disk_roundtrip_preserves_3tuple(tmp_path):
    mgr = _mgr()
    mgr.save_scored(1, b"new_state", 100, True, bonus=42, source_cp=0)
    p = tmp_path / "checkpoints.pkl"
    mgr.save_to_disk(str(p))

    mgr2 = _mgr()
    mgr2.load_from_disk(str(p))
    assert mgr2.checkpoints[1][0] == (0, 42, b"new_state")


def test_disk_roundtrip_normalizes_legacy_2tuple(tmp_path):
    import pickle

    # Simulate an old checkpoints.pkl with (bonus, state) entries.
    legacy = {
        "checkpoints": [[], [(55, b"old1")], [], [], []],
        "stats": {"saves": [0] * 5, "starts": [0] * 5},
    }
    p = tmp_path / "legacy.pkl"
    with p.open("wb") as f:
        pickle.dump(legacy, f)

    mgr = _mgr()
    mgr.load_from_disk(str(p))
    # Legacy entry gets source_cp = level (1) and the original bonus.
    assert mgr.checkpoints[1][0] == (1, 55, b"old1")
