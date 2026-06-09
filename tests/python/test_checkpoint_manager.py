"""Tests for the CheckpointManager seed-pool logic in
scripts/train_checkpoint_curriculum.py.

Covers the deferred play-based admission filter (survival /
reached-next) and the bonus-priority retention buffer that replaced
the old passive-noop probe.
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
        reset_fraction=1.0,
        frontier_fraction=0.0,
        earlier_fraction=0.0,
        min_survival_frames=30,
    )
    kwargs.update(overrides)
    return CheckpointManager(**kwargs)


def _pool_bonuses(mgr, level):
    return sorted(b for b, _s in mgr.checkpoints[level])


def test_reached_next_is_admitted_regardless_of_survival():
    mgr = _mgr()
    # survived only 5 frames (< 30) but reached the next checkpoint.
    mgr.save_scored(1, b"state", survived_frames=5, reached_next=True, bonus=100)
    assert len(mgr.checkpoints[1]) == 1
    assert mgr.stats["rejected_precarious"][1] == 0


def test_short_survival_no_next_is_rejected():
    mgr = _mgr()
    # grabbed fruit then died 5 frames later, never reached next CP.
    mgr.save_scored(1, b"state", survived_frames=5, reached_next=False, bonus=100)
    assert len(mgr.checkpoints[1]) == 0
    assert mgr.stats["rejected_precarious"][1] == 1


def test_long_survival_admitted_even_without_next():
    mgr = _mgr()
    mgr.save_scored(1, b"state", survived_frames=200, reached_next=False, bonus=100)
    assert len(mgr.checkpoints[1]) == 1


def test_bonus_priority_eviction_keeps_best():
    mgr = _mgr(max_states_per_checkpoint=3)
    # Fill with bonuses 10, 20, 30 (all admitted via reached_next).
    for b in (10, 20, 30):
        mgr.save_scored(2, f"s{b}".encode(), 100, True, bonus=b)
    assert _pool_bonuses(mgr, 2) == [10, 20, 30]

    # A higher-bonus newcomer (40) should evict the lowest (10).
    mgr.save_scored(2, b"s40", 100, True, bonus=40)
    assert _pool_bonuses(mgr, 2) == [20, 30, 40]

    # A lower-bonus newcomer (5) should be dropped, pool unchanged.
    mgr.save_scored(2, b"s5", 100, True, bonus=5)
    assert _pool_bonuses(mgr, 2) == [20, 30, 40]


def test_pick_start_returns_highest_nonempty_checkpoint():
    # reset_fraction=0, frontier_fraction=1.0 => always pick the
    # highest non-empty checkpoint pool.
    mgr = _mgr(reset_fraction=0.0, frontier_fraction=1.0, earlier_fraction=0.0)
    mgr.save_scored(1, b"cp1_state", 100, True, bonus=10)
    mgr.save_scored(2, b"cp2_state", 100, True, bonus=10)
    for _ in range(20):
        level, state = mgr.pick_start()
        assert level == 2
        assert state == b"cp2_state"


def test_pick_start_reset_returns_none():
    mgr = _mgr(reset_fraction=1.0, frontier_fraction=0.0, earlier_fraction=0.0)
    mgr.save_scored(2, b"cp2_state", 100, True, bonus=10)
    # reset_fraction=1.0 => always level 0 (fresh reset), no state.
    for _ in range(20):
        level, state = mgr.pick_start()
        assert level == 0
        assert state is None


def test_save_checkpoint_seed_archive_path_uses_default_bonus():
    mgr = _mgr()
    mgr.save_checkpoint(1, b"seed")  # no bonus arg -> default 0
    assert len(mgr.checkpoints[1]) == 1
    assert mgr.checkpoints[1][0][0] == 0


def test_disk_roundtrip_normalizes_old_bytes_format(tmp_path):
    mgr = _mgr()
    mgr.save_scored(1, b"new_state", 100, True, bonus=42)
    p = tmp_path / "checkpoints.pkl"
    mgr.save_to_disk(str(p))

    mgr2 = _mgr()
    mgr2.load_from_disk(str(p))
    assert mgr2.checkpoints[1][0] == (42, b"new_state")
