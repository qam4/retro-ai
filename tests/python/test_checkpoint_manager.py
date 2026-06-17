"""Tests for the CheckpointManager seed-pool logic in
scripts/train_checkpoint_curriculum.py.

Covers (approach 30 + 31):
- the deferred play-based admission filter (survival / reached-next),
- reset-origin retention (evict highest source_cp first, bonus
  tiebreak),
- reach-gated, success-weighted across-CP start selection with a CP0
  floor (approach 31: a level is eligible only once it is reachable
  from reset, weighted by the responsive per-segment success EMA).
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

    # A *more artificial* newcomer (source_cp=4 > worst tier 3) is
    # dropped — we never pollute with something less reset-origin.
    mgr.save_scored(3, b"art_worse", 100, True, bonus=999, source_cp=4)
    assert _pool_source_cps(mgr, 3) == [0, 0, 3]


def test_full_pool_refreshes_instead_of_freezing():
    # Regression test for the v6 freeze: with bonus-tiebreak eviction the
    # pool locked onto the few highest-bonus states and stopped accepting
    # newcomers. Diversity-preserving eviction must keep admitting recent
    # same-tier states (the most-recently inserted one is always present).
    import random

    random.seed(0)
    mgr = _mgr(max_states_per_checkpoint=3)
    for i in range(20):
        # All reset-origin (source_cp=0); deliberately *decreasing* bonus
        # so the old bonus-rule would have rejected every one after the
        # first three.
        mgr.save_scored(1, f"s{i}".encode(), 100, True, bonus=100 - i, source_cp=0)
    pool_states = {s for _src, _b, s in mgr.checkpoints[1]}
    assert len(mgr.checkpoints[1]) == 3
    # The last inserted state must be in the pool (proves no freeze).
    assert b"s19" in pool_states
    # And the pool is not stuck on the earliest few.
    assert pool_states != {b"s0", b"s1", b"s2"}


# ---------------------------------------------------------------------------
# Reach gate / EMA bookkeeping (approach 31)
# ---------------------------------------------------------------------------


def test_reset_reach_ema_rises_only_for_reached_levels():
    mgr = _mgr()
    # Many reset episodes that reach CP2 (collected 2 fruits).
    for _ in range(500):
        mgr.record_episode(start_level=0, reached_level=2)
    # CP1 and CP2 should be considered reached; CP3/CP4 should not.
    assert mgr.reset_reach_ema[1] > 0.9
    assert mgr.reset_reach_ema[2] > 0.9
    assert mgr.reset_reach_ema[3] < 0.1
    assert mgr.reset_reach_ema[4] < 0.1


def test_reset_reach_tracks_princess_from_reset():
    # reset_reach_ema[5] is the princess-from-reset rate (the win
    # condition). Reset episodes that reach the princess (reached_level=5,
    # via the H-M fix) must drive it up; reaching only CP4 must not.
    mgr = _mgr()
    for _ in range(500):
        mgr.record_episode(start_level=0, reached_level=5)
    assert mgr.reset_reach_ema[5] > 0.9
    mgr2 = _mgr()
    for _ in range(500):
        mgr2.record_episode(start_level=0, reached_level=4)
    assert mgr2.reset_reach_ema[5] < 0.1
    assert mgr2.reset_reach_ema[4] > 0.9


def test_non_reset_episodes_do_not_move_reach_ema():
    mgr = _mgr()
    # Episodes starting from CP2 are not evidence of reset-reachability.
    for _ in range(500):
        mgr.record_episode(start_level=2, reached_level=3)
    assert mgr.reset_reach_ema[3] == 0.0
    # But they do update the CP2 success EMA.
    assert mgr.seg_success_ema[2] > 0.9


def test_cp4_princess_touch_counts_as_success():
    # H-M regression: a CP4 start that reaches the princess is logged
    # with reached_level=5, which must register as a CP4 segment success
    # (reached_level > start_level). Previously the env passed
    # reached_level=4-fruits which caps at 4, so CP4 success was never
    # recorded and its curriculum weight stayed pinned at the max.
    mgr = _mgr()
    for _ in range(500):
        mgr.record_episode(start_level=4, reached_level=5)
    assert mgr.seg_success_ema[4] > 0.9
    assert mgr.segment_successes[4] == 500


def test_seg_success_ema_tracks_recent_outcomes():
    mgr = _mgr()
    for _ in range(500):
        mgr.record_episode(start_level=1, reached_level=2)  # success
    assert mgr.seg_success_ema[1] > 0.9
    for _ in range(500):
        mgr.record_episode(start_level=1, reached_level=1)  # failure
    assert mgr.seg_success_ema[1] < 0.1


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
    # Reach gate: CP2 is only eligible once it's reset-reachable.
    mgr.reset_reach_ema[2] = 1.0
    for _ in range(20):
        level, state = mgr.pick_start()
        assert level == 2
        assert state == b"cp2_state"


def test_pick_start_gated_out_returns_reset():
    # A populated pool whose reach EMA is below threshold must NOT be
    # selected — the agent can't get there from reset yet.
    mgr = _mgr(reset_fraction=0.0, reach_threshold=0.15)
    mgr.save_scored(3, b"cp3_state", 100, True, bonus=10, source_cp=0)
    mgr.reset_reach_ema[3] = 0.05  # below the gate
    for _ in range(20):
        level, state = mgr.pick_start()
        assert level == 0
        assert state is None


def test_pick_start_weights_toward_failing_segment():
    mgr = _mgr(reset_fraction=0.0, max_states_per_checkpoint=10)
    # Two levels available, each with a state.
    mgr.save_scored(1, b"cp1", 100, True, bonus=10, source_cp=0)
    mgr.save_scored(2, b"cp2", 100, True, bonus=10, source_cp=0)
    # Both reset-reachable (eligible).
    mgr.reset_reach_ema[1] = 1.0
    mgr.reset_reach_ema[2] = 1.0
    # CP1 is "solved" (high success EMA), CP2 is "failing" (low).
    mgr.seg_success_ema[1] = 0.95  # -> weight 0.05
    mgr.seg_success_ema[2] = 0.05  # -> weight 0.95
    counts = {1: 0, 2: 0}
    import random

    random.seed(0)
    for _ in range(2000):
        level, _ = mgr.pick_start()
        counts[level] += 1
    # The failing segment (CP2) should be sampled far more often.
    assert counts[2] > counts[1] * 3


def test_pick_start_floor_prevents_starvation():
    # With an anti-starvation floor, a near-solved segment (low weight)
    # still gets a meaningful minimum share instead of being starved by a
    # much-harder sibling.
    mgr = _mgr(reset_fraction=0.0, segment_floor=0.5, max_states_per_checkpoint=10)
    mgr.save_scored(1, b"cp1", 100, True, bonus=10, source_cp=0)
    mgr.save_scored(2, b"cp2", 100, True, bonus=10, source_cp=0)
    mgr.reset_reach_ema[1] = 1.0
    mgr.reset_reach_ema[2] = 1.0
    mgr.seg_success_ema[1] = 0.95  # "solved" -> raw weight 0.05
    mgr.seg_success_ema[2] = 0.05  # "failing" -> raw weight 0.95
    import random

    random.seed(0)
    counts = {1: 0, 2: 0}
    for _ in range(4000):
        level, _ = mgr.pick_start()
        counts[level] += 1
    frac1 = counts[1] / 4000
    # Pure weighting would give CP1 ~5%; the 0.5 floor lifts it toward
    # ~0.5/2 = 0.25, so it's not starved...
    assert frac1 > 0.18
    # ...while the failing segment is still favored.
    assert counts[2] > counts[1]


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
