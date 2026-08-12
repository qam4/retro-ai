"""Tests for the CheckpointManager seed-pool logic in
scripts/train_checkpoint_curriculum.py.

Covers (approach 30 + 31):
- the deferred play-based admission filter (survival / reached-next),
- reset-origin retention (evict highest source_cp first, bonus
  tiebreak),
- reach-gated, success-weighted across-CP start selection with a CP0
  floor (approach 31: a level is eligible only once it is reachable
  from reset, weighted by the responsive per-segment success EMA),
- the H-AB frame-stack carried per pool entry (4-tuple; None when
  stack-less).

NOTE: pool entries are 4-tuples (source_cp, bonus, state_bytes, stack)
and pick_start returns 3 values (key, state, stack).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "mo5" / "yeti"))

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
        min_survival_steps=30,
    )
    kwargs.update(overrides)
    return CheckpointManager(**kwargs)


def _pool_source_cps(mgr, level):
    return sorted(e[0] for e in mgr.checkpoints[level].states)


# ---------------------------------------------------------------------------
# Admission filter
# ---------------------------------------------------------------------------


def test_reached_next_is_admitted_regardless_of_survival():
    mgr = _mgr()
    mgr.save_scored(
        1, b"state", survived_steps=5, reached_next=True, bonus=100, source_cp=0
    )
    assert len(mgr.checkpoints[1]) == 1
    assert mgr.stats["rejected_precarious"][1] == 0


def test_short_survival_no_next_is_rejected():
    mgr = _mgr()
    mgr.save_scored(
        1, b"state", survived_steps=5, reached_next=False, bonus=100, source_cp=0
    )
    assert len(mgr.checkpoints[1]) == 0
    assert mgr.stats["rejected_precarious"][1] == 1


def test_long_survival_admitted_even_without_next():
    mgr = _mgr()
    mgr.save_scored(
        1, b"state", survived_steps=200, reached_next=False, bonus=100, source_cp=0
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
    pool_states = {e[2] for e in mgr.checkpoints[1].states}
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
        level, state, stack, _ = mgr.pick_start()
        assert level == 0
        assert state is None
        assert stack is None


def test_pick_start_picks_only_nonempty_level():
    mgr = _mgr(reset_fraction=0.0)  # never forced reset
    mgr.save_scored(2, b"cp2_state", 100, True, bonus=10, source_cp=0)
    # Reach gate: CP2 is only eligible once it's reset-reachable.
    mgr.reset_reach_ema[2] = 1.0
    # H-T: reset (CP0) always competes; only EMPTY levels (1, 3, 4) are
    # never picked. So a pick is either reset (0, None) or CP2 (2, state).
    for _ in range(50):
        level, state, _, _ = mgr.pick_start()
        assert level in (0, 2)
        if level == 2:
            assert state == b"cp2_state"
        else:
            assert state is None


def test_pick_start_gated_out_returns_reset():
    # A populated pool whose reach EMA is below threshold must NOT be
    # selected — the agent can't get there from reset yet.
    mgr = _mgr(reset_fraction=0.0, reach_threshold=0.15)
    mgr.save_scored(3, b"cp3_state", 100, True, bonus=10, source_cp=0)
    mgr.reset_reach_ema[3] = 0.05  # below the gate
    for _ in range(20):
        level, state, _, _ = mgr.pick_start()
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
    # H-T weights by (1 - goal_score): CP1 "solved" (high score),
    # CP2 "failing" (low score). (reset/CP0 also competes; we only
    # compare the two deep levels.) goal_score now lives per-pool.
    mgr.checkpoints[1].goal_score = 0.95  # -> weight 0.05
    mgr.checkpoints[2].goal_score = 0.05  # -> weight 0.95
    counts = {0: 0, 1: 0, 2: 0}
    import random

    random.seed(0)
    for _ in range(2000):
        level, _, _, _ = mgr.pick_start()
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
    # H-T weights by (1 - goal_score). Push reset (CP0) out of contention
    # (goal_score 1.0 -> weight ~0) to isolate the CP1-vs-CP2 floor
    # behavior. CP1 "solved", CP2 "failing". goal_score now lives per-pool.
    mgr.checkpoints[0].goal_score = 1.0
    mgr.checkpoints[1].goal_score = 0.95  # "solved" -> raw weight 0.05
    mgr.checkpoints[2].goal_score = 0.05  # "failing" -> raw weight 0.95
    import random

    random.seed(0)
    counts = {0: 0, 1: 0, 2: 0}
    for _ in range(4000):
        level, _, _, _ = mgr.pick_start()
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
    mgr.save_checkpoint(1, b"seed")  # defaults source_cp=0, bonus=0, stack=None
    assert len(mgr.checkpoints[1]) == 1
    assert mgr.checkpoints[1].states[0] == (0, 0, b"seed", None, frozenset())


def test_disk_roundtrip_preserves_entry(tmp_path):
    mgr = _mgr()
    mgr.save_scored(1, b"new_state", 100, True, bonus=42, source_cp=0)
    p = tmp_path / "checkpoints.pkl"
    mgr.save_to_disk(str(p))

    mgr2 = _mgr()
    mgr2.load_from_disk(str(p))
    # Stack-less play snapshot -> 4-tuple with stack None.
    assert mgr2.checkpoints[1].states[0] == (0, 42, b"new_state", None, frozenset())


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
    # Legacy entry gets source_cp = level (1), original bonus, stack None.
    assert mgr.checkpoints[1].states[0] == (1, 55, b"old1", None, frozenset())


# ---------------------------------------------------------------------------
# H-AB: frame-stack carried per entry
# ---------------------------------------------------------------------------


def test_insert_stores_and_samples_stack():
    mgr = _mgr(reset_fraction=0.0)
    blob = {"sig": ("g",), "frames": ["f0", "f1"]}
    mgr.save_scored(1, b"s", 100, True, bonus=1, source_cp=0, stack=blob)
    entry = mgr.checkpoints[1].states[0]
    assert entry[2] == b"s"
    assert entry[3] is blob


def test_stackless_entry_defaults_none():
    mgr = _mgr(reset_fraction=0.0)
    mgr.save_scored(1, b"s", 100, True, bonus=1, source_cp=0)  # no stack
    assert mgr.checkpoints[1].states[0][3] is None


def test_disk_roundtrip_preserves_stack(tmp_path):
    mgr = _mgr(reset_fraction=0.0)
    blob = {"sig": ("gray", (84, 84), 4, None, (84, 84, 1)), "frames": [1, 2, 3, 4]}
    mgr.save_scored(1, b"s", 100, True, bonus=7, source_cp=0, stack=blob)
    p = tmp_path / "checkpoints.pkl"
    mgr.save_to_disk(str(p))
    mgr2 = _mgr(reset_fraction=0.0)
    mgr2.load_from_disk(str(p))
    entry = mgr2.checkpoints[1].states[0]
    assert entry[:3] == (0, 7, b"s")
    assert entry[3] == blob


# ---------------------------------------------------------------------------
# Waypoint start-pools (optional, non-gating, position-based)
# ---------------------------------------------------------------------------


def test_waypoint_saved_creates_pool():
    mgr = _mgr(reset_fraction=0.0)
    mgr.save_waypoint("L34_top", b"wp_state", 100, True, source_cp=0, bonus=5)
    assert "L34_top" in mgr.waypoints
    assert len(mgr.waypoints["L34_top"]) == 1
    assert mgr.waypoints["L34_top"].states[0] == (0, 5, b"wp_state", None, frozenset())


def test_waypoint_rejected_when_doomed():
    # Survival gate (parity with save_scored): a capture whose episode died
    # too soon after it (didn't survive min_survival and made no CP progress)
    # is NOT admitted. This is the fix for the dying-fall Lesc_bot seeds.
    mgr = _mgr(reset_fraction=0.0)  # min_survival_steps=30
    mgr.save_waypoint("Lesc_bot", b"doomed", survived_steps=7, reached_next=False)
    assert "Lesc_bot" not in mgr.waypoints
    assert mgr.wp_rejected_precarious["Lesc_bot"] == 1


def test_waypoint_admitted_when_survived():
    mgr = _mgr(reset_fraction=0.0)
    mgr.save_waypoint("Lesc_bot", b"ok", survived_steps=50, reached_next=False)
    assert len(mgr.waypoints["Lesc_bot"]) == 1
    assert mgr.wp_admit_survived["Lesc_bot"] == 1


def test_waypoint_admitted_when_reached_next_even_if_short():
    # Leniency parity: reaching the next target admits even a short survival
    # (proves reachability), same as fruit checkpoints.
    mgr = _mgr(reset_fraction=0.0)
    mgr.save_waypoint("Lesc_bot", b"ok", survived_steps=3, reached_next=True)
    assert len(mgr.waypoints["Lesc_bot"]) == 1
    assert mgr.wp_admit_reached["Lesc_bot"] == 1


def test_waypoint_gate_matches_save_scored():
    # Same input -> same verdict for WP and CP: single source of truth
    # (_admit_by_play), so the survival gate can't diverge between them.
    for survived, reached in [(7, False), (50, False), (3, True)]:
        cp = _mgr(reset_fraction=0.0)
        wp = _mgr(reset_fraction=0.0)
        cp.save_scored(1, b"s", survived, reached, bonus=0, source_cp=0)
        wp.save_waypoint("W", b"s", survived, reached)
        cp_admitted = len(cp.checkpoints[1]) == 1
        wp_admitted = "W" in wp.waypoints
        assert cp_admitted == wp_admitted


def test_waypoint_stores_stack():
    mgr = _mgr(reset_fraction=0.0)
    blob = {"sig": ("g",), "frames": ["a"]}
    mgr.save_waypoint("L34_top", b"wp", 100, True, source_cp=0, bonus=5, stack=blob)
    assert mgr.waypoints["L34_top"].states[0][3] is blob


def test_fresh_waypoint_has_max_weight():
    # A newly-captured WP has goal_score 0 -> weight 1.0 -> heavily sampled
    # (this is what gives "more reps further down" automatically).
    mgr = _mgr(reset_fraction=0.0)
    mgr.save_waypoint("L45a_bot", b"s", 100, True)
    assert mgr.waypoints["L45a_bot"].weight() == pytest.approx(1.0)


def test_pick_start_can_return_waypoint():
    mgr = _mgr(reset_fraction=0.0)
    mgr.save_waypoint("L34_top", b"wp_state", 100, True)
    # Make reset unattractive (goal_score 1 -> weight ~0) so the WP wins.
    mgr.checkpoints[0].goal_score = 1.0
    import random

    random.seed(0)
    seen_wp = False
    for _ in range(50):
        key, state, _, _ = mgr.pick_start()
        if isinstance(key, str):
            seen_wp = True
            assert key == "L34_top"
            assert state == b"wp_state"
    assert seen_wp


def test_record_episode_waypoint_updates_only_wp_goal_score():
    mgr = _mgr(reset_fraction=0.0)
    mgr.save_waypoint("L34_top", b"s", 100, True)
    reach_before = list(mgr.reset_reach_ema)
    seg_before = list(mgr.seg_success_ema)
    w0 = mgr.waypoints["L34_top"].weight()
    # A WP start that reaches the princess (max) raises its goal_score.
    for _ in range(200):
        mgr.record_episode("L34_top", reached_level=mgr.N_RUNGS + 1)
    # WP weight dropped (self-regulated), but the CP/reset metrics are
    # untouched — WPs are non-gating and out of the success stats.
    assert mgr.waypoints["L34_top"].weight() < w0
    assert mgr.reset_reach_ema == reach_before
    assert mgr.seg_success_ema == seg_before


def test_record_episode_unknown_waypoint_is_safe():
    mgr = _mgr(reset_fraction=0.0)
    mgr.record_episode("nonexistent_wp", 1)  # no pool -> no-op, no crash


def test_waypoint_group_share_is_count_invariant():
    # H-AK: waypoints compete as ONE group whose weight is the MEAN of
    # member weights, so reset's share does NOT collapse as more waypoints
    # are added. (Under the old sum-of-votes rule, 20 fresh WPs would crowd
    # reset from ~33% down to ~2%.)
    import random

    def reset_share(n_waypoints, trials=6000):
        mgr = _mgr(reset_fraction=0.0, max_states_per_checkpoint=50)
        mgr.checkpoints[0].goal_score = 0.5  # reset weight 0.5
        for i in range(n_waypoints):
            mgr.save_waypoint(f"W{i}", b"s", 100, True)  # each fresh -> weight 1.0
        random.seed(0)
        c0 = 0
        for _ in range(trials):
            key, _, _, _ = mgr.pick_start()
            if key == 0:
                c0 += 1
        return c0 / trials

    share_1 = reset_share(1)
    share_20 = reset_share(20)
    # Mean-group: WP group weight = 1.0 whether 1 or 20 WPs, so reset share
    # ~ 0.5 / (0.5 + 1.0) = 0.333 in BOTH cases (count-invariant).
    assert share_1 == pytest.approx(0.333, abs=0.05)
    assert share_20 == pytest.approx(0.333, abs=0.05)
    # The whole point: 20 waypoints don't starve reset.
    assert share_20 > 0.25


def test_waypoint_group_internal_split_by_goal_score():
    # Within the group, waypoints are still drawn by 1 - goal_score, so a
    # fresh (unmastered) WP is sampled far more than a mastered one.
    import random

    mgr = _mgr(reset_fraction=0.0, max_states_per_checkpoint=50)
    mgr.checkpoints[0].goal_score = 1.0  # push reset out -> group almost always
    mgr.save_waypoint("mastered", b"m", 100, True)
    mgr.waypoints["mastered"].goal_score = 0.9  # weight 0.1
    mgr.save_waypoint("fresh", b"f", 100, True)  # goal_score 0.0 -> weight 1.0
    random.seed(0)
    counts = {"mastered": 0, "fresh": 0}
    for _ in range(4000):
        key, _, _, _ = mgr.pick_start()
        if isinstance(key, str):
            counts[key] += 1
    assert counts["fresh"] > counts["mastered"] * 3


def test_no_waypoints_leaves_cp_selection_unchanged():
    # With no WPs captured, pick_start returns only int CP keys (L1-safe:
    # identical to pre-WP behavior).
    mgr = _mgr(reset_fraction=0.0)
    mgr.save_scored(2, b"cp2", 100, True, bonus=10, source_cp=0)
    mgr.reset_reach_ema[2] = 1.0
    import random

    random.seed(0)
    for _ in range(50):
        key, _, _, _ = mgr.pick_start()
        assert isinstance(key, int)


def test_waypoint_disk_roundtrip(tmp_path):
    mgr = _mgr(reset_fraction=0.0)
    mgr.save_waypoint("L34_top", b"wp_state", 100, True, source_cp=0, bonus=7)
    mgr.waypoints["L34_top"].goal_score = 0.42
    p = tmp_path / "checkpoints.pkl"
    mgr.save_to_disk(str(p))

    mgr2 = _mgr(reset_fraction=0.0)
    mgr2.load_from_disk(str(p))
    assert "L34_top" in mgr2.waypoints
    assert mgr2.waypoints["L34_top"].states[0] == (0, 7, b"wp_state", None, frozenset())
    assert mgr2.waypoints["L34_top"].goal_score == pytest.approx(0.42)


def test_pre_wp_checkpoint_file_loads_without_waypoints(tmp_path):
    # A checkpoint file with no "waypoints" key (pre-WP) must load fine.
    mgr = _mgr(reset_fraction=0.0)
    mgr.save_scored(1, b"cp1", 100, True, bonus=10, source_cp=0)
    p = tmp_path / "checkpoints.pkl"
    mgr.save_to_disk(str(p))
    import pickle

    data = pickle.load(open(p, "rb"))
    del data["waypoints"]  # simulate an old file
    pickle.dump(data, open(p, "wb"))

    mgr2 = _mgr(reset_fraction=0.0)
    mgr2.load_from_disk(str(p))  # must not crash
    assert mgr2.waypoints == {}
    assert len(mgr2.checkpoints[1]) == 1
