"""Eval and training must count route depth the same way: satisfied GROUPS, not ids.

The trainer counted rung-per-group since ded0032 (see test_progress_rungs_are_groups);
the eval rollout kept counting mandatory ids. The difference only shows when an episode
reaches two names for one step, which on L4 means crossing rope 2: the crossing flies
over both members of the floor-13 OR-group, `Low2` (px 128) and `Lhi_down_bot` (px 104).
Measured on the v30 champion, every princess episode read `max_rung` 13 of 13 in eval
while the trainer's ladder has 12 rungs.

`targets.progress_ladder` is now the one definition; the trainer delegates to it and the
rollout uses it. Verified before the move that the trainer's ladder is unchanged on
levels 1-4.

No emulator needed.
"""

from __future__ import annotations

from retro_ai.training.targets import build_targets, progress_ladder, rung_of

# A real v30 princess episode's reached_points, from eval_from_reset --reach-mode
# sprite (2026-10-07). The old id count gave 13.
V30_PRINCESS = {
    "F1", "Fr1", "Fr1_launch", "Fr2", "Fr2_launch", "Lascent_top", "Lclimb1_top",
    "Lclimb2_top", "Lclimb3_top", "Lfruit_bot", "Lfruit_top", "Lhi_down_bot", "Low1",
    "Low2", "Lprincess_top", "Rope1", "Rope1_launch", "Spring", "Step",
}  # fmt: skip


def _old_id_count(level, reached):
    """The rollout's pre-fix rule, verbatim, as the reference for what changed."""
    ids = set()
    for t in build_targets(level):
        if t.mandatory and t.kind != "princess":
            ids.add(t.id)
            if t.node_ident:
                ids.add(t.node_ident)
    return len(set(reached) & ids)


def test_a_rope2_princess_episode_reaches_every_rung_once():
    groups, n = progress_ladder(4)
    assert n == 12
    assert rung_of(groups, V30_PRINCESS) == 12
    assert _old_id_count(4, V30_PRINCESS) == 13, "the defect this replaces"


def test_both_floor13_arrivals_are_one_rung():
    groups, _ = progress_ladder(4)
    base = rung_of(groups, {"Low1"})
    assert rung_of(groups, {"Low1", "Low2"}) == base + 1
    assert rung_of(groups, {"Low1", "Lhi_down_bot"}) == base + 1
    assert rung_of(groups, {"Low1", "Low2", "Lhi_down_bot"}) == base + 1


def test_ids_and_groups_agree_when_no_step_is_reached_twice():
    """Every L4 eval before rope 2 was crossable: the change must not move them. Any
    reached set with at most one name per step scores the same under both rules."""
    groups, _ = progress_ladder(4)
    one_name_each = [sorted(g)[0] for g in groups]
    for k in range(len(one_name_each) + 1):
        reached = set(one_name_each[:k]) | {"Low1", "Rope1_launch"}  # non-mandatory
        assert rung_of(groups, reached) == _old_id_count(4, reached) == k


def test_every_level_has_a_ladder_and_no_name_in_two_groups():
    for level in (1, 2, 3, 4):
        groups, n = progress_ladder(level)
        assert n == len(groups) > 0
        seen: set = set()
        for g in groups:
            assert not (g & seen), (level, g & seen)
            seen |= g


def test_none_and_empty_reach_nothing():
    groups, _ = progress_ladder(4)
    assert rung_of(groups, None) == 0
    assert rung_of(groups, set()) == 0
