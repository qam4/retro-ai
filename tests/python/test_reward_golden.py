"""GOLDEN (characterization) reward sequences — the safety net for refactors.

These freeze the EXACT per-step reward for fixed trajectories on levels 1, 2 and
3. Their only job is to make a refactor PROVABLY behaviour-preserving instead of
hoped to be: if a change is meant to be a pure restructuring, these must not
move; if a change is meant to alter behaviour, it must be opt-in (a param) so
these still hold with it off.

Written ahead of the "Target unification" work (one shared code path for
detection / credit / seeding / restore / metrics across fruits, waypoint
milestones and the princess), which touches the reward used by the SOLVED L1
(99.7%) and L2 (98.7%) champions.

Values generated from the then-current implementation via
debug/gen_reward_golden.py. Do NOT edit them to make a test pass — a diff here
means behaviour changed, which is either a bug or something that needs to be
gated behind a param.
"""

from __future__ import annotations

import pytest
from retro_ai.training.rewards import RewardContext, create

_L1 = dict(scale=0.01, fruit_scale=0.01, princess_scale=0.05, level=1, gamma=1.0)
_L2 = dict(
    scale=0.01,
    fruit_scale=0.01,
    princess_scale=0.05,
    level=2,
    gamma=1.0,
    defer_fruit_credit=True,
)
_L3 = dict(
    scale=0.01,
    fruit_scale=0.01,
    princess_scale=0.05,
    level=3,
    gamma=1.0,
    defer_fruit_credit=True,
    waypoint_reward_tol=2,
    ladder_segment_shaping=True,
)


def _ctx(x, y, pose, fruits_present, prev_fruits, curr_fruits, died=False, pr=False):
    return RewardContext(
        prev_fruits=prev_fruits,
        curr_fruits=curr_fruits,
        prev_bonus=1000,
        curr_bonus=1000,
        prev_score=0,
        curr_score=0,
        prev_lives=5,
        curr_lives=5,
        step_count=0,
        curr_x=x,
        curr_y=y,
        fruits_present=fruits_present,
        pose=pose,
        died=died,
        princess_touched=pr,
    )


# label -> (formula, params, steps)
# steps: (x, y, pose, fruits_present, prev_fruits, curr_fruits, died, princess)
CASES = {
    "l1_grounded_climb": (
        "fruit_bonus_path_progress_pbrs_grounded",
        _L1,
        [
            (30, 184, 0, (True, True, True, True), 4, 4, False, False),
            (34, 184, 1, (True, True, True, True), 4, 4, False, False),
            (46, 184, 2, (True, True, True, True), 4, 4, False, False),
            (46, 152, 8, (True, True, True, True), 4, 4, False, False),
            (46, 152, 0, (False, True, True, True), 4, 3, False, False),
            (50, 152, 1, (False, True, True, True), 3, 3, False, False),
            (50, 152, 11, (False, True, True, True), 3, 3, True, False),
        ],
    ),
    "l2_grounded_descend": (
        "fruit_bonus_path_progress_pbrs_grounded",
        _L2,
        [
            (20, 30, 0, (True, True), 2, 2, False, False),
            (20, 54, 8, (True, True), 2, 2, False, False),
            (24, 54, 1, (True, True), 2, 2, False, False),
            (24, 78, 8, (True, True), 2, 2, False, False),
            (16, 126, 0, (False, True), 2, 1, False, False),
            (16, 126, 0, (False, True), 1, 1, False, False),
            (16, 126, 11, (False, True), 1, 1, True, False),
        ],
    ),
    "l3_grounded_route": (
        "fruit_bonus_path_progress_pbrs_grounded",
        _L3,
        [
            (18, 94, 0, (True,), 1, 1, False, False),
            (24, 94, 0, (True,), 1, 1, False, False),
            (33, 94, 13, (True,), 1, 1, False, False),
            (40, 158, 0, (True,), 1, 1, False, False),
            (58, 110, 8, (True,), 1, 1, False, False),
            (70, 86, 8, (True,), 1, 1, False, False),
            (70, 86, 0, (True,), 1, 1, True, False),
        ],
    ),
    "l1_pbrs_plain": (
        "fruit_bonus_path_progress_pbrs",
        _L1,
        [
            (30, 184, 0, (True, True, True, True), 4, 4, False, False),
            (46, 184, 2, (True, True, True, True), 4, 4, False, False),
            (46, 152, 8, (True, True, True, True), 4, 4, False, False),
        ],
    ),
}

GOLDEN = {
    "l1_grounded_climb": [0.0, -0.32, 0.16, 0.96, 10.0, 0.16, 0.0],
    "l2_grounded_descend": [0.0, 0.8, 0.32, 2.72, 10.0, 0.0, 0.0],
    "l3_grounded_route": [0.0, 2.64, 3.96, 10.12, 0.0, 0.0, 0.0],
    "l1_pbrs_plain": [0.0, -0.16, 0.96],
}


@pytest.mark.parametrize("label", sorted(CASES))
def test_reward_sequence_unchanged(label):
    formula, params, steps = CASES[label]
    fn = create(formula, params)
    if hasattr(fn, "reset"):
        fn.reset()
    got = [round(float(fn(_ctx(*s))), 6) for s in steps]
    assert got == GOLDEN[label], (
        f"{label}: reward sequence changed.\n  expected {GOLDEN[label]}\n"
        f"  got      {got}\nIf this change is intentional it must be gated "
        f"behind a param so the default path still matches."
    )
