"""A mandatory JUMP LANDING must be markable while the agent is still airborne.

The defect this pins (measured 2026-09-14, 120 from-reset episodes on L4 v14's policy):

    target   px,y        sprite test fires   reward marked   poses it fires at
    Rope1   108,118          107/120            2/120        9(jump-right)x126
    Spring  212, 94           95/120           14/120        9(jump-right)x321

All five of L4's mandatory targets that are jump landings -- Fr1, Rope1, Spring, Step,
Low2 -- are points the agent FLIES through, so a landing is exactly when it is airborne.
The marking loop used to sit after ``if airborne: return reward``, which made it
unreachable on those frames however "ungated" the loop itself looked.

The cost is not a missing credit. An unmarked group stays in the potential's sum and
keeps pulling toward itself, and on floor 12 the unmarked backward terms outweigh the
forward ones and REVERSE the gradient: sum 984 at px 184 against 888 at px 232, i.e.
away from the rope-2 gap, where marking groups 0-7 gives 320 against 368, toward it.

What must NOT change is the airborne shaping FREEZE. Returning Phi=None while airborne
rebaselined prev_phi, deleted the return-leg debt and paid a free +Phi per
approach-then-jump-back cycle, which PPO farmed for 15M steps (H-AH). These tests assert
marking moved and shaping did not.

No emulator needed.
"""

from retro_ai import games
from retro_ai.games import yeti
from retro_ai.training import rewards as rw
from retro_ai.training.rewards import RewardContext
from retro_ai.training.rewards import create as create_reward

PARAMS = {
    "scale": 0.01,
    "fruit_scale": 0.01,
    "princess_scale": 0.05,
    "level": 4,
    "gamma": 1.0,
    "defer_fruit_credit": True,
    "waypoint_reward_tol": 2,
    "ladder_segment_shaping": True,
    "waypoint_reach_mode": "sprite",
}

# Rope1 / J6_7_b: mandatory, the rope-1 landing on floor 7. px 108 => x_ram 25, y 118.
ROPE1_X, ROPE1_Y = 25, 118
JUMP_RIGHT, JUMP_LEFT, FALL, DEATH_ANIM, WALK_RIGHT = 9, 10, 11, 12, 0


def _fresh(**over):
    p = dict(PARAMS)
    p.update(over)
    return create_reward("fruit_bonus_path_progress_pbrs_grounded", p)


def _ctx(x_ram, y, pose, **kw):
    return RewardContext(
        prev_fruits=1,
        curr_fruits=1,
        prev_bonus=1000,
        curr_bonus=1000,
        prev_score=0,
        curr_score=0,
        prev_lives=6,
        curr_lives=6,
        step_count=kw.pop("step", 10),
        curr_x=x_ram,
        curr_y=y,
        fruits_present=kw.pop("fruits_present", (False,)),
        pose=pose,
        **kw,
    )


def _group_of(ident):
    """Index of the reward OR-group containing ``ident``, for L4."""
    from retro_ai.training.yeti_map import get_level_map

    for gi, members in enumerate(get_level_map(4).reward_waypoints):
        if ident in members:
            return gi
    raise AssertionError(f"{ident} is not a reward group on L4")


def test_landing_is_marked_while_airborne():
    """The whole point: pose 9 (jump-right) at the Rope1 anchor must mark the group."""
    gi = _group_of("J6_7_b")
    fn = _fresh()
    fn(_ctx(ROPE1_X, ROPE1_Y, JUMP_RIGHT))
    assert gi in fn._reached_wp, (
        "a mandatory jump landing was not marked while airborne -- this is the defect "
        "that reversed floor 12's gradient"
    )


def test_landing_is_also_marked_when_grounded():
    """Regression guard: the grounded path must keep working."""
    gi = _group_of("J6_7_b")
    fn = _fresh()
    fn(_ctx(ROPE1_X, ROPE1_Y, WALK_RIGHT))
    assert gi in fn._reached_wp


def test_falling_past_a_waypoint_does_not_mark_it():
    """Fall (11) and the death anim (12) are blocked: falling PAST a waypoint is not
    reaching it. Measured as free -- across 120 episodes the blocklist and no gate at
    all fired on identical episode counts for every mandatory L4 target."""
    gi = _group_of("J6_7_b")
    for pose in (FALL, DEATH_ANIM):
        fn = _fresh()
        fn(_ctx(ROPE1_X, ROPE1_Y, pose))
        assert gi not in fn._reached_wp, f"pose {pose} must not mark a waypoint"


def test_a_death_frame_does_not_mark():
    gi = _group_of("J6_7_b")
    fn = _fresh()
    fn(_ctx(ROPE1_X, ROPE1_Y, JUMP_RIGHT, died=True))
    assert gi not in fn._reached_wp


def test_airborne_frames_still_pay_nothing():
    """The shaping FREEZE is untouched: a mid-air mark changes the target set but cannot
    itself pay. Guards H-AH, where crediting/rebaselining airborne gave PPO a farmable
    +Phi per approach-then-jump-back cycle."""
    fn = _fresh()
    # settle a baseline on a grounded frame first
    fn(_ctx(ROPE1_X, ROPE1_Y, WALK_RIGHT, step=1))
    fresh = _fresh()
    fresh(_ctx(ROPE1_X, ROPE1_Y, WALK_RIGHT, step=1))
    r = fresh(_ctx(ROPE1_X - 2, ROPE1_Y, JUMP_RIGHT, step=2))
    assert r == 0.0, f"an airborne frame paid {r}; shaping must stay frozen"


def test_reward_mark_blocklist_matches_yeti():
    """rewards.py keeps its own pose constants to stay game-agnostic; they must not
    drift from yeti's. If a pose is added to one, add it to the other."""
    assert rw.MARK_BLOCKED_POSES == yeti.NON_TRAVERSAL_POSES
    assert games is not None  # import guard: yeti must be importable for the comparison
