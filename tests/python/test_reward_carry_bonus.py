"""A flat, once-per-episode payment for the rope-carry pose.

WHY THIS TERM EXISTS. L4's rope 2 is an expected-value problem, not an exploration one.
Measured on the `Low2_launch` pad with `l4_pad_reward.py`, through the trainer's own
reward:

    hold still                                    0.000
    walk LEFT off the edge                       +0.075   CERTAIN
    jump and miss the rope                        0.000
    completed crossing (NOOP:20,JUMP_LEFT:70)    +2.940   median 2.920, n=6

The ordering was already correct -- a crossing pays 39x the cliff-walk -- but
attempting it is worth ~1% x 2.940 = +0.029 expected against a certain +0.075, and the
whole +2.94 arrives on LANDING. So stepping off the edge is the better bet and the agent
takes it, while a genuine attempt pays nothing at all.

The two things that can go wrong with the fix are both pinned below.

PLACEMENT. The carry pose is AIRBORNE, so a payment after the (D2) airborne freeze is
unreachable -- exactly the defect that made every jump-landing milestone unmarkable
until 2026-09-14 (see test_reward_airborne_marking.py). Verified by moving the payment
below that return on a backed-up copy of rewards.py: 6 of the 8 tests here fail, and
they pass again on restore.

ONCE PER EPISODE. The recorded crossings show the carry pose in two bursts (crossing 1:
poses [4,10,15,10,15,10,5]), so a per-frame payment banks 2-5x its face value for one
manoeuvre -- at +1.0 per frame that is up to 70% of a maximum episode (episodes.csv:
median 6.8, p75 57.2, max 70.5). `test_paid_once_per_episode` pins that.

No emulator needed.
"""

from retro_ai.training.rewards import RewardContext, reset_reward
from retro_ai.training.rewards import create as create_reward

PARAMS = {
    "scale": 0.01,
    "fruit_scale": 0.01,
    "princess_scale": 0.05,
    "level": 4,
    "gamma": 1.0,
    "defer_fruit_credit": True,
    "waypoint_reward_tol": 2,
    "mark_airborne": True,
    "pay_on_target_change": True,
    "ladder_segment_shaping": True,
    "waypoint_reach_mode": "sprite",
}

CARRY_LEFT, JUMP_LEFT, FALL, WALK_LEFT = 15, 10, 11, 4
# Floor 12's left edge, where the rope-2 attempt starts. px 188 => x_ram 45.
PAD_X, PAD_Y = 45, 70


def _fresh(**over):
    p = dict(PARAMS)
    p.update(over)
    return create_reward("fruit_bonus_path_progress_pbrs_grounded", p)


def _ctx(pose, x_ram=PAD_X, y=PAD_Y, **kw):
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


def test_off_by_default_so_every_existing_config_is_unchanged():
    fn = _fresh()
    before = fn(_ctx(CARRY_LEFT))
    fn2 = _fresh(carry_pose_bonus=0.0)
    assert before == fn2(_ctx(CARRY_LEFT))


def test_paid_while_airborne():
    """The carry pose is airborne. A payment below the (D2) freeze never fires."""
    fn = _fresh(carry_pose_bonus=1.0)
    assert fn(_ctx(CARRY_LEFT)) == 1.0


def test_paid_once_per_episode():
    fn = _fresh(carry_pose_bonus=1.0)
    assert fn(_ctx(CARRY_LEFT, step=10)) == 1.0
    # the real pose sequence returns to the carry after a jump frame
    fn(_ctx(JUMP_LEFT, step=11))
    assert fn(_ctx(CARRY_LEFT, step=12)) == 0.0
    assert fn(_ctx(CARRY_LEFT, step=13)) == 0.0


def test_the_allowance_comes_back_after_reset():
    fn = _fresh(carry_pose_bonus=1.0)
    assert fn(_ctx(CARRY_LEFT, step=10)) == 1.0
    assert fn(_ctx(CARRY_LEFT, step=11)) == 0.0
    reset_reward(fn)
    assert fn(_ctx(CARRY_LEFT, step=10)) == 1.0


def test_not_paid_on_a_death_frame():
    """A fatal grab must pay nothing -- it is the failure this term replaces, and (D3)
    and (D4) already refuse to credit shaping and fruit on a death frame."""
    fn = _fresh(carry_pose_bonus=1.0)
    assert fn(_ctx(CARRY_LEFT, died=True)) == 0.0
    # and the allowance is still intact for a real carry later in the episode
    assert fn(_ctx(CARRY_LEFT, step=11)) == 1.0


def test_other_poses_are_not_paid():
    fn = _fresh(carry_pose_bonus=1.0)
    for pose in (JUMP_LEFT, FALL, WALK_LEFT):
        f = _fresh(carry_pose_bonus=1.0)
        assert f(_ctx(pose)) != 1.0, pose
    assert fn(_ctx(CARRY_LEFT)) == 1.0


def test_the_paid_pose_set_is_configurable():
    """Pose 15 is L4's LEFTWARD carry; rope 1 is crossed rightward and shows pose 14. A
    level that needs both, or a different code, should not need a code change."""
    fn = _fresh(carry_pose_bonus=1.0, carry_pose_ids=(14, 15))
    assert fn(_ctx(14)) == 1.0
    fn2 = _fresh(carry_pose_bonus=1.0, carry_pose_ids=(14,))
    assert fn2(_ctx(CARRY_LEFT)) != 1.0


def test_it_does_not_disturb_the_grounded_shaping():
    """The bonus adds to `reward` only. A grounded frame with the term off and on must
    differ by exactly the bonus, so the potential and its telescoping are untouched."""
    off = _fresh()
    on = _fresh(carry_pose_bonus=1.0)
    seq = [
        _ctx(WALK_LEFT, x_ram=45, step=10),
        _ctx(WALK_LEFT, x_ram=44, step=11),
        _ctx(WALK_LEFT, x_ram=43, step=12),
    ]
    a = [off(c) for c in seq]
    b = [on(c) for c in seq]
    assert a == b, "a grounded walk must be identical; the bonus fires on no pose here"
