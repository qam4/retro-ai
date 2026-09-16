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


def test_mark_airborne_false_reproduces_the_old_placement():
    """The control arm must be a CONFIG, not a git checkout.

    v14 (marking after the airborne return) and v15 (before it) are an A/B whose
    control has to be reproducible, so `mark_airborne: false` restores the old
    placement: marking is then only reachable on frames in the reward's SURFACE_POSES,
    which is what made a jump landing nearly unmarkable.
    """
    gi = _group_of("J6_7_b")

    off = _fresh(mark_airborne=False)
    off(_ctx(ROPE1_X, ROPE1_Y, JUMP_RIGHT))
    assert gi not in off._reached_wp, "mark_airborne=False must not mark mid-jump"

    # ...but it still marks on a grounded frame, so the control arm is the OLD behaviour
    # and not simply "marking disabled".
    off2 = _fresh(mark_airborne=False)
    off2(_ctx(ROPE1_X, ROPE1_Y, WALK_RIGHT))
    assert gi in off2._reached_wp

    on = _fresh(mark_airborne=True)
    on(_ctx(ROPE1_X, ROPE1_Y, JUMP_RIGHT))
    assert gi in on._reached_wp


def test_default_is_mark_airborne():
    """Pin the default, in the same spirit as the reach-mode default: nothing pinned the
    previous placement, so a silent revert would have been invisible."""
    gi = _group_of("J6_7_b")
    fn = _fresh()  # no mark_airborne key at all
    fn(_ctx(ROPE1_X, ROPE1_Y, JUMP_RIGHT))
    assert gi in fn._reached_wp


# ---------------------------------------------------------------------------
# pay_on_target_change (D6): marking a landing mid-air must not DELETE the
# landing's payment.
#
# Marking while airborne was necessary (above) but not sufficient. The shaping
# freeze banks a whole jump into the landing frame -- prev_phi is held while
# airborne, so the landing pays everything covered since take-off. A mid-air
# mark changes the waypoint list, and the code cannot see that until the next
# grounded frame, which IS the landing. The pre-existing guard for a list change
# is to skip the frame, so the skip lands on the most valuable frame of the run.
#
# Measured on L4 climb2 -> Spring (v16a policy, 29/29 crossings, one trajectory
# replayed through both variants): the floor-9 arrival pays +3.200 when marking
# happens on the ground and exactly 0.000 when it happens in the air; the floor-7
# Rope1 arrival loses 5.120 the same way. With pay_on_target_change the same
# arrival pays +3.920 and only 3 frames of 359 differ from the broken arm.
#
# The property these tests pin is that marking POSITION stops mattering: a frame
# is priced against the list it started with, so it is paid for movement and
# never for a deletion.
# ---------------------------------------------------------------------------

# floor 7 spans px 104..168, i.e. x_ram 24..40. The Rope1 anchor is px 108 and its
# sprite test fires for x_ram 24..26, so x_ram 30/34 are on the same floor and clear
# of it -- a grounded frame there marks nothing.
F7_A, F7_B = 30, 34


def _run(fn, seq):
    """Feed ``seq`` of (x_ram, y, pose) and return the reward of the LAST frame."""
    r = 0.0
    for step, (x, y, pose) in enumerate(seq, start=1):
        r = fn(_ctx(x, y, pose, step=step))
    return r


def _land_after_airborne_mark():
    """grounded -> airborne ON the Rope1 anchor -> grounded elsewhere on floor 7."""
    return [
        (F7_A, ROPE1_Y, WALK_RIGHT),  # settles the baseline
        (ROPE1_X, ROPE1_Y, JUMP_RIGHT),  # mid-air, on the anchor
        (F7_B, ROPE1_Y, WALK_RIGHT),  # the landing frame
    ]


def test_airborne_mark_deletes_the_landing_payment_by_default():
    """Pins the defect, so the fix below is measured against something real."""
    gi = _group_of("J6_7_b")
    fn = _fresh(mark_airborne=True)  # pay_on_target_change absent => today's skip
    r = _run(fn, _land_after_airborne_mark())
    assert gi in fn._reached_wp, "precondition: the mid-air frame must mark the group"
    assert r == 0.0, (
        f"the landing frame paid {r}; without pay_on_target_change the list change is "
        "expected to swallow it -- if this now pays, the default changed"
    )


def test_pay_on_target_change_restores_the_landing_payment():
    """The landing pays again, and pays exactly what movement earned.

    The reference is a run where the group is never marked at all
    (``mark_airborne=False`` plus a landing clear of the anchor), so the landing is
    an ordinary shaping frame with the full list. Equality with that is the real
    property: the frame is priced for MOVEMENT, not for the deletion, so where the
    mark happened stops affecting the total.
    """
    gi = _group_of("J6_7_b")
    fixed = _fresh(mark_airborne=True, pay_on_target_change=True)
    r_fixed = _run(fixed, _land_after_airborne_mark())
    assert gi in fixed._reached_wp

    never = _fresh(mark_airborne=False)
    r_never = _run(never, _land_after_airborne_mark())
    assert gi not in never._reached_wp, "reference must leave the group on the list"

    assert r_fixed > 0.0, "the landing frame must be paid"
    # Exact: path distances are ints, so both sums are integer-exact before scaling.
    assert r_fixed == r_never, (
        f"landing paid {r_fixed} but movement alone is worth {r_never}: the frame is "
        "being paid for the list change, not just for moving"
    )


def test_pay_on_target_change_is_a_noop_when_the_list_is_unchanged():
    """No list change => phi_pay is phi, so the flag cannot alter anything."""
    seq = [(F7_A, ROPE1_Y, WALK_RIGHT), (F7_B, ROPE1_Y, WALK_RIGHT)]
    off = _fresh(mark_airborne=True)
    on = _fresh(mark_airborne=True, pay_on_target_change=True)
    r_off = _run(off, seq)
    r_on = _run(on, seq)
    assert (
        off._reached_wp == on._reached_wp
    ), "precondition: neither run marked anything"
    assert r_off == r_on, f"flag changed an unchanged-list frame: {r_off} vs {r_on}"


def test_default_is_off():
    """Byte-identical to the shipped reward unless a run opts in."""
    fn = _fresh(mark_airborne=True)
    explicit = _fresh(mark_airborne=True, pay_on_target_change=False)
    assert _run(fn, _land_after_airborne_mark()) == _run(
        explicit, _land_after_airborne_mark()
    )


def test_airborne_frames_still_pay_nothing_with_the_flag_on():
    """The freeze (H-AH) must survive the new flag: mid-air frames still pay 0."""
    fn = _fresh(mark_airborne=True, pay_on_target_change=True)
    fn(_ctx(F7_A, ROPE1_Y, WALK_RIGHT, step=1))
    r = fn(_ctx(ROPE1_X, ROPE1_Y, JUMP_RIGHT, step=2))
    assert r == 0.0, f"an airborne frame paid {r}; shaping must stay frozen"
