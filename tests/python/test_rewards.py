"""Tests for the named-reward registry."""

from __future__ import annotations

import pytest
from retro_ai.training import rewards
from retro_ai.training.rewards import RewardContext, available, create, register


def _ctx(**overrides) -> RewardContext:
    base = dict(
        prev_fruits=4,
        curr_fruits=4,
        prev_bonus=1000,
        curr_bonus=1000,
        prev_score=0,
        curr_score=0,
        prev_lives=5,
        curr_lives=5,
        step_count=0,
    )
    base.update(overrides)
    return RewardContext(**base)


# ---------------------------------------------------------------------------
# Registry plumbing
# ---------------------------------------------------------------------------


def test_built_in_formulas_available():
    names = available()
    assert "fruit_flat" in names
    assert "fruit_bonus" in names
    assert "score_delta_survival" in names


def test_create_unknown_formula_raises():
    with pytest.raises(KeyError):
        create("not_a_real_formula")


def test_register_duplicate_raises():
    # First registration succeeds; second one for the same name should fail.
    @register("dup_test_one")
    def _factory_1(_):
        return lambda _: 0.0

    with pytest.raises(ValueError):

        @register("dup_test_one")
        def _factory_2(_):
            return lambda _: 0.0

    # Clean up so we don't leak test state into other tests.
    rewards._REGISTRY.pop("dup_test_one", None)


# ---------------------------------------------------------------------------
# fruit_flat
# ---------------------------------------------------------------------------


def test_fruit_flat_no_fruit_no_reward():
    fn = create("fruit_flat")
    assert fn(_ctx(prev_fruits=4, curr_fruits=4)) == 0.0


def test_fruit_flat_default_is_ten():
    fn = create("fruit_flat")
    assert fn(_ctx(prev_fruits=4, curr_fruits=3)) == 10.0


def test_fruit_flat_respects_per_fruit_param():
    fn = create("fruit_flat", {"per_fruit": 5.0})
    assert fn(_ctx(prev_fruits=4, curr_fruits=3)) == 5.0


def test_fruit_flat_multiple_fruits_in_one_step():
    fn = create("fruit_flat", {"per_fruit": 10.0})
    assert fn(_ctx(prev_fruits=4, curr_fruits=2)) == 20.0


# ---------------------------------------------------------------------------
# fruit_bonus
# ---------------------------------------------------------------------------


def test_fruit_bonus_no_fruit_no_reward():
    fn = create("fruit_bonus")
    assert fn(_ctx(prev_fruits=4, curr_fruits=4, curr_bonus=500)) == 0.0


def test_fruit_bonus_matches_previous_inline_formula():
    # Old inline formula: reward += (prev_fruits - curr_fruits) * bonus * 0.01
    fn = create("fruit_bonus")
    ctx = _ctx(prev_fruits=4, curr_fruits=3, curr_bonus=800)
    assert fn(ctx) == pytest.approx(1 * 800 * 0.01)


def test_fruit_bonus_scale_param():
    fn = create("fruit_bonus", {"scale": 0.05})
    ctx = _ctx(prev_fruits=4, curr_fruits=3, curr_bonus=1000)
    assert fn(ctx) == pytest.approx(1 * 1000 * 0.05)


def test_fruit_bonus_zero_bonus_yields_zero():
    fn = create("fruit_bonus")
    ctx = _ctx(prev_fruits=4, curr_fruits=3, curr_bonus=0)
    assert fn(ctx) == 0.0


# ---------------------------------------------------------------------------
# score_delta_survival
# ---------------------------------------------------------------------------


def test_score_delta_survival_default_step_bonus_only():
    fn = create("score_delta_survival")
    ctx = _ctx(prev_score=0, curr_score=0)
    assert fn(ctx) == pytest.approx(0.01)


def test_score_delta_survival_scores_increment():
    fn = create("score_delta_survival")
    ctx = _ctx(prev_score=10, curr_score=30)
    assert fn(ctx) == pytest.approx(20 * 0.1 + 0.01)


def test_score_delta_survival_negative_delta_is_clipped():
    # Score reset mid-episode must not produce a negative reward spike.
    fn = create("score_delta_survival")
    ctx = _ctx(prev_score=50, curr_score=0)
    assert fn(ctx) == pytest.approx(0.01)


def test_score_delta_survival_custom_params():
    fn = create(
        "score_delta_survival",
        {"score_scale": 1.0, "step_bonus": 0.0},
    )
    ctx = _ctx(prev_score=0, curr_score=7)
    assert fn(ctx) == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# fruit_bonus_floor_novelty
# ---------------------------------------------------------------------------


def test_floor_novelty_pays_on_first_visit():
    """One-shot bonus the first time a new floor is entered."""
    fn = create("fruit_bonus_floor_novelty", {"scale": 0.01, "novelty_bonus": 1.0})
    # y=182 -> floor 0 (spawn). First visit pays.
    r = fn(_ctx(curr_y=182))
    assert r == pytest.approx(1.0)


def test_floor_novelty_no_reward_on_same_floor():
    fn = create("fruit_bonus_floor_novelty", {"scale": 0.01, "novelty_bonus": 1.0})
    fn(_ctx(curr_y=182))  # visit floor 0
    r = fn(_ctx(curr_y=180))  # still floor 0
    assert r == 0.0


def test_floor_novelty_pays_per_new_floor():
    fn = create("fruit_bonus_floor_novelty", {"scale": 0.01, "novelty_bonus": 2.0})
    # Visit four different floors; each pays once.
    r0 = fn(_ctx(curr_y=182))  # floor 0
    r1 = fn(_ctx(curr_y=150))  # floor 1
    r2 = fn(_ctx(curr_y=118))  # floor 2
    r3 = fn(_ctx(curr_y=86))  # floor 3
    # Re-visit should pay nothing.
    r_repeat = fn(_ctx(curr_y=150))
    assert (r0, r1, r2, r3, r_repeat) == (2.0, 2.0, 2.0, 2.0, 0.0)


def test_floor_novelty_ignores_death_animation_region():
    """Floor bucket >= 4 (y < 72) is the death-animation region;
    don't count that as a legitimate "new floor" visit."""
    fn = create("fruit_bonus_floor_novelty", {"scale": 0.01, "novelty_bonus": 1.0})
    r = fn(_ctx(curr_y=24))  # floor bucket 5 (clamped away)
    assert r == 0.0
    r = fn(_ctx(curr_y=16))  # death animation
    assert r == 0.0


def test_floor_novelty_stacks_with_fruit_bonus():
    """When a fruit is collected AND a new floor is visited in the same
    step, both terms apply."""
    fn = create("fruit_bonus_floor_novelty", {"scale": 0.01, "novelty_bonus": 1.0})
    r = fn(_ctx(curr_y=118, prev_fruits=3, curr_fruits=2, curr_bonus=800))
    # fruit term = 1 fruit * 800 * 0.01 = 8.0, novelty = 1.0 -> 9.0
    assert r == pytest.approx(9.0)


def test_floor_novelty_reset_clears_visited_floors():
    """reset_reward must clear per-episode state, else novelty fires
    on only the first episode's floors."""
    fn = create("fruit_bonus_floor_novelty", {"scale": 0.01, "novelty_bonus": 1.0})
    r = fn(_ctx(curr_y=182))
    assert r == 1.0  # first visit
    r = fn(_ctx(curr_y=182))
    assert r == 0.0  # already visited
    rewards.reset_reward(fn)
    r = fn(_ctx(curr_y=182))
    assert r == 1.0  # reset -> fresh visit


def test_reset_reward_is_noop_for_stateless_rewards():
    """Existing reward formulas don't define reset() — reset_reward
    must not crash on them."""
    fn = create("fruit_bonus", {"scale": 0.01})
    rewards.reset_reward(fn)  # should not raise


def test_floor_novelty_registered():
    assert "fruit_bonus_floor_novelty" in available()


# ---------------------------------------------------------------------------
# fruit_bonus_climb_novelty
# ---------------------------------------------------------------------------


def _ctx_cy(y, fruits_rem=2, fp=None, collected_this_step=False, curr_bonus=800):
    """Shortcut: RewardContext with curr_y and fruits_present set."""
    prev_fruits = fruits_rem + (1 if collected_this_step else 0)
    return _ctx(
        prev_fruits=prev_fruits,
        curr_fruits=fruits_rem,
        curr_bonus=curr_bonus,
        prev_bonus=curr_bonus,
        curr_y=y,
        fruits_present=(
            fp if fp is not None else (True,) * fruits_rem + (False,) * (4 - fruits_rem)
        ),
    )


def test_climb_novelty_pays_on_first_climb():
    fn = create("fruit_bonus_climb_novelty", {"scale": 0.01, "climb_bonus": 2.0})
    # Start at spawn (floor 0, y=182), fruit 3 still present (above).
    fn(_ctx_cy(y=182, fruits_rem=2, fp=(False, False, True, True)))
    # Climb to floor 1 (y=150) — first climb, fruit 3 above us.
    r = fn(_ctx_cy(y=150, fruits_rem=2, fp=(False, False, True, True)))
    assert r == pytest.approx(2.0)


def test_climb_novelty_one_shot_per_floor():
    fn = create("fruit_bonus_climb_novelty", {"scale": 0.01, "climb_bonus": 2.0})
    fn(_ctx_cy(y=182, fruits_rem=2, fp=(False, False, True, True)))
    r1 = fn(_ctx_cy(y=150, fruits_rem=2, fp=(False, False, True, True)))
    # Re-visit floor 1 after going back up doesn't pay again.
    fn(_ctx_cy(y=182, fruits_rem=2, fp=(False, False, True, True)))
    r2 = fn(_ctx_cy(y=150, fruits_rem=2, fp=(False, False, True, True)))
    assert r1 == pytest.approx(2.0)
    assert r2 == 0.0


def test_climb_novelty_pays_per_floor_climbed():
    fn = create("fruit_bonus_climb_novelty", {"scale": 0.01, "climb_bonus": 2.0})
    fp = (False, False, True, True)  # fruits 3 and 4 remain (above spawn)
    fn(_ctx_cy(y=182, fruits_rem=2, fp=fp))  # init at floor 0
    r1 = fn(_ctx_cy(y=150, fruits_rem=2, fp=fp))  # floor 1
    r2 = fn(_ctx_cy(y=118, fruits_rem=2, fp=fp))  # floor 2
    r3 = fn(_ctx_cy(y=86, fruits_rem=2, fp=fp))  # floor 3
    assert (r1, r2, r3) == (2.0, 2.0, 2.0)


def test_climb_novelty_descent_pays_zero():
    fn = create("fruit_bonus_climb_novelty", {"scale": 0.01, "climb_bonus": 2.0})
    fp = (True, True, False, False)
    fn(_ctx_cy(y=86, fruits_rem=2, fp=fp))  # init at top
    r = fn(_ctx_cy(y=150, fruits_rem=2, fp=fp))  # descend to floor 1
    assert r == 0.0


def test_climb_novelty_no_fruit_above_no_reward():
    """Remaining fruit is below the agent — don't reward climbing."""
    fn = create("fruit_bonus_climb_novelty", {"scale": 0.01, "climb_bonus": 2.0})
    # Agent on floor 2 (y=118), only fruit 1 remains (y=184, below).
    fp = (True, False, False, False)
    fn(_ctx_cy(y=118, fruits_rem=1, fp=fp))
    r = fn(_ctx_cy(y=86, fruits_rem=1, fp=fp))  # climb to top, but target is BELOW
    assert r == 0.0


def test_climb_novelty_jumping_in_place_no_reward():
    """Brief upward bounce without crossing a floor boundary pays zero."""
    fn = create("fruit_bonus_climb_novelty", {"scale": 0.01, "climb_bonus": 2.0})
    fp = (False, False, True, True)
    fn(_ctx_cy(y=182, fruits_rem=2, fp=fp))
    # Jump +10px (to y=172) then fall back.
    r1 = fn(_ctx_cy(y=172, fruits_rem=2, fp=fp))
    r2 = fn(_ctx_cy(y=182, fruits_rem=2, fp=fp))
    # Both still floor 0 (bucket = (200-172)//32 = 0), no crossing.
    assert r1 == 0.0
    assert r2 == 0.0


def test_climb_novelty_stacks_with_fruit_term():
    fn = create("fruit_bonus_climb_novelty", {"scale": 0.01, "climb_bonus": 2.0})
    fp = (False, False, True, True)
    fn(_ctx_cy(y=182, fruits_rem=2, fp=fp))  # init
    # Climb to floor 2 AND collect a fruit (curr_fruits < prev_fruits).
    ctx = _ctx(
        prev_fruits=2,
        curr_fruits=1,
        curr_bonus=800,
        prev_bonus=800,
        curr_y=118,
        fruits_present=(False, False, True, True),
    )
    # Skips the intermediate floor, but one floor transition still pays.
    r = fn(ctx)
    # fruit term: 1 * 800 * 0.01 = 8.0; climb: 2.0 -> 10.0
    assert r == pytest.approx(10.0)


def test_climb_novelty_reset_clears_best_floor():
    fn = create("fruit_bonus_climb_novelty", {"scale": 0.01, "climb_bonus": 2.0})
    fp = (False, False, True, True)
    fn(_ctx_cy(y=182, fruits_rem=2, fp=fp))  # init on floor 0
    r = fn(_ctx_cy(y=150, fruits_rem=2, fp=fp))  # climb -> 2.0
    assert r == 2.0
    # Without reset: returning to floor 0 and climbing again pays nothing.
    fn(_ctx_cy(y=182, fruits_rem=2, fp=fp))
    r = fn(_ctx_cy(y=150, fruits_rem=2, fp=fp))
    assert r == 0.0
    # After reset: best_floor is cleared, climbing pays again.
    rewards.reset_reward(fn)
    fn(_ctx_cy(y=182, fruits_rem=2, fp=fp))  # re-initialise
    r = fn(_ctx_cy(y=150, fruits_rem=2, fp=fp))
    assert r == 2.0


def test_climb_novelty_without_fruits_present_falls_back_to_count():
    """When fruits_present is unknown, fall back to 'any fruit => reward climbs'."""
    fn = create("fruit_bonus_climb_novelty", {"scale": 0.01, "climb_bonus": 2.0})
    # No fruits_present provided (empty tuple default).
    fn(_ctx_cy(y=182, fruits_rem=2, fp=()))
    r = fn(_ctx_cy(y=150, fruits_rem=2, fp=()))
    assert r == 2.0


def test_climb_novelty_registered():
    assert "fruit_bonus_climb_novelty" in available()


# ---------------------------------------------------------------------------
# fruit_bonus_path_progress
# ---------------------------------------------------------------------------


def _pc(
    x=0,
    y=184,
    fp=(True, True, True, True),
    fruits_rem=4,
    prev_fruits=None,
    curr_bonus=800,
):
    """Shortcut for path-progress context with agent at (x, y)."""
    if prev_fruits is None:
        prev_fruits = fruits_rem
    return _ctx(
        prev_fruits=prev_fruits,
        curr_fruits=fruits_rem,
        curr_bonus=curr_bonus,
        prev_bonus=curr_bonus,
        curr_x=x,
        curr_y=y,
        fruits_present=fp,
    )


def test_path_progress_registered():
    assert "fruit_bonus_path_progress" in available()


def test_path_progress_first_step_no_reward():
    """First step initialises per-fruit best_d but doesn't pay."""
    fn = create("fruit_bonus_path_progress", {"scale": 0.01})
    r = fn(_pc(x=0, y=184))
    assert r == 0.0


def test_path_progress_rewards_approach_to_nearest_fruit():
    """Walking toward F1 pays progress to F1 AND to any other fruit
    whose path distance also dropped (since horizontal walk can
    reduce distance to multiple fruits via the same ladder route)."""
    fn = create("fruit_bonus_path_progress", {"scale": 0.01})
    fn(_pc(x=0, y=184))
    # Moving right 20 ram (80 pixels) closer along floor 1.
    r = fn(_pc(x=20, y=184))
    # Floor-1 approach reduces distance to F1 by exactly 80 px, and
    # since every path to F2 via L12a/L12b also traverses floor 1,
    # F2's distance drops too. Progress to F1 alone = 0.80.
    assert r >= 0.80 - 1e-6


def test_path_progress_oscillation_ratchets_then_zeros_out():
    """Walking back and forth ratchets each fruit's best_d once, then
    pays zero on round-trips."""
    fn = create("fruit_bonus_path_progress", {"scale": 0.01})
    # Seed init at floor 1, ram_x=0.
    fn(_pc(x=0, y=184))
    # Walk right ram_x=0 -> 20 (first approach pays).
    r1 = fn(_pc(x=20, y=184))
    # Back to ram_x=0 (retreat, pays zero).
    r_back = fn(_pc(x=0, y=184))
    # Return to ram_x=20 (same-as-best, pays zero).
    r_return = fn(_pc(x=20, y=184))
    assert r1 > 0
    assert r_back == 0.0
    assert r_return == 0.0


def test_path_progress_jumping_pays_zero():
    """Jumping doesn't change pixel x, so path distance doesn't drop."""
    fn = create("fruit_bonus_path_progress", {"scale": 0.01})
    fn(_pc(x=20, y=184))
    r_mid_jump = fn(_pc(x=20, y=168))  # mid-air
    r_back = fn(_pc(x=20, y=184))  # landed
    assert r_mid_jump == 0.0
    assert r_back == 0.0


def test_path_progress_mid_air_uses_last_known_floor():
    """When agent is mid-jump (y between floors), the reward reuses
    the last-known floor instead of skipping shaping entirely."""
    fn = create("fruit_bonus_path_progress", {"scale": 0.01})
    fn(_pc(x=0, y=184))
    r = fn(_pc(x=20, y=170))  # mid-air during horizontal move
    assert r > 0


def test_path_progress_clears_picked_fruit_tracking():
    """After a fruit is picked, its best_d is cleared."""
    fn = create("fruit_bonus_path_progress", {"scale": 0.01})
    fn(_pc(x=0, y=184, fp=(True, True, True, True), fruits_rem=4))
    fn(_pc(x=46, y=184, fp=(True, True, True, True), fruits_rem=4))
    r_pick = fn(
        _pc(
            x=46,
            y=184,
            fp=(False, True, True, True),
            fruits_rem=3,
            prev_fruits=4,
        )
    )
    # Pickup term = 1 * 800 * 0.01 = 8.0 plus any residual progress.
    assert r_pick >= 8.0


def test_path_progress_reset_clears_state():
    """reset_reward must clear best_d dict and last_floor."""
    fn = create("fruit_bonus_path_progress", {"scale": 0.01})
    fn(_pc(x=0, y=184))
    r1 = fn(_pc(x=20, y=184))
    assert r1 > 0
    rewards.reset_reward(fn)
    # After reset, next approach should once again pay.
    fn(_pc(x=0, y=184))
    r_after_reset = fn(_pc(x=20, y=184))
    assert r_after_reset > 0


def test_path_progress_universal_rebaselines_best_d_on_pickup():
    """Regression for the F2->F3 reward leak (approach 33).

    A fruit pickup must re-baseline best_d for the remaining fruits at
    the new position, so the next leg gets a fresh full-distance
    progress budget. Without the fix, best_d[F3] holds the closest the
    agent ever drifted to F3 during earlier travel (e.g. passing the
    L23 ladder on the way to F2), so the actual F2->F3 leg pays nothing
    until it beats that leaked-low value.
    """
    fn = create("fruit_bonus_path_progress_universal", {"scale": 0.01})
    # Floor 2 (y=152). F2 (ram x~20) and F3 present.
    f2f3 = (False, True, True, False)
    # Step 0: at F2 (ram x=20) -> baseline best_d[F3] at d3~280.
    fn(_pc(x=20, y=152, fp=f2f3, fruits_rem=2))
    # Step 1: walk right to/past L23 (ram x=60 = 248px) -> ratchets
    # best_d[F3] down to ~136 (the leak source).
    r1 = fn(_pc(x=60, y=152, fp=f2f3, fruits_rem=2))
    assert r1 > 0
    # Step 2: back to F2 (ram x=20) and COLLECT F2 -> only F3 remains.
    only_f3 = (False, False, True, False)
    fn(_pc(x=20, y=152, fp=only_f3, fruits_rem=1, prev_fruits=2))
    # Step 3: approach F3 (ram x=40 = 168px, d3~200). With the fix,
    # best_d[F3] was re-baselined at x=20 on pickup (d3~280), so this
    # closer step pays. Without the fix best_d[F3] would still be ~136
    # and this would pay zero.
    r3 = fn(_pc(x=40, y=152, fp=only_f3, fruits_rem=1))
    assert r3 > 0


# ---------------------------------------------------------------------------
# fruit_bonus_path_progress_universal
# ---------------------------------------------------------------------------


def _puc(
    x=0,
    y=184,
    fp=(True, True, True, True),
    fruits_rem=4,
    prev_fruits=None,
    curr_bonus=800,
    prev_bonus=None,
    curr_lives=5,
    prev_lives=5,
    princess_touched=False,
):
    """Shortcut for universal-path-progress context."""
    if prev_fruits is None:
        prev_fruits = fruits_rem
    if prev_bonus is None:
        prev_bonus = curr_bonus
    return _ctx(
        prev_fruits=prev_fruits,
        curr_fruits=fruits_rem,
        curr_bonus=curr_bonus,
        prev_bonus=prev_bonus,
        prev_lives=prev_lives,
        curr_lives=curr_lives,
        curr_x=x,
        curr_y=y,
        fruits_present=fp,
        princess_touched=princess_touched,
    )


def test_universal_registered():
    assert "fruit_bonus_path_progress_universal" in available()


def test_universal_first_step_no_reward():
    fn = create("fruit_bonus_path_progress_universal", {"scale": 0.01})
    r = fn(_puc(x=0, y=184))
    assert r == 0.0


def test_universal_fruit_progress_when_fruits_remain():
    """Same fruit-progress behaviour as path_progress."""
    fn = create("fruit_bonus_path_progress_universal", {"scale": 0.01})
    fn(_puc(x=0, y=184))
    r = fn(_puc(x=20, y=184))
    assert r > 0


def test_universal_princess_progress_when_no_fruits_remain():
    """When all fruits are collected, target = princess."""
    fn = create(
        "fruit_bonus_path_progress_universal",
        {"scale": 0.01, "princess_scale": 0.05},
    )
    # Init: agent at (ram_x=20, y=88) on floor 4, all fruits collected.
    fp_done = (False, False, False, False)
    fn(_puc(x=20, y=88, fp=fp_done, fruits_rem=0))
    # Walk right toward L45 (closer to princess in path-distance).
    r = fn(_puc(x=40, y=88, fp=fp_done, fruits_rem=0))
    assert r > 0


def test_universal_no_princess_progress_while_fruits_remain():
    """Fruit-progress fires, princess is ignored."""
    fn = create(
        "fruit_bonus_path_progress_universal",
        {"scale": 0.01, "princess_scale": 0.05},
    )
    fp = (False, False, False, True)  # only F4 remains
    fn(_puc(x=0, y=184, fp=fp, fruits_rem=1))
    # Princess best_d should remain None (not initialised) because we
    # never targeted it.
    assert fn.best_d_princess is None


def test_universal_princess_touch_pays_one_shot():
    """A princess touch event (caller flagged the rising edge of the
    level-cleared flag) pays prev_bonus * princess_scale."""
    fn = create(
        "fruit_bonus_path_progress_universal",
        {"scale": 0.01, "princess_scale": 0.05},
    )
    # Init at CP4 (no fruits remaining).
    fp_done = (False, False, False, False)
    fn(_puc(x=70, y=56, fp=fp_done, fruits_rem=0))
    # Princess touch: caller sets princess_touched=True. Note that on
    # this exact frame the fruits/bonus haven't changed yet — the
    # game keeps fruits=0 and bonus near its current value at the
    # touch frame, then resets ~370 frames later.
    r = fn(
        _puc(
            x=70,
            y=56,
            fp=fp_done,
            fruits_rem=0,
            prev_fruits=0,
            curr_bonus=400,
            prev_bonus=400,
            princess_touched=True,
        )
    )
    # princess term = 400 * 0.05 = 20.0
    assert r >= 20.0 - 1e-6


def test_universal_death_respawn_does_not_count_as_princess():
    """Without ``princess_touched=True`` the reward must not pay the
    princess term, even if fruits go from 0 -> 4 (death respawn)."""
    fn = create(
        "fruit_bonus_path_progress_universal",
        {"scale": 0.01, "princess_scale": 0.05},
    )
    fp_done = (False, False, False, False)
    fn(_puc(x=70, y=56, fp=fp_done, fruits_rem=0))
    fp_post = (True, True, True, True)
    r = fn(
        _puc(
            x=0,
            y=182,
            fp=fp_post,
            fruits_rem=4,
            prev_fruits=0,
            curr_bonus=1000,
            prev_bonus=400,
            curr_lives=4,
            prev_lives=5,
            princess_touched=False,
        )
    )
    # No princess flag -> no princess reward.
    assert r == 0.0


def test_universal_princess_touch_clears_best_d():
    """After a princess touch, best_d should reset for both fruits
    and princess (game just respawned the level)."""
    fn = create(
        "fruit_bonus_path_progress_universal",
        {"scale": 0.01, "princess_scale": 0.05},
    )
    # Build up some best_d state on fruits.
    fn(_puc(x=0, y=184))
    fn(_puc(x=20, y=184))
    # Trigger a princess touch.
    fn(
        _puc(
            x=0,
            y=184,
            fp=(True, True, True, True),
            fruits_rem=4,
            prev_fruits=0,
            curr_bonus=400,
            prev_bonus=400,
            princess_touched=True,
        )
    )
    # All best_d entries should be None or freshly set this step.
    # We test it by checking values directly: anything that wasn't
    # touched this step should be None.
    bd = fn.best_d
    # Fruit 1 is at floor 1 same as agent — got initialised this step.
    assert bd[1] is not None
    # The previous (smaller best_d[1] from before the touch) was
    # cleared. Verify by checking it's at the freshly-computed
    # distance for x=0, not the smaller one we'd seen at x=20.
    # At x=0 floor=1, agent centre pix = 0*4+8 = 8. F1 centre = 184.
    # |8-184| = 176.
    assert bd[1] == 176


def test_universal_reset_clears_state():
    fn = create(
        "fruit_bonus_path_progress_universal",
        {"scale": 0.01, "princess_scale": 0.05},
    )
    fn(_puc(x=0, y=184))
    fn(_puc(x=20, y=184))
    rewards.reset_reward(fn)
    assert fn.best_d_princess is None
    assert fn.last_floor is None
    assert all(v is None for v in fn.best_d.values())
