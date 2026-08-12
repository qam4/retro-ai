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
# fruit_bonus_path_progress_pbrs (Markovian PBRS shaping)
# ---------------------------------------------------------------------------


def test_pbrs_registered():
    assert "fruit_bonus_path_progress_pbrs" in available()


def test_pbrs_first_step_no_shaping():
    """First step has no previous potential, so it pays no shaping."""
    fn = create("fruit_bonus_path_progress_pbrs", {"scale": 0.01, "gamma": 1.0})
    assert fn(_pc(x=0, y=184)) == 0.0


def test_pbrs_rewards_approach_and_penalizes_retreat():
    """Unlike the ratchet, PBRS is symmetric: moving toward pays +,
    moving away pays - (so round trips cancel)."""
    fn = create("fruit_bonus_path_progress_pbrs", {"scale": 0.01, "gamma": 1.0})
    fn(_pc(x=0, y=184))  # baseline
    r_toward = fn(_pc(x=20, y=184))  # closer to F1
    r_away = fn(_pc(x=0, y=184))  # back to start
    assert r_toward > 0
    assert r_away < 0
    # With gamma=1 a round trip telescopes to ~0 net.
    assert abs(r_toward + r_away) < 1e-6


def test_pbrs_is_markovian_no_history_dependence():
    """Same transition (s->s') yields the same shaping regardless of the
    path taken to reach s. This is the property the best_d ratchet
    violated."""
    a = create("fruit_bonus_path_progress_pbrs", {"scale": 0.01, "gamma": 1.0})
    b = create("fruit_bonus_path_progress_pbrs", {"scale": 0.01, "gamma": 1.0})

    # Instance A reaches (x=20) directly from x=0.
    a(_pc(x=0, y=184))
    a(_pc(x=20, y=184))
    r_a = a(_pc(x=40, y=184))

    # Instance B reaches (x=20) after first wandering out to x=40 and
    # back (which would have ratcheted best_d in the legacy reward).
    b(_pc(x=0, y=184))
    b(_pc(x=40, y=184))
    b(_pc(x=20, y=184))
    r_b = b(_pc(x=40, y=184))

    # The 20->40 step pays the same for both despite different history.
    assert abs(r_a - r_b) < 1e-6


def test_pbrs_pickup_rebaselines_no_shaping_spike():
    """On a pickup the remaining-target set changes; that step pays the
    sparse fruit term and re-baselines the potential (no shaping spike
    from a vanishing distance term)."""
    fn = create("fruit_bonus_path_progress_pbrs", {"scale": 0.01, "gamma": 1.0})
    fn(_pc(x=0, y=184, fp=(True, True, True, True), fruits_rem=4))
    fn(_pc(x=44, y=184, fp=(True, True, True, True), fruits_rem=4))
    r_pick = fn(
        _pc(x=46, y=184, fp=(False, True, True, True), fruits_rem=3, prev_fruits=4)
    )
    # Sparse pickup term = 1 * 800 * 0.01 = 8.0, and the shaping is
    # skipped on the pickup step, so reward is exactly the pickup term.
    assert abs(r_pick - 8.0) < 1e-6


def test_pbrs_reset_clears_state():
    fn = create("fruit_bonus_path_progress_pbrs", {"scale": 0.01, "gamma": 1.0})
    fn(_pc(x=0, y=184))
    assert fn(_pc(x=20, y=184)) > 0
    rewards.reset_reward(fn)
    # After reset the next step is again a no-shaping baseline.
    assert fn(_pc(x=20, y=184)) == 0.0


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


# ---------------------------------------------------------------------------
# fruit_bonus_path_progress_pbrs_grounded (level-2 airborne-freeze + death gate)
#
# Regression guards for the v5 loiter-farm bug (experiments/003 H-AH): the
# earlier version returned Phi=None while airborne, which rebaselined prev_phi
# and DELETED the return-leg debt -> an approach-then-jump-back round trip
# banked free reward. The fix holds prev_phi across airborne (restoring PBRS
# telescoping) and gates credit on aliveness (0x2AFC via ctx.died).
# Level-2 floors: floor 1 at y=30, floor 2 at y=54. Poses: 0-5,8 = surface
# (grounded/ladder); 9/10 = jump, 11 = fall (airborne).
# ---------------------------------------------------------------------------

_L2 = {
    "scale": 0.01,
    "fruit_scale": 0.01,
    "princess_scale": 0.05,
    "gamma": 1.0,
    "level": 2,
}


def _g2(x, y, pose, died=False):
    """Level-2 grounded-reward context at (x, y) with a sprite pose."""
    return _ctx(
        prev_fruits=2,
        curr_fruits=2,
        prev_bonus=1000,
        curr_bonus=1000,
        curr_x=x,
        curr_y=y,
        fruits_present=(True, True),
        pose=pose,
        died=died,
    )


def _run_g2(traj):
    fn = create("fruit_bonus_path_progress_pbrs_grounded", _L2)
    fn.reset()
    return sum(fn(_g2(*step)) for step in traj)


def test_grounded_registered():
    assert "fruit_bonus_path_progress_pbrs_grounded" in available()


def test_grounded_grounded_roundtrip_nets_zero():
    """All-grounded walk out and back telescopes to 0 (PBRS invariant)."""
    net = _run_g2([(1, 30, 0), (3, 30, 0), (5, 30, 0), (3, 30, 0), (1, 30, 0)])
    assert abs(net) < 1e-9


def test_grounded_jumpback_farm_nets_zero():
    """THE v5 BUG: approach grounded, then jump back to the SAME floor.
    Holding prev_phi across airborne charges the retreat on landing, so the
    round trip nets 0 (no free reward to farm)."""
    net = _run_g2(
        [
            (1, 30, 0),
            (3, 30, 0),
            (5, 30, 0),
            (7, 30, 0),  # banked +0.48 approaching
            (7, 28, 9),  # jump (airborne)
            (5, 26, 9),
            (3, 28, 11),
            (1, 30, 0),  # land back at spawn -> retreat charged
        ]
    )
    assert abs(net) < 1e-9


def test_grounded_fatal_fall_not_credited():
    """Fall to a deeper floor that ends in death (ctx.died on landing) is
    not credited -- the death gate suppresses the potential jump."""
    net = _run_g2(
        [
            (7, 30, 0),
            (7, 38, 11),  # falling
            (7, 46, 11),
            (7, 54, 0, True),  # lands on floor 2 but DEAD
        ]
    )
    assert abs(net) < 1e-9


def test_grounded_survived_descent_is_credited():
    """A descent the agent survives (alive on the grounded landing) IS real
    progress and is credited. Cannot be farmed: climbing back up is grounded
    and charged symmetrically at gamma=1."""
    net = _run_g2([(7, 30, 0), (7, 38, 11), (7, 46, 11), (7, 54, 0, False)])
    assert net > 0


def test_grounded_gap_cross_is_credited():
    """A jump that lands on the SAME floor further along (a gap crossing) is
    credited -- this is the level-2 skill we want to reinforce."""
    net = _run_g2([(3, 30, 0), (4, 28, 9), (6, 28, 9), (8, 30, 0, False)])
    assert net > 0


def test_grounded_ladder_descent_is_credited():
    """A ladder descent (grounded pose 8 throughout) is credited continuously
    -- it is the intended way down, not a fall."""
    net = _run_g2([(7, 30, 8), (7, 38, 8), (7, 46, 8), (7, 54, 8)])
    assert net > 0


def test_grounded_reset_clears_state():
    fn = create("fruit_bonus_path_progress_pbrs_grounded", _L2)
    fn.reset()
    fn(_g2(1, 30, 0))
    assert fn(_g2(3, 30, 0)) > 0
    rewards.reset_reward(fn)
    assert fn.last_floor is None
    # After reset the next step is again a no-shaping baseline.
    assert fn(_g2(3, 30, 0)) == 0.0


# ---------------------------------------------------------------------------
# (D4) defer_fruit_credit — the sparse fruit reward is paid on the next
# grounded-ALIVE frame, so a fruit grabbed mid-air that falls to death pays 0
# (the measured L2 fruit-2 grab-and-fall). Default off = credit at pickup.
# Fixed position -> shaping is 0/rebaselined each step, isolating the fruit
# term (1 fruit * curr_bonus 1000 * fruit_scale 0.01 = 10.0).
# ---------------------------------------------------------------------------

_L2_DEFER = {**_L2, "defer_fruit_credit": True}


def _g2f(x, y, pose, pf, cf, died=False, bonus=1000):
    """L2 grounded ctx allowing fruit-count control (pf=prev, cf=curr)."""
    fp = {2: (True, True), 1: (True, False), 0: (False, False)}[cf]
    return _ctx(
        prev_fruits=pf,
        curr_fruits=cf,
        prev_bonus=bonus,
        curr_bonus=bonus,
        curr_x=x,
        curr_y=y,
        fruits_present=fp,
        pose=pose,
        died=died,
    )


def _run_params(params, traj):
    fn = create("fruit_bonus_path_progress_pbrs_grounded", params)
    fn.reset()
    return sum(fn(_g2f(*s)) for s in traj)


# grab fruit 2 while falling (pose 11), then land DEAD -> fatal grab.
_FATAL_GRAB = [
    (7, 30, 0, 2, 2),  # grounded baseline
    (7, 28, 9, 2, 2),  # jump
    (7, 32, 11, 2, 1),  # grabbed fruit mid-fall
    (7, 40, 11, 1, 1),  # still falling
    (7, 54, 0, 1, 1, True),  # lands DEAD
]
# grab airborne (pose 9), then land ALIVE -> survivable grab.
_SURV_GRAB = [
    (7, 30, 0, 2, 2),
    (7, 28, 9, 2, 2),
    (7, 26, 9, 2, 1),  # grabbed airborne
    (7, 30, 0, 1, 1),  # lands alive
]


def test_defer_fatal_airborne_grab_pays_zero():
    assert _run_params(_L2_DEFER, _FATAL_GRAB) == pytest.approx(0.0, abs=1e-9)


def test_default_credits_fatal_grab():
    # Documents the bug the flag fixes: default pays the fruit at the grab
    # even though the agent dies in the ensuing fall.
    assert _run_params(_L2, _FATAL_GRAB) == pytest.approx(10.0)


def test_defer_survived_grab_credited_on_landing():
    assert _run_params(_L2_DEFER, _SURV_GRAB) == pytest.approx(10.0)


def test_survived_grab_credited_in_both_modes():
    # Both credit a SURVIVED grab (defer just moves the fruit term to the
    # grounded landing). They may differ by shaping bookkeeping — default
    # credits an extra potential jump because an airborne pickup skips the
    # rebaseline, whereas defer rebaselines on the deferred credit — but both
    # pay at least the fruit value. The point is: only the FATAL grab differs
    # (0 vs credited), not the survived one.
    assert _run_params(_L2_DEFER, _SURV_GRAB) >= 10.0 - 1e-9
    assert _run_params(_L2, _SURV_GRAB) >= 10.0 - 1e-9


def test_defer_grounded_grab_credited_same_step():
    # A grounded grab (never airborne) is credited immediately, like default.
    assert _run_params(
        _L2_DEFER, [(7, 30, 0, 2, 2), (7, 30, 0, 2, 1)]
    ) == pytest.approx(10.0)


def test_defer_preserves_shaping_roundtrip_zero():
    # Shaping invariant intact with defer on (no fruit): grounded round trip
    # telescopes to 0.
    traj = [
        (1, 30, 0, 2, 2),
        (3, 30, 0, 2, 2),
        (5, 30, 0, 2, 2),
        (3, 30, 0, 2, 2),
        (1, 30, 0, 2, 2),
    ]
    assert abs(_run_params(_L2_DEFER, traj)) < 1e-9


# ---------------------------------------------------------------------------
# L3 mandatory-waypoint reward targets (LevelMap.reward_waypoints). The
# path-progress potential sums distance to remaining fruits AND not-yet-reached
# mandatory waypoint groups (unordered), min over an OR-group's members,
# dropping any group currently unreachable in the nav graph (the escalator /
# final-jump gaps). Reaching any member of a group (grounded, alive, within
# tol) marks that group done. L1/L2 have no reward_waypoints, so all of this
# is inert there (covered by the byte-identical grounded tests above).
#
# L3 WP-group member positions (ram_x, y), from the nav graph:
#   group 0 goat (OR): Lgoat_a_top (16, 80), Lgoat_b_top (22, 80)
#   group 1 Ldown_bot (40, 184)  [post-escalator; unreachable pre-escalator]
#   ... climb tops ... group 6 Lprincess_top (4, 48)
# ---------------------------------------------------------------------------

_L3 = {
    "scale": 0.01,
    "fruit_scale": 0.01,
    "princess_scale": 0.05,
    "gamma": 1.0,
    "level": 3,
    "defer_fruit_credit": True,
    "waypoint_reward_tol": 2,
}


def _g3(x, y, pose, died=False, pf=1, cf=1):
    """Level-3 grounded-reward context at (x, y) with a sprite pose."""
    return _ctx(
        prev_fruits=pf,
        curr_fruits=cf,
        prev_bonus=1000,
        curr_bonus=1000,
        curr_x=x,
        curr_y=y,
        fruits_present=(True,),
        pose=pose,
        died=died,
    )


# L3 goat platform (GOAT, y94, ram 18-27) is the OR-group reached via either
# goat-ladder top: Lgoat_a_top (ram18, y94) / Lgoat_b_top (ram24, y94). ram21
# is on the goat platform but not within tol of either WP (a clean "on the
# platform, not at the waypoint" spot).


def test_l3_wp_orgroup_reached_via_either_member():
    """The goat platform is one OR-group reached via EITHER ladder top."""
    for x in (18, 24):  # Lgoat_a_top / Lgoat_b_top
        fn = create("fruit_bonus_path_progress_pbrs_grounded", _L3)
        fn.reset()
        fn(_g3(x, 94, 0))
        assert 0 in fn._reached_wp


def test_l3_wp_not_marked_when_airborne():
    """A WP target is only marked on a surface pose (grounded/ladder)."""
    fn = create("fruit_bonus_path_progress_pbrs_grounded", _L3)
    fn.reset()
    fn(_g3(24, 94, 11))  # pose 11 = fall (airborne)
    assert 0 not in fn._reached_wp


def test_l3_wp_not_marked_on_death():
    """A death frame never marks a WP target (death gate)."""
    fn = create("fruit_bonus_path_progress_pbrs_grounded", _L3)
    fn.reset()
    fn(_g3(24, 94, 0, died=True))
    assert 0 not in fn._reached_wp


def test_l3_wp_reached_persists():
    """Once a group is marked reached it stays reached (a chokepoint the
    agent has passed drops out of the target sum for the rest of the ep)."""
    fn = create("fruit_bonus_path_progress_pbrs_grounded", _L3)
    fn.reset()
    fn(_g3(21, 94, 0))  # on goat platform, not at a WP
    fn(_g3(24, 94, 0))  # step onto the goat WP
    assert 0 in fn._reached_wp
    fn(_g3(21, 94, 0))  # move away (still on the platform)
    assert 0 in fn._reached_wp  # stays reached


def test_l3_wp_reached_when_active_rebaselines():
    """When the reached group WAS in the active (reachable) set, reaching it
    changes the active set and that step rebaselines (sparse-only reward).
    On the goat platform (floor 4) the goat group is reachable, so it is
    active before being reached."""
    fn = create("fruit_bonus_path_progress_pbrs_grounded", _L3)
    fn.reset()
    fn(_g3(21, 94, 0))  # on goat platform, goat group active, not at the WP
    r_reach = fn(_g3(24, 94, 0))  # reach Lgoat_b_top -> active set changes
    assert 0 in fn._reached_wp
    assert abs(r_reach) < 1e-9  # rebaselined: no shaping charged this step


def test_l3_unreachable_groups_dropped_reward_bounded():
    """Groups unreachable in the nav graph (post-escalator, across the gap)
    are dropped from the sum -- the 1e9 path sentinel never leaks into the
    reward, so shaping between adjacent grounded steps stays small."""
    fn = create("fruit_bonus_path_progress_pbrs_grounded", _L3)
    fn.reset()
    fn(_g3(21, 94, 0))
    r = fn(_g3(27, 94, 0))  # both on the goat platform, neither at a WP
    # A leaked 1e9 sentinel * scale(0.01) would be ~1e7; a real move is O(1).
    assert abs(r) < 1e3


def test_l3_reset_clears_wp_state():
    fn = create("fruit_bonus_path_progress_pbrs_grounded", _L3)
    fn(_g3(24, 94, 0))
    assert 0 in fn._reached_wp
    fn.reset()
    assert fn._reached_wp == set()
    assert fn._prev_active_wp == frozenset()


# --- Seeded-start milestone restore (the "paid to retreat" bug) -------------
# Milestone "reached" state is POSITIONAL, so unlike CP progress (fruit bytes in
# RAM) it does NOT survive a load_state: it lived only in the reward object and
# was wiped by reset(). A seeded episode therefore treated every milestone
# BEHIND it as pending and the potential paid it to RETREAT (measured on L3: the
# milestone-sum bottomed out at SN3, so a seed above SN3 gained ~+3.7 by
# climbing back down and a seed AT SN3 lost reward for leaving -> the SN3->A1
# hand-off measured 0%). restore_reached_waypoints() re-marks what the seed had
# already banked.


def test_restore_reached_waypoints_marks_groups():
    """Restoring a seed's ids marks the matching milestone groups reached."""
    fn = create("fruit_bonus_path_progress_pbrs_grounded", _L3)
    fn.reset()
    assert fn._reached_wp == set()
    rewards.restore_reached_waypoints(fn, {"Lgoat_a_top"})
    assert 0 in fn._reached_wp  # group 0 = the goat OR-group


def test_restore_reached_waypoints_is_noop_for_empty_and_unknown():
    fn = create("fruit_bonus_path_progress_pbrs_grounded", _L3)
    fn.reset()
    rewards.restore_reached_waypoints(fn, set())
    assert fn._reached_wp == set()
    rewards.restore_reached_waypoints(fn, {"not_a_waypoint"})
    assert fn._reached_wp == set()


def test_restore_reached_waypoints_noop_on_rewards_without_milestones():
    """Helper must be safe on formulas that have no milestones (all L1/L2)."""
    fn = create("fruit_bonus_path_progress_pbrs", dict(_L3, level=1))
    rewards.restore_reached_waypoints(fn, {"Lgoat_a_top"})  # must not raise


def test_restored_milestone_removes_backward_pull():
    """THE REGRESSION GUARD. Standing on the goat platform, moving AWAY from
    an unreached goat WP is charged (its distance grows). With that milestone
    restored as already-reached it drops out of the sum, so the same move is
    no longer penalised for the milestone term."""
    # Unrestored: the goat group is pending, so stepping away from it costs.
    fn = create("fruit_bonus_path_progress_pbrs_grounded", _L3)
    fn.reset()
    fn(_g3(24, 94, 0))  # AT Lgoat_b_top -> marks group 0 reached
    assert 0 in fn._reached_wp

    # A fresh episode seeded here WITHOUT the restore: group 0 pending again.
    fn2 = create("fruit_bonus_path_progress_pbrs_grounded", _L3)
    fn2.reset()
    fn2(_g3(21, 94, 0))  # on the platform, not at the WP
    r_pending = fn2(_g3(18, 94, 0))  # move onto Lgoat_a_top (reaches group 0)

    # Same seeded start WITH the restore: group 0 already banked.
    fn3 = create("fruit_bonus_path_progress_pbrs_grounded", _L3)
    fn3.reset()
    rewards.restore_reached_waypoints(fn3, {"Lgoat_a_top", "Lgoat_b_top"})
    fn3(_g3(21, 94, 0))
    r_restored = fn3(_g3(18, 94, 0))

    # The restored run must NOT re-mark the group (it is already reached) and
    # must not produce the reach-rebaseline the pending run does.
    assert 0 in fn3._reached_wp
    assert r_pending != r_restored


# --- (D5) unified credit rule: credit only for progress you SURVIVE ---------
# One rule for every target type. Previously only FRUITS were protected
# (defer_fruit_credit held the sparse grab until grounded-alive); milestones and
# path-progress banked credit for an arrival the agent died from. Measured on L3:
# "touch SN3 then die" paid +5.04 vs 0.00 for waiting, so arriving recklessly
# strictly dominated waiting for a safe phase — and the policy never learned to
# survive SN3 (it is no better than random there).

_L3_SURV = dict(
    scale=0.01,
    fruit_scale=0.01,
    princess_scale=0.05,
    level=3,
    gamma=1.0,
    credit_requires_survival=True,
    waypoint_reward_tol=2,
    ladder_segment_shaping=True,
)


def _seq(fn, steps):
    """Run (x, y[, pose, died]) steps, returning the summed reward."""
    return sum(fn(_g3(*s)) for s in steps)


def test_progress_then_death_pays_nothing():
    """A milestone arrival the agent dies from must net ~0 (was +5.04)."""
    fn = create("fruit_bonus_path_progress_pbrs_grounded", _L3_SURV)
    fn.reset()
    fn(_g3(58, 110, 8))  # baseline at SN2
    total = _seq(
        fn,
        [
            (62, 110, 0),
            (66, 110, 0),
            (70, 110, 0),
            (70, 102, 8),
            (70, 94, 8),
            (70, 86, 8),  # reaches the SN3 milestone
            (70, 86, 0, True),  # ...and dies
        ],
    )
    assert abs(total) < 1e-6, f"reckless arrival still pays {total:+.3f}"


def test_progress_then_survival_still_pays():
    """The fix must not flatten genuine progress: surviving the same climb and
    continuing must pay clearly MORE than dying on arrival (which nets ~0).
    Asserted comparatively — the absolute value depends on pose/segment detail."""
    fn = create("fruit_bonus_path_progress_pbrs_grounded", _L3_SURV)
    fn.reset()
    fn(_g3(58, 110, 8))
    total = _seq(
        fn,
        [
            (62, 110, 0),
            (66, 110, 0),
            (70, 110, 0),
            (70, 102, 8),
            (70, 94, 8),
            (70, 86, 8),
            (64, 86, 0),
            (56, 86, 0),
            (48, 86, 0),  # traverse on toward A1
        ],
    )
    assert total > 1.0, f"surviving progress should still pay, got {total:+.3f}"


def test_waiting_is_not_dominated_by_dying():
    """The behavioural point: patience must be at least as good as arrive-and-die."""

    def run(steps):
        fn = create("fruit_bonus_path_progress_pbrs_grounded", _L3_SURV)
        fn.reset()
        fn(_g3(58, 110, 8))
        return _seq(fn, steps)

    die = run(
        [
            (62, 110, 0),
            (66, 110, 0),
            (70, 110, 0),
            (70, 102, 8),
            (70, 94, 8),
            (70, 86, 8),
            (70, 86, 0, True),
        ]
    )
    wait = run([(58, 110, 8)] * 7)
    assert wait >= die - 1e-9, f"dying ({die:+.3f}) must not beat waiting ({wait:+.3f})"


def test_credit_rule_implies_fruit_deferral():
    """The flag subsumes defer_fruit_credit — they were the same idea applied to
    one target type, so a fatal airborne grab must not pay under the new flag
    either (regression guard for the L2 fix)."""
    params = dict(_L3_SURV)
    params.pop("credit_requires_survival")
    params["credit_requires_survival"] = True
    fn = create("fruit_bonus_path_progress_pbrs_grounded", params)
    fn.reset()
    fn(_g3(18, 62, 0, pf=1, cf=1))
    # grab the fruit while FALLING (pose 11) and die: must not bank the fruit
    r = fn(_g3(18, 62, 11, pf=1, cf=0))
    r += fn(_g3(18, 90, 11, pf=0, cf=0, died=True))
    assert r < 1.0, f"fatal airborne grab paid {r:+.3f}"


def test_death_refund_is_opt_in():
    """Default path is unchanged (golden tests also cover this)."""
    params = dict(_L3_SURV)
    params["credit_requires_survival"] = False
    params["defer_fruit_credit"] = True
    fn = create("fruit_bonus_path_progress_pbrs_grounded", params)
    fn.reset()
    fn(_g3(58, 110, 8))
    total = _seq(
        fn,
        [
            (62, 110, 0),
            (66, 110, 0),
            (70, 110, 0),
            (70, 102, 8),
            (70, 94, 8),
            (70, 86, 8),
            (70, 86, 0, True),
        ],
    )
    assert total > 1.0, "with the flag off, the old (banked) behaviour must remain"
