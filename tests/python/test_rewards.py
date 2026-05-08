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
