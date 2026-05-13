"""Tests for the reward tracer."""

from __future__ import annotations

import os
import pickle

import pytest
from retro_ai.training.reward_trace import (
    EpisodeTracer,
    compute_path_progress_bound,
)
from retro_ai.training.yeti_map import build_navigation_map


@pytest.fixture(scope="module")
def nav():
    return build_navigation_map()


def test_bound_zero_when_no_fruits_remain(nav):
    """No remaining fruits means no progress reward and no pickup
    reward — bound is 0."""
    b = compute_path_progress_bound(
        nav,
        agent_pix_x=0,
        agent_floor=1,
        fruits_present=(False, False, False, False),
        progress_scale=0.01,
        fruit_scale=0.01,
        initial_bonus=800,
    )
    assert b == 0.0


def test_bound_includes_progress_and_pickup(nav):
    """Bound = sum of (initial path distance * scale) + sum of
    (pickup payment for each remaining fruit at initial bonus)."""
    b = compute_path_progress_bound(
        nav,
        agent_pix_x=0,  # spawn x=0
        agent_floor=1,
        fruits_present=(True, False, False, True),  # F1 and F4 remain
        progress_scale=0.01,
        fruit_scale=0.01,
        initial_bonus=800,
    )
    # F1 distance from (1, 0) = 184; F4 distance = 496 (per draw script).
    # progress_bound = (184 + 496) * 0.01 = 6.80
    # pickup_bound = 2 * 800 * 0.01 = 16.0
    assert b == pytest.approx(6.80 + 16.0)


def test_tracer_does_not_dump_under_bound(tmp_path):
    """Episodes whose total stays under bound are not written."""
    tracer = EpisodeTracer(env_id=0, output_dir=str(tmp_path))
    tracer.reset(meta={"episode_id": 42})
    for s in range(1, 5):
        tracer.record_step(
            step=s,
            agent_x=10,
            agent_y=184,
            agent_floor=1,
            fruits_present=(True, True, True, True),
            action=[0, 0, 0],
            reward=0.5,
            done=False,
            truncated=False,
            reward_state={"best_d": {1: 100, 2: 200, 3: 300, 4: 400}},
        )
    out = tracer.finalize_and_maybe_dump(total_reward=2.0, bound=10.0)
    assert out is None
    assert not list(tmp_path.iterdir())


def test_tracer_dumps_over_bound(tmp_path):
    """Episodes that exceed the bound get pickled."""
    tracer = EpisodeTracer(env_id=2, output_dir=str(tmp_path))
    tracer.reset(meta={"episode_id": 9999, "start_state_hash": "abc"})
    tracer.record_step(
        step=1,
        agent_x=10,
        agent_y=184,
        agent_floor=1,
        fruits_present=(True, True, True, True),
        action=[0, 1, 0],
        reward=20.0,
        done=False,
        truncated=False,
        reward_state={"best_d": {1: 100, 2: 200, 3: 300, 4: 400}},
    )
    tracer.record_step(
        step=2,
        agent_x=11,
        agent_y=184,
        agent_floor=1,
        fruits_present=(True, True, True, True),
        action=[0, 1, 0],
        reward=30.0,
        done=True,
        truncated=False,
        reward_state={"best_d": {1: 80, 2: 200, 3: 300, 4: 400}},
    )
    out = tracer.finalize_and_maybe_dump(total_reward=50.0, bound=2.0)
    assert out is not None
    assert os.path.exists(out)

    with open(out, "rb") as f:
        payload = pickle.load(f)
    assert payload["meta"]["bound_exceeded"] is True
    assert payload["meta"]["total_reward"] == 50.0
    assert payload["meta"]["bound"] == 2.0
    assert payload["meta"]["episode_id"] == 9999
    assert len(payload["steps"]) == 2
    # Sanity check stored fields.
    s0 = payload["steps"][0]
    assert s0["agent_x"] == 10
    assert s0["reward"] == 20.0
    assert s0["fruits_present"] == (True, True, True, True)
