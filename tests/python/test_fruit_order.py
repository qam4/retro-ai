"""The evaluator records the ORDER fruits were taken in, not just how many.

On L1 the route is the open question: F1 -> F2 -> F3 -> F4 is the shortest tour (960
px), and the agent was believed to take F1 -> F3 -> F4 -> F2 (1472 px) because the
sum-of-distances potential pays for it. `reached_points` is a set and `max_cp` a count,
so until now no evaluation could say which route was taken.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from retro_ai.games.yeti_rollout import newly_collected

REPO = Path(__file__).resolve().parents[2]


def test_a_pickup_is_a_fruit_that_was_there_and_is_not():
    before = {1: True, 2: True, 3: False, 4: True}
    assert newly_collected(before, {1: True, 2: False, 3: False, 4: True}) == [2]


def test_nothing_taken_and_an_already_gone_fruit_are_not_pickups():
    gone = {1: False, 2: True}
    assert newly_collected(gone, gone) == []
    assert newly_collected(gone, {1: False, 2: True}) == []


def test_a_fruit_reappearing_is_not_a_pickup():
    assert newly_collected({1: False}, {1: True}) == []


def test_the_rollout_records_the_order_it_took_them_in():
    """One real L1 episode: the order lists each collected fruit once, and as many as
    the fruit counter says were collected."""
    model_path = REPO / (
        "output/mo5/yeti/training/yeti_curriculum_l1_v17_v30recipe_cold_10m/"
        "best/best_model.zip"
    )
    if not os.environ.get("RETRO_AI_ROM_DIR") or not model_path.is_file():
        pytest.skip("needs the ROMs and the local l1_v17 model")
    from retro_ai.games.yeti_rollout import rollout_episode
    from retro_ai.training.env_builder import build_training_env
    from retro_ai.training.run_config import EnvConfig
    from stable_baselines3 import PPO

    cfg = EnvConfig(
        profile="yeti_fruit",
        action_mode="joystick",
        max_steps=1500,
        stall_threshold=40,
        resize=(84, 84),
        resize_mode="max",
    )
    stack = build_training_env("yeti_fruit", cfg)
    model = PPO.load(str(model_path), device="cpu")
    r = rollout_episode(
        stack, model, level=1, fruits_total=4, max_steps=1500, stall_threshold=40
    )
    assert r.fruits_collected > 0, "the champion should take at least one fruit"
    assert len(r.fruit_order) == r.fruits_collected
    assert len(set(r.fruit_order)) == len(r.fruit_order)
    assert set(r.fruit_order) <= {"F1", "F2", "F3", "F4"}
