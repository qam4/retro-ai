"""Integration test: load_state resets per-episode death trackers.

Regression test for a bug where ``bonus_stall_count_`` and
``previous_bonus_`` on the C++ MO5 reward wrapper carried over across
``load_state`` calls. That caused the first few steps after a load to
spuriously report ``done=True`` — e.g. training envs loading a
checkpoint would see an immediate episode end.

The fix is in ``src/mo5_rl.cpp``: ``load_state`` now re-initialises the
lives/bonus/height/fruit trackers from the just-loaded memory, just
like ``reset`` does.

This test skips if the native build is not present (CI-less dev
machines running the Python suite alone).
"""

from __future__ import annotations

import os
import pickle

import pytest

STATE_PATH = "debug/death_fresh/state_at_death.pkl"


@pytest.fixture
def env():
    try:
        from retro_ai.training.env_builder import build_training_env
        from retro_ai.training.run_config import EnvConfig
    except Exception as e:
        pytest.skip(f"training env not importable: {e}")

    # Skip if the ROM env isn't set up; this is an integration test.
    if not os.environ.get("RETRO_AI_ROM_DIR"):
        pytest.skip("RETRO_AI_ROM_DIR not set")
    if not os.path.exists(STATE_PATH):
        pytest.skip(f"missing fixture: {STATE_PATH}")

    env_cfg = EnvConfig(
        profile="yeti_fruit",
        action_mode="joystick",
        max_steps=1_000_000,
        stall_threshold=1_000_000,
        resize=(84, 84),
    )
    stack = build_training_env("yeti_fruit", env_cfg)
    base = stack.base
    base.reset(seed=0)
    return base


def _load_state(base):
    with open(STATE_PATH, "rb") as f:
        state = pickle.load(f)
    if not isinstance(state, (bytes, bytearray)):
        state = bytes(state)
    base._interface.load_state(state)


def _first_done_frame(base, n_frames: int = 30) -> int | None:
    """Return the 1-indexed frame at which env.step returns done=True,
    or None if no done fires during the first ``n_frames`` steps."""
    for f in range(1, n_frames + 1):
        _, _, done, _, _ = base.step([0, 0, 0])
        if done:
            return f
    return None


def test_first_done_is_deterministic_across_load_contexts(env):
    """Same save loaded in two different contexts should give the same
    first_done frame. Before the fix, this differed wildly.
    """
    base = env

    # Context 1: fresh reset, load, probe
    base.reset(seed=0)
    _load_state(base)
    first_a = _first_done_frame(base, n_frames=30)

    # Context 2: run until bonus-stall fires naturally, then load, probe
    base.reset(seed=0)
    for _ in range(1000):
        base.step([0, 0, 0])
    _load_state(base)
    first_b = _first_done_frame(base, n_frames=30)

    assert first_a == first_b, (
        f"load_state leaks trackers across calls: "
        f"first_done_frame was {first_a} from fresh reset, {first_b} "
        f"after running until stall. They should match because the "
        f"same save-state is loaded in both cases."
    )


def test_double_load_gives_same_first_done(env):
    """Loading the same state twice in a row should give the same
    first_done frame. Before the fix, the second load inherited
    trackers accumulated during the first load+probe.
    """
    base = env

    base.reset(seed=0)
    _load_state(base)
    first_1 = _first_done_frame(base, n_frames=30)

    _load_state(base)
    first_2 = _first_done_frame(base, n_frames=30)

    assert first_1 == first_2, (
        f"Second load should behave identically to the first, "
        f"got {first_1} vs {first_2}."
    )
