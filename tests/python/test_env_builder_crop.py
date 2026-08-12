"""``build_training_env`` must honour the crop, from the profile or the config.

Regression: the builder used to drop ``crop`` on the floor, so a profile that
declared one was silently ignored on the curriculum-training and eval paths
(while ``training/pipeline.py`` honoured it). A crop that goes missing between
training and eval is invisible and changes the observation the policy sees.

Stubbed, so these run without a ROM or the native module.
"""

from __future__ import annotations

import numpy as np
import pytest
from retro_ai.training import env_builder
from retro_ai.training.game_profile import GameProfile
from retro_ai.training.run_config import EnvConfig

RAW_H, RAW_W = 200, 320
HUD_CROP = (16, 0, 184, 320)  # drop the top 16 rows (Yeti's HUD strip)


class _StubEnv:
    """Stands in for BaseEnv; yields blank frames of the raw MO5 size."""

    def __init__(self, *args, **kwargs):
        pass

    def reset(self, seed=None):
        return np.zeros((RAW_H, RAW_W, 3), dtype=np.uint8), {}

    def step(self, action):
        return np.zeros((RAW_H, RAW_W, 3), dtype=np.uint8), 0.0, False, False, {}

    def get_observation_space(self):
        return {"width": RAW_W, "height": RAW_H, "channels": 3, "bits_per_channel": 8}

    def get_action_space(self):
        return {"type": "discrete", "shape": [4]}


@pytest.fixture
def stub_builder(monkeypatch):
    """Patch out the ROM-backed env and the profile registry."""

    def _build(profile_crop):
        profile = GameProfile(
            name="stub",
            emulator_type="mo5",
            rom_path="/nonexistent.rom",
            reward_mode="stub",
            crop=profile_crop,
            grayscale=True,
            frame_stack=1,
            frame_skip=1,
        )
        monkeypatch.setattr(env_builder, "BaseEnv", _StubEnv)
        monkeypatch.setattr(
            env_builder.GameProfileRegistry, "load", lambda self, name: profile
        )
        return profile

    return _build


def _pipeline_of(stack):
    return stack.preprocessed.preprocessing


def test_profile_crop_is_applied(stub_builder):
    stub_builder(HUD_CROP)
    stack = env_builder.build_training_env(
        "stub", EnvConfig(profile="stub", resize=None)
    )
    assert _pipeline_of(stack).crop == HUD_CROP
    obs, _ = stack.gym.reset()
    assert obs.shape[:2] == (184, RAW_W), "cropped rows must not reach the policy"


def test_env_config_crop_overrides_profile(stub_builder):
    stub_builder(HUD_CROP)
    override = (32, 8, 100, 200)
    stack = env_builder.build_training_env(
        "stub", EnvConfig(profile="stub", resize=None, crop=override)
    )
    assert _pipeline_of(stack).crop == override
    obs, _ = stack.gym.reset()
    assert obs.shape[:2] == (100, 200)


def test_no_crop_by_default(stub_builder):
    """Absent an explicit crop, observations keep the full raw frame."""
    stub_builder(None)
    stack = env_builder.build_training_env(
        "stub", EnvConfig(profile="stub", resize=None)
    )
    assert _pipeline_of(stack).crop is None
    obs, _ = stack.gym.reset()
    assert obs.shape[:2] == (RAW_H, RAW_W)
