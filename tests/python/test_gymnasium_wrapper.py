"""Tests for the Gymnasium wrapper."""

import numpy as np
import pytest

# Skip entire module if gymnasium is not installed
gym = pytest.importorskip("gymnasium")

from retro_ai.wrappers.gymnasium_wrapper import GymnasiumWrapper


# ------------------------------------------------------------------
# Minimal mock BaseEnv for testing without native module
# ------------------------------------------------------------------


class _MockBaseEnv:
    """Lightweight stand-in for BaseEnv."""

    def __init__(self, width=160, height=200, channels=3, n_actions=18):
        self._w = width
        self._h = height
        self._c = channels
        self._n = n_actions
        self._obs = np.zeros((height, width, channels), dtype=np.uint8)

    def get_observation_space(self):
        return {
            "width": self._w,
            "height": self._h,
            "channels": self._c,
            "bits_per_channel": 8,
        }

    def get_action_space(self):
        return {"type": 0, "shape": [self._n]}

    def reset(self, seed=None):
        return self._obs.copy(), {"seed": seed}

    def step(self, action):
        return self._obs.copy(), 1.0, False, False, {}

    def save_state(self):
        return b"\x00"

    def load_state(self, state):
        pass

    def set_reward_mode(self, mode):
        pass

    def available_reward_modes(self):
        return ["survival", "memory"]


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestGymnasiumSpaces:
    def _make(self, **kw):
        return GymnasiumWrapper(_MockBaseEnv(**kw))

    def test_observation_space_shape(self):
        env = self._make(width=160, height=200, channels=3)
        assert env.observation_space.shape == (200, 160, 3)
        assert env.observation_space.dtype == np.uint8

    def test_discrete_action_space(self):
        env = self._make(n_actions=18)
        assert isinstance(env.action_space, gym.spaces.Discrete)
        assert env.action_space.n == 18


class TestGymnasiumReset:
    def test_returns_obs_and_info(self):
        env = GymnasiumWrapper(_MockBaseEnv())
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)

    def test_seed_forwarded(self):
        env = GymnasiumWrapper(_MockBaseEnv())
        obs, info = env.reset(seed=42)
        assert info.get("seed") == 42


class TestGymnasiumStep:
    def test_returns_5_tuple(self):
        env = GymnasiumWrapper(_MockBaseEnv())
        env.reset()
        result = env.step(0)
        assert len(result) == 5
        obs, reward, done, truncated, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)


class TestGymnasiumRender:
    def test_rgb_array(self):
        env = GymnasiumWrapper(_MockBaseEnv(), render_mode="rgb_array")
        env.reset()
        frame = env.render()
        assert isinstance(frame, np.ndarray)

    def test_none_without_render_mode(self):
        env = GymnasiumWrapper(_MockBaseEnv())
        env.reset()
        assert env.render() is None


class TestGymnasiumDelegation:
    def test_save_load_state(self):
        env = GymnasiumWrapper(_MockBaseEnv())
        state = env.save_state()
        assert isinstance(state, bytes)
        env.load_state(state)

    def test_reward_modes(self):
        env = GymnasiumWrapper(_MockBaseEnv())
        modes = env.available_reward_modes()
        assert "survival" in modes
