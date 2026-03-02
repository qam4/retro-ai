"""Unit tests for the preprocessing module."""

import numpy as np
import pytest

from retro_ai.core.preprocessing import PreprocessedEnv, PreprocessingPipeline


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _random_rgb_frame(h: int = 60, w: int = 80, rng=None) -> np.ndarray:
    """Return a random (H, W, 3) uint8 frame."""
    if rng is None:
        rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


class FakeEnv:
    """Minimal stand-in for BaseEnv used by PreprocessedEnv tests."""

    def __init__(self, h: int = 60, w: int = 80, channels: int = 3):
        self._h = h
        self._w = w
        self._c = channels
        self._rng = np.random.default_rng(0)
        self._step_count = 0

    def reset(self, seed=None):
        self._step_count = 0
        obs = self._rng.integers(0, 256, (self._h, self._w, self._c), dtype=np.uint8)
        return obs, {"reset": True}

    def step(self, action):
        self._step_count += 1
        obs = self._rng.integers(0, 256, (self._h, self._w, self._c), dtype=np.uint8)
        done = self._step_count >= 100
        return obs, 1.0, done, False, {"step": self._step_count}

    def get_observation_space(self):
        return {
            "width": self._w,
            "height": self._h,
            "channels": self._c,
            "bits_per_channel": 8,
        }

    def get_action_space(self):
        return {"type": "discrete", "shape": [4]}

    def save_state(self):
        return b"\x00"

    def load_state(self, state):
        pass

    def set_reward_mode(self, mode):
        pass

    def available_reward_modes(self):
        return ["survival"]


# ==================================================================
# PreprocessingPipeline tests
# ==================================================================


class TestGrayscaleConversion:
    """Requirement 18.1 — grayscale conversion."""

    def test_output_shape(self):
        frame = _random_rgb_frame(60, 80)
        pipe = PreprocessingPipeline(grayscale=True)
        out = pipe.reset(frame)
        assert out.shape == (60, 80, 1)

    def test_luminance_formula(self):
        """Verify the exact 0.299/0.587/0.114 coefficients."""
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        frame[0, 0] = [100, 0, 0]  # pure red
        frame[0, 1] = [0, 100, 0]  # pure green
        frame[1, 0] = [0, 0, 100]  # pure blue
        frame[1, 1] = [100, 100, 100]  # gray

        pipe = PreprocessingPipeline(grayscale=True)
        out = pipe.reset(frame)

        assert out[0, 0, 0] == int(0.299 * 100)
        assert out[0, 1, 0] == int(0.587 * 100)
        assert out[1, 0, 0] == int(0.114 * 100)
        # Equal channels → same value
        expected_gray = int(0.299 * 100 + 0.587 * 100 + 0.114 * 100)
        assert out[1, 1, 0] == expected_gray

    def test_dtype_uint8(self):
        frame = _random_rgb_frame()
        pipe = PreprocessingPipeline(grayscale=True)
        out = pipe.reset(frame)
        assert out.dtype == np.uint8

    def test_noop_when_disabled(self):
        frame = _random_rgb_frame()
        pipe = PreprocessingPipeline(grayscale=False)
        out = pipe.reset(frame)
        np.testing.assert_array_equal(out, frame)


class TestResize:
    """Requirement 18.2 — frame resizing."""

    def test_output_dimensions(self):
        frame = _random_rgb_frame(60, 80)
        pipe = PreprocessingPipeline(resize=(84, 84))
        out = pipe.reset(frame)
        assert out.shape == (84, 84, 3)

    def test_downscale(self):
        frame = _random_rgb_frame(120, 160)
        pipe = PreprocessingPipeline(resize=(30, 40))
        out = pipe.reset(frame)
        assert out.shape == (30, 40, 3)

    def test_upscale(self):
        frame = _random_rgb_frame(30, 40)
        pipe = PreprocessingPipeline(resize=(60, 80))
        out = pipe.reset(frame)
        assert out.shape == (60, 80, 3)

    def test_resize_with_grayscale(self):
        frame = _random_rgb_frame(60, 80)
        pipe = PreprocessingPipeline(grayscale=True, resize=(42, 42))
        out = pipe.reset(frame)
        assert out.shape == (42, 42, 1)

    def test_noop_when_none(self):
        frame = _random_rgb_frame(60, 80)
        pipe = PreprocessingPipeline(resize=None)
        out = pipe.reset(frame)
        assert out.shape == (60, 80, 3)


class TestFrameStacking:
    """Requirement 18.3 — frame stacking."""

    def test_stacked_shape(self):
        frame = _random_rgb_frame(60, 80)
        pipe = PreprocessingPipeline(frame_stack=4)
        out = pipe.reset(frame)
        # 3 channels × 4 stacked = 12
        assert out.shape == (60, 80, 12)

    def test_initial_frames_identical(self):
        """On reset, all stacked frames should be copies of the first."""
        frame = _random_rgb_frame(10, 10)
        pipe = PreprocessingPipeline(frame_stack=3)
        out = pipe.reset(frame)
        # Each slice of 3 channels should be the same
        for i in range(3):
            np.testing.assert_array_equal(
                out[..., i * 3 : (i + 1) * 3], frame
            )

    def test_new_frame_replaces_oldest(self):
        pipe = PreprocessingPipeline(grayscale=True, frame_stack=2)
        frame1 = np.full((4, 4, 3), 50, dtype=np.uint8)
        frame2 = np.full((4, 4, 3), 150, dtype=np.uint8)

        out1 = pipe.reset(frame1)
        assert out1.shape == (4, 4, 2)  # 1 channel × 2 stacked

        out2 = pipe.process(frame2)
        # Channel 0 = old frame1 grayscale, channel 1 = new frame2 grayscale
        assert out2.shape == (4, 4, 2)
        # The second channel should reflect frame2's grayscale value
        expected_gray2 = int(0.299 * 150 + 0.587 * 150 + 0.114 * 150)
        assert out2[0, 0, 1] == expected_gray2

    def test_stack_1_no_buffer(self):
        pipe = PreprocessingPipeline(frame_stack=1)
        assert pipe.frame_buffer is None
        frame = _random_rgb_frame(10, 10)
        out = pipe.reset(frame)
        np.testing.assert_array_equal(out, frame)

    def test_stacking_with_grayscale_and_resize(self):
        pipe = PreprocessingPipeline(
            grayscale=True, resize=(42, 42), frame_stack=4
        )
        frame = _random_rgb_frame(60, 80)
        out = pipe.reset(frame)
        assert out.shape == (42, 42, 4)  # 1 channel × 4 stacked


class TestFrameSkip:
    """Requirement 18.4 — frame skipping (tested via PreprocessedEnv)."""

    def test_skip_attribute(self):
        pipe = PreprocessingPipeline(frame_skip=4)
        assert pipe.frame_skip == 4


# ==================================================================
# PreprocessedEnv tests
# ==================================================================


class TestPreprocessedEnvReset:
    def test_returns_preprocessed_obs_and_info(self):
        env = FakeEnv(60, 80)
        pipe = PreprocessingPipeline(grayscale=True, resize=(42, 42))
        wrapped = PreprocessedEnv(env, pipe)
        obs, info = wrapped.reset()
        assert obs.shape == (42, 42, 1)
        assert isinstance(info, dict)

    def test_reset_with_seed(self):
        env = FakeEnv()
        pipe = PreprocessingPipeline()
        wrapped = PreprocessedEnv(env, pipe)
        obs, _ = wrapped.reset(seed=123)
        assert isinstance(obs, np.ndarray)


class TestPreprocessedEnvStep:
    def test_returns_5_tuple(self):
        env = FakeEnv()
        pipe = PreprocessingPipeline()
        wrapped = PreprocessedEnv(env, pipe)
        wrapped.reset()
        result = wrapped.step(0)
        assert len(result) == 5
        obs, reward, done, truncated, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_frame_skip_accumulates_reward(self):
        env = FakeEnv()
        pipe = PreprocessingPipeline(frame_skip=4)
        wrapped = PreprocessedEnv(env, pipe)
        wrapped.reset()
        _, reward, _, _, _ = wrapped.step(0)
        # FakeEnv returns 1.0 per step, 4 skips → 4.0
        assert reward == pytest.approx(4.0)

    def test_frame_skip_stops_on_done(self):
        """If the env terminates mid-skip, stop early."""

        class QuickDoneEnv(FakeEnv):
            def step(self, action):
                self._step_count += 1
                obs = np.zeros((self._h, self._w, self._c), dtype=np.uint8)
                done = self._step_count >= 2  # done after 2 steps
                return obs, 1.0, done, False, {}

        env = QuickDoneEnv()
        pipe = PreprocessingPipeline(frame_skip=10)
        wrapped = PreprocessedEnv(env, pipe)
        wrapped.reset()
        _, reward, done, _, _ = wrapped.step(0)
        assert done is True
        assert reward == pytest.approx(2.0)  # only 2 steps executed

    def test_preprocessed_obs_shape(self):
        env = FakeEnv(60, 80)
        pipe = PreprocessingPipeline(grayscale=True, resize=(42, 42), frame_stack=4)
        wrapped = PreprocessedEnv(env, pipe)
        wrapped.reset()
        obs, _, _, _, _ = wrapped.step(0)
        assert obs.shape == (42, 42, 4)


class TestPreprocessedEnvObservationSpace:
    def test_basic(self):
        env = FakeEnv(60, 80, 3)
        pipe = PreprocessingPipeline()
        wrapped = PreprocessedEnv(env, pipe)
        space = wrapped.get_observation_space()
        assert space == {"width": 80, "height": 60, "channels": 3, "bits_per_channel": 8}

    def test_with_grayscale(self):
        env = FakeEnv(60, 80, 3)
        pipe = PreprocessingPipeline(grayscale=True)
        wrapped = PreprocessedEnv(env, pipe)
        space = wrapped.get_observation_space()
        assert space["channels"] == 1

    def test_with_resize(self):
        env = FakeEnv(60, 80, 3)
        pipe = PreprocessingPipeline(resize=(84, 84))
        wrapped = PreprocessedEnv(env, pipe)
        space = wrapped.get_observation_space()
        assert space["height"] == 84
        assert space["width"] == 84

    def test_with_all_transforms(self):
        env = FakeEnv(60, 80, 3)
        pipe = PreprocessingPipeline(
            grayscale=True, resize=(42, 42), frame_stack=4
        )
        wrapped = PreprocessedEnv(env, pipe)
        space = wrapped.get_observation_space()
        assert space == {"width": 42, "height": 42, "channels": 4, "bits_per_channel": 8}


class TestPreprocessedEnvDelegation:
    """Verify that delegated methods pass through correctly."""

    def test_get_action_space(self):
        env = FakeEnv()
        wrapped = PreprocessedEnv(env, PreprocessingPipeline())
        assert wrapped.get_action_space() == {"type": "discrete", "shape": [4]}

    def test_save_load_state(self):
        env = FakeEnv()
        wrapped = PreprocessedEnv(env, PreprocessingPipeline())
        state = wrapped.save_state()
        wrapped.load_state(state)  # should not raise

    def test_reward_modes(self):
        env = FakeEnv()
        wrapped = PreprocessedEnv(env, PreprocessingPipeline())
        assert wrapped.available_reward_modes() == ["survival"]
        wrapped.set_reward_mode("survival")  # should not raise
