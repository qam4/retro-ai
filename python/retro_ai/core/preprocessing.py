"""Observation preprocessing pipeline for retro-ai environments.

Provides composable transforms (grayscale, resize, frame stacking) and a
``PreprocessedEnv`` wrapper that applies them transparently around a
:class:`~retro_ai.envs.base_env.BaseEnv`.

All transforms use only NumPy — no OpenCV dependency is required.

Requirements: 18.1, 18.2, 18.3, 18.4, 18.5
"""

from collections import deque
from typing import Any, Dict, Optional, Tuple

import numpy as np


class PreprocessingPipeline:
    """Apply preprocessing transformations to observations.

    Parameters
    ----------
    grayscale : bool
        Convert RGB (H, W, 3) frames to grayscale (H, W, 1) using the
        luminance formula 0.299×R + 0.587×G + 0.114×B.
    resize : tuple of (int, int) or None
        Target ``(height, width)`` for nearest-neighbour resizing.
        ``None`` keeps the original dimensions.
    frame_stack : int
        Number of consecutive frames to stack along the channel axis.
        A value of 1 disables stacking.
    frame_skip : int
        Number of times to repeat the same action, accumulating rewards.
        A value of 1 means no skipping.
    """

    def __init__(
        self,
        grayscale: bool = False,
        resize: Optional[Tuple[int, int]] = None,
        frame_stack: int = 1,
        frame_skip: int = 1,
    ) -> None:
        self.grayscale = grayscale
        self.resize = resize  # (target_height, target_width)
        self.frame_stack = frame_stack
        self.frame_skip = frame_skip

        if frame_stack > 1:
            self.frame_buffer: Optional[deque] = deque(maxlen=frame_stack)
        else:
            self.frame_buffer = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, observation: np.ndarray) -> np.ndarray:
        """Reset internal state and process the initial observation.

        When frame stacking is enabled the buffer is filled with copies
        of the first processed frame so that the output shape is
        immediately ``(H, W, C * frame_stack)``.
        """
        processed = self._process_single_frame(observation)

        if self.frame_buffer is not None:
            self.frame_buffer.clear()
            for _ in range(self.frame_stack):
                self.frame_buffer.append(processed)
            return self._stack_frames()

        return processed

    def process(self, observation: np.ndarray) -> np.ndarray:
        """Process a single observation through the pipeline."""
        processed = self._process_single_frame(observation)

        if self.frame_buffer is not None:
            self.frame_buffer.append(processed)
            return self._stack_frames()

        return processed

    # ------------------------------------------------------------------
    # Internal transforms
    # ------------------------------------------------------------------

    def _process_single_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply grayscale and resize to one frame."""
        # Grayscale conversion  (Req 18.1)
        if self.grayscale and frame.ndim == 3 and frame.shape[-1] == 3:
            gray = (
                0.299 * frame[..., 0]
                + 0.587 * frame[..., 1]
                + 0.114 * frame[..., 2]
            )
            frame = np.expand_dims(gray.astype(np.uint8), axis=-1)

        # Nearest-neighbour resize using pure NumPy  (Req 18.2)
        if self.resize is not None:
            target_h, target_w = self.resize
            src_h, src_w = frame.shape[0], frame.shape[1]
            row_idx = (np.arange(target_h) * src_h // target_h).astype(int)
            col_idx = (np.arange(target_w) * src_w // target_w).astype(int)
            frame = frame[np.ix_(row_idx, col_idx)]

        return frame

    def _stack_frames(self) -> np.ndarray:
        """Concatenate buffered frames along the channel axis."""
        return np.concatenate(list(self.frame_buffer), axis=-1)


class PreprocessedEnv:
    """Wrapper that applies a :class:`PreprocessingPipeline` to a BaseEnv.

    Frame skipping (action repetition with reward accumulation) is handled
    here rather than inside the pipeline so that the environment's ``step``
    is called the correct number of times.

    Parameters
    ----------
    env : BaseEnv
        The underlying environment to wrap.
    preprocessing : PreprocessingPipeline
        The pipeline that will transform observations.
    """

    def __init__(self, env: Any, preprocessing: PreprocessingPipeline) -> None:
        self.env = env
        self.preprocessing = preprocessing

    # ------------------------------------------------------------------
    # Core RL API
    # ------------------------------------------------------------------

    def reset(
        self, seed: Optional[int] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the wrapped environment and preprocess the observation.

        Returns
        -------
        observation : np.ndarray
            Preprocessed initial observation.
        info : dict
            Metadata from the underlying environment.
        """
        obs, info = self.env.reset(seed=seed)
        return self.preprocessing.reset(obs), info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute *action* with frame skipping and preprocessing.

        The action is repeated ``frame_skip`` times (or until the episode
        ends).  Rewards are accumulated across skipped frames.
        """
        total_reward = 0.0
        done = False
        truncated = False
        info: Dict[str, Any] = {}

        for _ in range(self.preprocessing.frame_skip):
            obs, reward, done, truncated, info = self.env.step(action)
            total_reward += reward
            if done or truncated:
                break

        processed_obs = self.preprocessing.process(obs)
        return processed_obs, total_reward, done, truncated, info

    # ------------------------------------------------------------------
    # Delegated helpers
    # ------------------------------------------------------------------

    def get_observation_space(self) -> Dict[str, Any]:
        """Return the observation space *after* preprocessing."""
        original = self.env.get_observation_space()

        if self.preprocessing.resize is not None:
            height, width = self.preprocessing.resize
        else:
            width = original["width"]
            height = original["height"]

        channels = 1 if self.preprocessing.grayscale else original["channels"]
        channels *= self.preprocessing.frame_stack

        return {
            "width": width,
            "height": height,
            "channels": channels,
            "bits_per_channel": original["bits_per_channel"],
        }

    def get_action_space(self) -> Dict[str, Any]:
        """Delegate to the wrapped environment."""
        return self.env.get_action_space()

    def save_state(self) -> bytes:
        """Delegate to the wrapped environment."""
        return self.env.save_state()

    def load_state(self, state: bytes) -> None:
        """Delegate to the wrapped environment."""
        self.env.load_state(state)

    def set_reward_mode(self, mode: str) -> None:
        """Delegate to the wrapped environment."""
        self.env.set_reward_mode(mode)

    def available_reward_modes(self) -> list:
        """Delegate to the wrapped environment."""
        return self.env.available_reward_modes()
