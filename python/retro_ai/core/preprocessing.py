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
        crop: Optional[Tuple[int, int, int, int]] = None,
        augmentation: bool = False,
        aug_pad: int = 4,
        aug_jitter: int = 10,
    ) -> None:
        if not (1 <= frame_skip <= 16):
            raise ValueError(
                f"frame_skip must be between 1 and 16 inclusive, got {frame_skip}"
            )

        self.grayscale = grayscale
        self.resize = resize  # (target_height, target_width)
        self.frame_stack = frame_stack
        self.frame_skip = frame_skip
        self.crop = crop  # (y, x, height, width) — applied before grayscale/resize
        self.augmentation = augmentation
        self.aug_pad = aug_pad
        self.aug_jitter = aug_jitter
        self._rng = np.random.default_rng()

        if frame_stack > 1:
            self.frame_buffer: Optional[deque] = deque(maxlen=frame_stack)
        else:
            self.frame_buffer = None
        # When True, the next ``process()`` re-seeds the stack (fills it with
        # the incoming frame, like ``reset()``) instead of appending. Set via
        # ``mark_reseed()`` after a mid-episode ``load_state`` so no pre-load
        # frame survives in the stack. See PreprocessedEnv.notify_state_loaded.
        self._reseed_pending = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, observation: np.ndarray) -> np.ndarray:
        """Reset internal state and process the initial observation.

        When frame stacking is enabled the buffer is filled with copies
        of the first processed frame so that the output shape is
        immediately ``(H, W, C * frame_stack)``.
        """
        self._reseed_pending = False
        processed = self._process_single_frame(observation)

        if self.frame_buffer is not None:
            self.frame_buffer.clear()
            for _ in range(self.frame_stack):
                self.frame_buffer.append(processed)
            return self._stack_frames()

        return processed

    def mark_reseed(self) -> None:
        """Request that the next :meth:`process` re-seed the frame stack
        (fill it entirely with the incoming frame) instead of appending.

        Used after a mid-episode ``load_state`` so the stack contains no
        frames from before the load — without relying on stepping
        ``>= frame_stack`` noops to flush them.
        """
        self._reseed_pending = True

    def process(self, observation: np.ndarray) -> np.ndarray:
        """Process a single observation through the pipeline."""
        processed = self._process_single_frame(observation)

        if self.frame_buffer is not None:
            if self._reseed_pending:
                # Re-seed: drop any pre-load frames, fill with this frame.
                self._reseed_pending = False
                self.frame_buffer.clear()
                for _ in range(self.frame_stack):
                    self.frame_buffer.append(processed)
            else:
                self.frame_buffer.append(processed)
            return self._stack_frames()

        return processed

    # ------------------------------------------------------------------
    # Internal transforms
    # ------------------------------------------------------------------

    def _process_single_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply crop, grayscale, resize, and optional augmentation to one frame."""
        # Crop to region of interest (applied first)
        if self.crop is not None:
            y, x, h, w = self.crop
            frame = frame[y : y + h, x : x + w]

        # Grayscale conversion  (Req 18.1)
        if self.grayscale and frame.ndim == 3 and frame.shape[-1] == 3:
            gray = 0.299 * frame[..., 0] + 0.587 * frame[..., 1] + 0.114 * frame[..., 2]
            frame = np.expand_dims(gray.astype(np.uint8), axis=-1)

        # Nearest-neighbour resize using pure NumPy  (Req 18.2)
        if self.resize is not None:
            target_h, target_w = self.resize
            src_h, src_w = frame.shape[0], frame.shape[1]
            row_idx = (np.arange(target_h) * src_h // target_h).astype(int)
            col_idx = (np.arange(target_w) * src_w // target_w).astype(int)
            frame = frame[np.ix_(row_idx, col_idx)]

        # Data augmentation (Req 13.4 — after grayscale/resize, before stacking)
        if self.augmentation:
            frame = self._augment(frame)

        return frame

    def _augment(self, frame: np.ndarray) -> np.ndarray:
        """Apply random crop + intensity jitter (pure NumPy, Req 13.2, 13.3, 13.5)."""
        h, w = frame.shape[:2]
        pad = self.aug_pad

        # Pad with edge values, then random crop back to original size
        if frame.ndim == 3:
            padded = np.pad(frame, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
        else:
            padded = np.pad(frame, ((pad, pad), (pad, pad)), mode="edge")

        y0 = self._rng.integers(0, 2 * pad + 1)
        x0 = self._rng.integers(0, 2 * pad + 1)
        frame = padded[y0 : y0 + h, x0 : x0 + w]

        # Intensity jitter
        jitter = self._rng.integers(-self.aug_jitter, self.aug_jitter + 1)
        frame = np.clip(frame.astype(np.int16) + jitter, 0, 255).astype(np.uint8)

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

    def __init__(
        self,
        env: Any,
        preprocessing: PreprocessingPipeline,
        frame_maxpool: bool = False,
    ) -> None:
        self.env = env
        self.preprocessing = preprocessing
        self._frame_maxpool = frame_maxpool
        self._prev_raw_frame: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Core RL API
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the wrapped environment and preprocess the observation.

        Returns
        -------
        observation : np.ndarray
            Preprocessed initial observation.
        info : dict
            Metadata from the underlying environment.
        """
        obs, info = self.env.reset(seed=seed)
        if self._frame_maxpool:
            self._prev_raw_frame = obs.copy()
        return self.preprocessing.reset(obs), info

    def notify_state_loaded(self) -> None:
        """Call right after a mid-episode ``load_state`` on the underlying
        env (which bypasses :meth:`reset`).

        Clears the cross-step buffers so no pre-load frame leaks into the
        policy's observations: the maxpool ``_prev_raw_frame`` is dropped
        (so the next step doesn't max a post-load frame against a pre-load
        one) and the frame stack is flagged to re-seed on the next
        ``process``. After this, ONE step yields a fully post-load stacked
        observation — no dependence on stepping ``>= frame_stack`` noops.
        """
        self._prev_raw_frame = None
        self.preprocessing.mark_reseed()

    # ------------------------------------------------------------------
    # Frame-stack save / restore (on-distribution save-state starts)
    # ------------------------------------------------------------------
    # notify_state_loaded() RE-SEEDS the stack (fills it with copies of the
    # single post-load frame), so a seeded episode's first observation shows
    # a MOTIONLESS scene even if the agent was moving when captured — off
    # distribution, and it forces settle noops that advance the game. The
    # pair below instead snapshots the real frame stack at capture and
    # restores it on load, so the seeded start is a faithful continuation of
    # the live trajectory with NO stepping. See experiments/003 H-AB.

    def _stack_signature(self) -> tuple:
        """Config fingerprint guarding stack compatibility across runs.

        A restored stack is only valid if the pipeline that consumes it
        matches the one that produced it (same grayscale/resize/stack/crop,
        hence same per-frame shape); otherwise fall back to reseed.
        """
        p = self.preprocessing
        shape = None
        if p.frame_buffer is not None and len(p.frame_buffer) > 0:
            shape = tuple(p.frame_buffer[0].shape)
        return (
            bool(p.grayscale),
            tuple(p.resize) if p.resize else None,
            int(p.frame_stack),
            tuple(p.crop) if p.crop else None,
            shape,
        )

    def export_frame_stack(self) -> Optional[Dict[str, Any]]:
        """Snapshot the current (processed) frame stack for save-stating.

        Returns a picklable dict {sig, frames} or None when there is no
        populated multi-frame stack to save (frame_stack <= 1, or not yet
        filled). ``frames`` are copies of the processed frames currently in
        the stack (the real motion history).
        """
        p = self.preprocessing
        if p.frame_buffer is None or len(p.frame_buffer) == 0:
            return None
        return {
            "sig": self._stack_signature(),
            "frames": [f.copy() for f in p.frame_buffer],
        }

    def restore_frame_stack(self, blob: Optional[Dict[str, Any]]) -> bool:
        """Restore a stack saved by :meth:`export_frame_stack`.

        Returns True if the stack was restored (then :meth:`current_observation`
        is valid with NO stepping). Returns False when the blob is missing,
        malformed, or its signature does not match this pipeline — the caller
        must then fall back to :meth:`notify_state_loaded` + a settle step.
        """
        p = self.preprocessing
        if not blob or p.frame_buffer is None:
            return False
        if blob.get("sig") != self._stack_signature():
            return False
        frames = blob.get("frames")
        if not frames or len(frames) != p.frame_stack:
            return False
        p.frame_buffer.clear()
        for f in frames:
            p.frame_buffer.append(f)
        p._reseed_pending = False
        # Skip maxpool carry-over for the first post-load step (would need the
        # raw pre-load frame; the one-step skip is negligible).
        self._prev_raw_frame = None
        return True

    def current_observation(self) -> np.ndarray:
        """The current stacked observation WITHOUT stepping the env.

        Valid after :meth:`restore_frame_stack`; lets a seeded episode start
        at settle 0 with the faithful, on-distribution observation.
        """
        return self.preprocessing._stack_frames()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute *action* with frame skipping and preprocessing.

        When ``frame_skip > 1`` and the underlying environment exposes a
        C++ ``step_n`` method (via ``BaseEnv._interface``), the entire
        skip sequence is executed in a single Python→C++ call.  Otherwise
        the existing Python-side loop is used as a fallback.
        """
        if self.preprocessing.frame_skip > 1 and hasattr(self.env, "step_n"):
            # Fast path: single C++ round-trip for all skipped frames
            obs, reward, done, truncated, info = self.env.step_n(
                action, self.preprocessing.frame_skip
            )

            if self._frame_maxpool:
                if self._prev_raw_frame is not None:
                    maxpooled = np.maximum(self._prev_raw_frame, obs)
                else:
                    maxpooled = obs
                self._prev_raw_frame = obs.copy()
                obs = maxpooled

            processed_obs = self.preprocessing.process(obs)
            return processed_obs, reward, done, truncated, info

        # Fallback: Python-side frame skip loop
        total_reward = 0.0
        done = False
        truncated = False
        info: Dict[str, Any] = {}

        for _ in range(self.preprocessing.frame_skip):
            obs, reward, done, truncated, info = self.env.step(action)
            total_reward += reward
            if done or truncated:
                break

        if self._frame_maxpool:
            if self._prev_raw_frame is not None:
                maxpooled = np.maximum(self._prev_raw_frame, obs)
            else:
                maxpooled = obs
            self._prev_raw_frame = obs.copy()
            obs = maxpooled

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
