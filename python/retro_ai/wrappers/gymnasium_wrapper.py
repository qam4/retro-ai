"""Optional Gymnasium-compatible wrapper for retro-ai environments.

Requires the ``gymnasium`` package.  Import this module only when you
need Gymnasium integration — the rest of retro-ai works without it.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    raise ImportError(
        "gymnasium is required for GymnasiumWrapper. " "Install it with:  pip install gymnasium"
    )

from retro_ai.envs.base_env import BaseEnv


class GymnasiumWrapper(gym.Env):
    """Wrap a :class:`BaseEnv` as a Gymnasium environment.

    Parameters
    ----------
    base_env : BaseEnv
        An already-constructed :class:`BaseEnv` instance.
    render_mode : str or None
        Gymnasium render mode.  Only ``"rgb_array"`` is supported
        (returns the raw framebuffer).
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        base_env: BaseEnv,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._env = base_env
        self.render_mode = render_mode

        # Build observation space
        obs = self._env.get_observation_space()
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(obs["height"], obs["width"], obs["channels"]),
            dtype=np.uint8,
        )

        # Build action space
        act = self._env.get_action_space()
        shape = act["shape"]
        if len(shape) == 1:
            self.action_space = spaces.Discrete(shape[0])
        else:
            self.action_space = spaces.MultiDiscrete(shape)

        self._last_obs: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment.

        Returns
        -------
        observation : np.ndarray
        info : dict
        """
        super().reset(seed=seed)
        obs, info = self._env.reset(seed=seed)
        self._last_obs = obs
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step.

        Returns
        -------
        observation, reward, terminated, truncated, info
        """
        obs, reward, done, truncated, info = self._env.step(action)
        self._last_obs = obs
        return obs, reward, done, truncated, info

    def render(self) -> Optional[np.ndarray]:
        """Return the current frame as an RGB array."""
        if self.render_mode == "rgb_array":
            return self._last_obs
        return None

    def close(self) -> None:
        """Clean up resources."""
        pass

    # ------------------------------------------------------------------
    # State management (pass-through)
    # ------------------------------------------------------------------

    def save_state(self) -> bytes:
        """Serialize emulator state."""
        return self._env.save_state()

    def load_state(self, state: bytes) -> None:
        """Restore emulator state."""
        self._env.load_state(state)

    def set_reward_mode(self, mode: str) -> None:
        """Switch reward mode."""
        self._env.set_reward_mode(mode)

    def available_reward_modes(self) -> list:
        """List available reward modes."""
        return self._env.available_reward_modes()
