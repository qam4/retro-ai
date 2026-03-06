"""Framework-agnostic environment for retro game emulators.

This module provides BaseEnv, a thin Python wrapper around the C++ RLInterface
exposed by retro_ai_native.  It deliberately avoids importing Gymnasium or any
other RL framework so that users can integrate with any training library or
write custom training loops.
"""

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class BaseEnv:
    """Framework-agnostic environment for retro game emulators.

    Wraps a C++ ``RLInterface`` implementation (Videopac or MO5) and exposes a
    simple *reset / step* API that returns NumPy arrays.  No Gymnasium or other
    RL-framework dependency is required.

    Parameters
    ----------
    emulator_type : str
        Emulator backend to use.  One of ``"videopac"`` or ``"mo5"``.
    rom_path : str
        Path to the ROM (or tape) file.
    bios_path : str or None
        Path to the BIOS file.  Required for Videopac, ignored for MO5.
    reward_mode : str
        Initial reward computation mode (e.g. ``"survival"``).
    config : dict or None
        Reserved for future per-emulator configuration options.
    """

    def __init__(
        self,
        emulator_type: str,
        rom_path: str,
        bios_path: Optional[str] = None,
        reward_mode: str = "survival",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._interface = self._create_interface(
            emulator_type, rom_path, bios_path, reward_mode, config
        )
        self._obs_space = self._interface.observation_space()
        self._action_space = self._interface.action_space()

    # ------------------------------------------------------------------
    # Core RL API
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment and return the initial observation.

        Parameters
        ----------
        seed : int or None
            Optional RNG seed for deterministic resets.  ``None`` leaves the
            seed unchanged (maps to ``-1`` on the C++ side).

        Returns
        -------
        observation : numpy.ndarray
            Initial observation with shape ``(height, width, channels)`` and
            dtype ``uint8``.
        info : dict
            Metadata dictionary parsed from the native JSON info string.
        """
        native_seed = seed if seed is not None else -1
        result = self._interface.reset_numpy(native_seed)
        observation = result["observation"]
        info = self._parse_info(result["info"])
        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one environment step.

        Parameters
        ----------
        action : int
            The discrete action to execute.

        Returns
        -------
        observation : numpy.ndarray
            Current observation (``uint8``, shape ``(H, W, C)``).
        reward : float
            Scalar reward for this step.
        done : bool
            ``True`` when the episode has ended (game over).
        truncated : bool
            ``True`` when the episode was truncated (e.g. invalid action).
        info : dict
            Additional metadata dictionary.
        """
        result = self._interface.step_numpy([action])
        observation = result["observation"]
        reward = float(result["reward"])
        done = bool(result["done"])
        truncated = bool(result["truncated"])
        info = self._parse_info(result["info"])
        return observation, reward, done, truncated, info

    # ------------------------------------------------------------------
    # Space queries
    # ------------------------------------------------------------------

    def get_observation_space(self) -> Dict[str, Any]:
        """Return the observation-space specification as a plain dict.

        Keys: ``width``, ``height``, ``channels``, ``bits_per_channel``.
        """
        return {
            "width": self._obs_space.width,
            "height": self._obs_space.height,
            "channels": self._obs_space.channels,
            "bits_per_channel": self._obs_space.bits_per_channel,
        }

    def get_action_space(self) -> Dict[str, Any]:
        """Return the action-space specification as a plain dict.

        Keys: ``type`` (ActionType enum value), ``shape`` (list of ints).
        """
        return {
            "type": self._action_space.type,
            "shape": list(self._action_space.shape),
        }

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def save_state(self) -> bytes:
        """Serialize the current emulator state.

        Returns
        -------
        bytes
            Opaque snapshot that can be passed back to :meth:`load_state`.
        """
        state = self._interface.save_state()
        return bytes(state) if not isinstance(state, bytes) else state

    def load_state(self, state: bytes) -> None:
        """Restore a previously saved emulator state.

        Parameters
        ----------
        state : bytes
            A snapshot previously obtained from :meth:`save_state`.
        """
        self._interface.load_state(state)

    # ------------------------------------------------------------------
    # Reward configuration
    # ------------------------------------------------------------------

    def set_reward_mode(self, mode: str) -> None:
        """Switch the active reward computation mode.

        Parameters
        ----------
        mode : str
            One of the names returned by :meth:`available_reward_modes`.
        """
        self._interface.set_reward_mode(mode)

    def available_reward_modes(self) -> List[str]:
        """Return the list of supported reward mode names."""
        return list(self._interface.available_reward_modes())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_info(info_json: str) -> Dict[str, Any]:
        """Parse a JSON info string into a Python dict."""
        if not info_json:
            return {}
        try:
            parsed = json.loads(info_json)
            return parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return {}

    @staticmethod
    def _create_interface(
        emulator_type: str,
        rom_path: str,
        bios_path: Optional[str],
        reward_mode: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Factory: instantiate the correct native RLInterface."""
        import retro_ai_native  # local import to keep module-level clean

        # Flatten reward_params for the native constructor
        reward_params_flat: Dict[str, str] = {}
        if config and "reward_params" in config:
            reward_params_flat = BaseEnv._flatten_reward_params(config["reward_params"])

        emu = emulator_type.lower()
        if emu == "videopac":
            if bios_path is None:
                raise ValueError("Videopac emulator requires a bios_path")
            joystick_index = 0
            if config and "joystick_index" in config:
                joystick_index = int(config["joystick_index"])
            return retro_ai_native.VideopacRLInterface(
                bios_path, rom_path, reward_mode, joystick_index,
                reward_params=reward_params_flat,
            )
        if emu == "mo5":
            return retro_ai_native.MO5RLInterface(
                rom_path, reward_mode,
                reward_params=reward_params_flat,
            )

        raise ValueError(
            f"Unknown emulator type: {emulator_type!r}. "
            f"Supported types: 'videopac', 'mo5'"
        )

    @staticmethod
    def _flatten_reward_params(reward_params: Dict[str, Any]) -> Dict[str, str]:
        """Flatten nested reward_params into a string key-value map for C++."""
        flat: Dict[str, str] = {}
        if "screen_region" in reward_params:
            sr = reward_params["screen_region"]
            key_map = {"x": "x", "y": "y", "width": "w", "height": "h"}
            for src_key, dst_suffix in key_map.items():
                if src_key in sr:
                    flat[f"screen_region_{dst_suffix}"] = str(sr[src_key])
        if "score_addresses" in reward_params:
            for i, entry in enumerate(reward_params["score_addresses"]):
                flat[f"score_address_{i}_addr"] = str(entry["address"])
                flat[f"score_address_{i}_bytes"] = str(entry.get("num_bytes", 1))
                flat[f"score_address_{i}_bcd"] = str(int(entry.get("is_bcd", False)))
            flat["score_address_count"] = str(len(reward_params["score_addresses"]))
        return flat

