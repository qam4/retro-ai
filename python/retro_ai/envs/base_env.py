"""Framework-agnostic environment for retro game emulators.

This module provides BaseEnv, a thin Python wrapper around the C++ RLInterface
exposed by retro_ai_native.  It deliberately avoids importing Gymnasium or any
other RL framework so that users can integrate with any training library or
write custom training loops.
"""

import json
from typing import Any, Dict, List, Optional, Tuple, Union

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
    observation_mode : str
        Observation source.  ``"framebuffer"`` (default) returns the RGB
        pixel buffer; ``"ram"`` returns raw emulator RAM bytes, skipping
        framebuffer extraction entirely.
    """

    _VALID_OBSERVATION_MODES = {"framebuffer", "ram"}

    def __init__(
        self,
        emulator_type: str,
        rom_path: str,
        bios_path: Optional[str] = None,
        reward_mode: str = "survival",
        config: Optional[Dict[str, Any]] = None,
        observation_mode: str = "framebuffer",
        action_mode: str = "multi_discrete",
    ) -> None:
        if observation_mode not in self._VALID_OBSERVATION_MODES:
            raise ValueError(
                f"Invalid observation_mode: {observation_mode!r}. "
                f"Must be one of {sorted(self._VALID_OBSERVATION_MODES)}"
            )
        self._observation_mode = observation_mode
        self._action_mode = action_mode
        self._interface = self._create_interface(
            emulator_type,
            rom_path,
            bios_path,
            reward_mode,
            config,
            action_mode=action_mode,
        )
        self._obs_space = self._interface.observation_space()
        self._action_space = self._interface.action_space()
        self._last_raw_obs: Optional[np.ndarray] = None

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
        if self._observation_mode == "ram":
            ram_bytes = self._interface.read_ram()
            observation = np.frombuffer(bytes(ram_bytes), dtype=np.uint8).copy()
        else:
            observation = result["observation"]
        self._last_raw_obs = result["observation"]  # cache for video recording
        info = self._parse_info(result["info"])
        return observation, info

    def step(
        self, action: Union[int, List[int]]
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one environment step.

        Parameters
        ----------
        action : int or list[int]
            The action to execute.  For discrete mode, a single ``int``.
            For multi-discrete mode, a list of 5 binary values
            ``[up, down, left, right, fire]``.

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
        if self._action_mode == "multi_discrete":
            result = self._interface.step_numpy(
                action if isinstance(action, list) else list(action)
            )
        else:
            result = self._interface.step_numpy([action])
        if self._observation_mode == "ram":
            ram_bytes = self._interface.read_ram()
            observation = np.frombuffer(bytes(ram_bytes), dtype=np.uint8).copy()
        else:
            observation = result["observation"]
        self._last_raw_obs = result["observation"]  # cache for video recording
        reward = float(result["reward"])
        done = bool(result["done"])
        truncated = bool(result["truncated"])
        info = self._parse_info(result["info"])
        return observation, reward, done, truncated, info

    def step_n(
        self, action: Union[int, List[int]], n: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute *n* environment steps with the same action via C++ batching.

        Delegates to the native ``step_n_numpy`` method which runs all *n*
        frames in a single Python→C++ round-trip, avoiding per-frame overhead.

        Parameters
        ----------
        action : int or list[int]
            The action to repeat for *n* frames.  For discrete mode, a single
            ``int``.  For multi-discrete mode, a list of 5 binary values.
        n : int
            Number of frames to step.

        Returns
        -------
        observation : numpy.ndarray
            Final observation after *n* steps (``uint8``, shape ``(H, W, C)``).
        reward : float
            Accumulated reward across all *n* steps.
        done : bool
            ``True`` when the episode ended during the sequence.
        truncated : bool
            ``True`` when the episode was truncated.
        info : dict
            Metadata from the final step.
        """
        if self._action_mode == "multi_discrete":
            result = self._interface.step_n_numpy(
                action if isinstance(action, list) else list(action), n
            )
        else:
            result = self._interface.step_n_numpy([action], n)
        if self._observation_mode == "ram":
            ram_bytes = self._interface.read_ram()
            observation = np.frombuffer(bytes(ram_bytes), dtype=np.uint8).copy()
        else:
            observation = result["observation"]
        self._last_raw_obs = result["observation"]  # cache for video recording
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

        When ``observation_mode`` is ``"ram"``, the space is 1-D:
        ``(ram_size, 1, 1)`` with 8 bits per channel.
        """
        if self._observation_mode == "ram":
            ram_size = self._interface.ram_size()
            return {
                "width": ram_size,
                "height": 1,
                "channels": 1,
                "bits_per_channel": 8,
            }
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
        action_mode: str = "multi_discrete",
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
                bios_path,
                rom_path,
                reward_mode,
                joystick_index,
                reward_params=reward_params_flat,
                action_mode=action_mode,
            )
        if emu == "mo5":
            return retro_ai_native.MO5RLInterface(
                rom_path,
                reward_mode,
                reward_params=reward_params_flat,
            )

        raise ValueError(
            f"Unknown emulator type: {emulator_type!r}. "
            f"Supported types: 'videopac', 'mo5'"
        )

    @staticmethod
    def _flatten_reward_params(reward_params: Dict[str, Any]) -> Dict[str, str]:
        """Flatten nested reward_params into a string key-value map for C++.

        Handles two formats:
        1. Already-flat string maps (e.g. from YAML game profiles) — passed
           through as-is.
        2. Nested dicts (``screen_region``, ``score_addresses``) — expanded
           into the flat key convention expected by the C++ side.
        """
        flat: Dict[str, str] = {}

        for key, value in reward_params.items():
            # Nested structures get special handling
            if key == "screen_region" and isinstance(value, dict):
                key_map = {"x": "x", "y": "y", "width": "w", "height": "h"}
                for src_key, dst_suffix in key_map.items():
                    if src_key in value:
                        flat[f"screen_region_{dst_suffix}"] = str(value[src_key])
            elif key == "score_addresses" and isinstance(value, list):
                for i, entry in enumerate(value):
                    flat[f"score_address_{i}_addr"] = str(entry["address"])
                    flat[f"score_address_{i}_bytes"] = str(entry.get("num_bytes", 1))
                    flat[f"score_address_{i}_bcd"] = str(
                        int(entry.get("is_bcd", False))
                    )
                flat["score_address_count"] = str(len(value))
            else:
                # Scalar / already-flat key — pass through as string
                flat[key] = str(value)

        return flat
