"""Game profile system: GameProfile, StartupSequence, and registry."""

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import yaml

from retro_ai import ConfigurationError

# ---------------------------------------------------------------------------
# Data classes (Task 2.1)
# ---------------------------------------------------------------------------


@dataclass
class StartupAction:
    """A single action in a startup sequence."""

    action: int  # discrete action index
    frames: int = 1  # hold for N frames


@dataclass
class StartupSequence:
    """Ordered list of actions to reach gameplay from boot."""

    actions: List[StartupAction] = field(default_factory=list)
    post_delay_frames: int = 60  # wait after sequence completes
    random_noop_max: int = 0  # max random noop frames after startup (0 = disabled)


@dataclass
class GameProfile:
    """Game-specific configuration."""

    name: str  # e.g. "satellite_attack"
    emulator_type: str  # "videopac" or "mo5"
    rom_path: str
    bios_path: Optional[str] = None
    display_name: str = ""  # human-readable
    action_count: Optional[int] = None  # override action space size
    reward_mode: str = "survival"
    reward_params: Dict[str, Any] = field(default_factory=dict)
    startup_sequence: Optional[StartupSequence] = None
    # Preprocessing defaults
    grayscale: bool = True
    resize: Optional[Tuple[int, int]] = (84, 84)
    frame_stack: int = 4
    frame_skip: int = 4
    frame_skip_rationale: Optional[str] = None
    resize_rationale: Optional[str] = None
    # Observation mode
    observation_mode: Optional[str] = None  # "framebuffer" (default) or "ram"
    # Controller config
    joystick_index: int = 0

    # -- Serialization helpers ------------------------------------------------

    @staticmethod
    def _deserialize(data: dict) -> "GameProfile":
        """Build a GameProfile from a plain dict."""
        data = dict(data)  # shallow copy

        # Nested StartupSequence / StartupAction
        seq = data.get("startup_sequence")
        if isinstance(seq, dict):
            actions = [
                StartupAction(**a) if isinstance(a, dict) else a
                for a in seq.get("actions", [])
            ]
            data["startup_sequence"] = StartupSequence(
                actions=actions,
                post_delay_frames=seq.get("post_delay_frames", 60),
            )

        # YAML/JSON deserializes tuples as lists
        resize = data.get("resize")
        if isinstance(resize, list):
            data["resize"] = tuple(resize)

        # Resolve $RETRO_AI_ROM_DIR in rom_path / bios_path
        rom_dir = os.environ.get("RETRO_AI_ROM_DIR", "")
        for key in ("rom_path", "bios_path"):
            val = data.get(key)
            if isinstance(val, str) and val.startswith("$RETRO_AI_ROM_DIR"):
                data[key] = val.replace("$RETRO_AI_ROM_DIR", rom_dir, 1)

        # Validate reward_params before constructing the profile
        reward_params = data.get("reward_params", {})
        if reward_params:
            GameProfile._validate_reward_params(reward_params)

        return GameProfile(**data)

    @staticmethod
    def _validate_reward_params(reward_params: Dict[str, Any]) -> None:
        """Validate reward_params values. Raises ConfigurationError on invalid input."""
        if "screen_region" in reward_params:
            region = reward_params["screen_region"]
            for key in ("x", "y", "width", "height"):
                val = region.get(key)
                if val is not None and (not isinstance(val, int) or val < 0):
                    raise ConfigurationError(
                        f"screen_region.{key} must be a "
                        f"non-negative integer, got {val!r}"
                    )
        if "score_addresses" in reward_params:
            for i, entry in enumerate(reward_params["score_addresses"]):
                addr = entry.get("address", 0)
                if not isinstance(addr, int) or addr < 0 or addr > 65535:
                    raise ConfigurationError(
                        f"score_addresses[{i}].address must be "
                        f"in [0, 65535], got {addr!r}"
                    )
                nb = entry.get("num_bytes", 1)
                if nb not in (1, 2, 4):
                    raise ConfigurationError(
                        f"score_addresses[{i}].num_bytes must be 1, 2, or 4, got {nb!r}"
                    )

    @staticmethod
    def from_yaml(path: str) -> "GameProfile":
        """Load a GameProfile from a YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return GameProfile._deserialize(data)

    @staticmethod
    def from_json(path: str) -> "GameProfile":
        """Load a GameProfile from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return GameProfile._deserialize(data)

    def to_dict(self) -> dict:
        """Serialize to a plain dict (tuples become lists for YAML/JSON)."""
        data = asdict(self)
        if data.get("resize") is not None:
            data["resize"] = list(data["resize"])
        return data


# ---------------------------------------------------------------------------
# Registry (Task 2.2)
# ---------------------------------------------------------------------------

_DEFAULT_PROFILE_DIR = os.path.join(os.getcwd(), "game_profiles")


class GameProfileRegistry:
    """Discover and load game profiles from directories."""

    def __init__(self, profile_dirs: Optional[List[str]] = None):
        self._dirs: List[str] = []
        if profile_dirs:
            self._dirs.extend(profile_dirs)
        if _DEFAULT_PROFILE_DIR not in self._dirs:
            self._dirs.append(_DEFAULT_PROFILE_DIR)

    # -- public API -----------------------------------------------------------

    def list_profiles(self) -> List[str]:
        """Return names of all discovered profiles."""
        names: List[str] = []
        for d in self._dirs:
            if not os.path.isdir(d):
                continue
            for fname in sorted(os.listdir(d)):
                if fname.endswith((".yaml", ".yml", ".json")):
                    path = os.path.join(d, fname)
                    try:
                        profile = self._load_file(path)
                        if profile.name not in names:
                            names.append(profile.name)
                    except Exception:
                        continue
        return names

    def load(self, name_or_path: str) -> GameProfile:
        """Load a profile by name (lookup in dirs) or by file path."""
        # Direct file path
        if os.path.isfile(name_or_path):
            return self._load_file(name_or_path)

        # Search directories by profile name
        for d in self._dirs:
            if not os.path.isdir(d):
                continue
            for fname in os.listdir(d):
                if not fname.endswith((".yaml", ".yml", ".json")):
                    continue
                path = os.path.join(d, fname)
                try:
                    profile = self._load_file(path)
                    if profile.name == name_or_path:
                        return profile
                except Exception:
                    continue

        raise ConfigurationError(
            f"Game profile '{name_or_path}' not found in {self._dirs}"
        )

    # -- helpers --------------------------------------------------------------

    @staticmethod
    def _load_file(path: str) -> GameProfile:
        if path.endswith(".json"):
            return GameProfile.from_json(path)
        return GameProfile.from_yaml(path)


# ---------------------------------------------------------------------------
# StartupSequenceWrapper (Task 2.4)
# ---------------------------------------------------------------------------


class StartupSequenceWrapper(gym.Wrapper):
    """Gymnasium wrapper that executes a startup sequence on every reset()."""

    def __init__(self, env: gym.Env, sequence: StartupSequence):
        super().__init__(env)
        self._sequence = sequence
        # Detect action space type for action conversion
        self._is_multi_discrete = isinstance(
            self.action_space, gym.spaces.MultiDiscrete
        )
        # Distinguish 5-dim [up,down,left,right,fire] from 3-dim [vert,horiz,fire]
        self._is_joystick = self._is_multi_discrete and len(self.action_space.nvec) == 3

    def _convert_action(self, action_int: int):
        """Convert a discrete action index for the underlying env.

        Handles discrete, multi-discrete [5], and joystick [3,3,2] modes.
        """
        if self._is_joystick:
            # [vertical(3), horizontal(3), fire(2)]
            # 0=neutral, 1=up/right, 2=down/left
            direction_map = {
                0: [0, 0, 0],  # noop
                1: [1, 0, 0],  # up
                2: [2, 0, 0],  # down
                3: [0, 2, 0],  # left
                4: [0, 1, 0],  # right
            }
            return direction_map.get(action_int, [0, 0, 0])
        if self._is_multi_discrete:
            direction_map = {
                0: [0, 0, 0, 0, 0],  # noop
                1: [1, 0, 0, 0, 0],  # up
                2: [0, 1, 0, 0, 0],  # down
                3: [0, 0, 1, 0, 0],  # left
                4: [0, 0, 0, 1, 0],  # right
            }
            if action_int in direction_map:
                return direction_map[action_int]
            return action_int
        return action_int

    def reset(self, **kwargs):  # type: ignore[override]
        obs, info = self.env.reset(**kwargs)
        for action in self._sequence.actions:
            converted = self._convert_action(action.action)
            for _ in range(action.frames):
                obs, _, done, truncated, info = self.env.step(converted)
                if done or truncated:
                    obs, info = self.env.reset(**kwargs)
                    break
        # Post-delay: step with no-op for N frames
        noop = self._convert_action(0)
        for _ in range(self._sequence.post_delay_frames):
            obs, _, done, truncated, info = self.env.step(noop)
            if done or truncated:
                break
        # Random noop frames to break determinism (standard Atari trick)
        if self._sequence.random_noop_max > 0:
            import random

            n = random.randint(0, self._sequence.random_noop_max)
            for _ in range(n):
                obs, _, done, truncated, info = self.env.step(noop)
                if done or truncated:
                    break
        return obs, info
