"""Training configuration dataclasses and parser."""

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Tuple

import yaml

from retro_ai import ConfigurationError


@dataclass
class AlgorithmConfig:
    """RL algorithm selection and hyperparameters."""

    name: str = "PPO"  # "PPO" or "DQN"
    learning_rate: float = 3e-4
    batch_size: int = 64
    extra: Dict[str, Any] = field(default_factory=dict)  # algo-specific kwargs


@dataclass
class TrainingConfig:
    """Complete specification for a training run."""

    # Algorithm
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    total_timesteps: int = 1_000_000

    # Environment (can be overridden by game_profile)
    emulator_type: Optional[str] = None
    rom_path: Optional[str] = None
    bios_path: Optional[str] = None
    reward_mode: str = "survival"
    reward_params: Dict[str, Any] = field(default_factory=dict)
    reward_weights: Optional[Dict[str, float]] = None  # multi-reward blending

    # Preprocessing
    grayscale: bool = True
    resize: Optional[Tuple[int, int]] = (84, 84)  # (H, W)
    frame_stack: int = 4
    frame_skip: int = 4

    # Game profile
    game_profile: Optional[str] = None  # profile name or path

    # Output
    output_dir: str = "output"
    checkpoint_interval: int = 50_000  # steps between checkpoints
    max_checkpoints: int = 5  # rolling window
    log_interval: int = 1_000  # steps between metric flushes

    # Metrics
    tensorboard: bool = False
    rolling_window: int = 100  # episodes for rolling average
    stagnation_threshold: int = 200_000  # steps without improvement -> warning

    # Policy network
    policy: str = "CnnPolicy"


class TrainingConfigParser:
    """Serialize, deserialize, and validate TrainingConfig objects."""

    @staticmethod
    def from_dict(data: dict) -> TrainingConfig:
        """Parse a dict into TrainingConfig.

        Handles nested AlgorithmConfig and list-to-tuple conversion.
        """
        data = dict(data)  # shallow copy
        # Convert nested algorithm dict to AlgorithmConfig
        algo = data.get("algorithm")
        if isinstance(algo, dict):
            data["algorithm"] = AlgorithmConfig(**algo)
        # YAML/JSON deserializes tuples as lists — convert resize back
        resize = data.get("resize")
        if isinstance(resize, list):
            data["resize"] = tuple(resize)
        return TrainingConfig(**data)

    @staticmethod
    def from_yaml(path: str) -> TrainingConfig:
        """Load a YAML file and parse it into a TrainingConfig."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return TrainingConfigParser.from_dict(data)

    @staticmethod
    def from_json(path: str) -> TrainingConfig:
        """Load a JSON file and parse it into a TrainingConfig."""
        with open(path, "r") as f:
            data = json.load(f)
        return TrainingConfigParser.from_dict(data)

    @staticmethod
    def to_dict(config: TrainingConfig) -> dict:
        """Serialize a TrainingConfig to a plain dict.

        Tuples are converted to lists for JSON/YAML compatibility.
        """
        data = asdict(config)
        # Convert tuples to lists for safe YAML/JSON serialization
        if data.get("resize") is not None:
            data["resize"] = list(data["resize"])
        return data

    @staticmethod
    def to_yaml(config: TrainingConfig, path: str) -> None:
        """Save a TrainingConfig to a YAML file."""
        data = TrainingConfigParser.to_dict(config)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

    @staticmethod
    def to_json(config: TrainingConfig, path: str) -> None:
        """Save a TrainingConfig to a JSON file."""
        data = TrainingConfigParser.to_dict(config)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def validate(config: TrainingConfig) -> None:
        """Validate a TrainingConfig, raising ConfigurationError on problems.

        If ``game_profile`` is set, emulator_type/rom_path validation is
        skipped because those fields will be filled by the merge logic later.
        """
        if not config.game_profile:
            if config.emulator_type is None:
                raise ConfigurationError("emulator_type")
            if config.rom_path is None:
                raise ConfigurationError("rom_path")

        if config.algorithm.name not in {"PPO", "DQN"}:
            raise ConfigurationError("algorithm.name")

        if config.total_timesteps <= 0:
            raise ConfigurationError("total_timesteps")

        if config.checkpoint_interval <= 0:
            raise ConfigurationError("checkpoint_interval")

        if config.resize is not None:
            if config.resize[0] <= 0 or config.resize[1] <= 0:
                raise ConfigurationError("resize")


# ---------------------------------------------------------------------------
# Config merge with GameProfile (Task 2.3)
# ---------------------------------------------------------------------------

# TrainingConfig defaults for merge-eligible fields
_TC_DEFAULTS: Dict[str, Any] = {
    "emulator_type": None,
    "rom_path": None,
    "bios_path": None,
    "reward_mode": "survival",
    "reward_params": {},
    "grayscale": True,
    "resize": (84, 84),
    "frame_stack": 4,
    "frame_skip": 4,
}


def merge_config_with_profile(
    config: TrainingConfig,
    profile: Any,
) -> TrainingConfig:
    """Merge a TrainingConfig with a GameProfile.

    Precedence: explicit TrainingConfig values > GameProfile values > defaults.
    A field is "explicitly set" if it differs from the dataclass default or is
    not None for Optional fields.

    Returns a *new* TrainingConfig with merged values.
    """
    from dataclasses import replace

    merged = {}
    for field_name, default_val in _TC_DEFAULTS.items():
        tc_val = getattr(config, field_name)
        gp_val = getattr(profile, field_name, None)

        # For Optional fields (default is None): explicitly set means not None
        if default_val is None:
            if tc_val is not None:
                merged[field_name] = tc_val
            elif gp_val is not None:
                merged[field_name] = gp_val
            # else: stays None (default)
        else:
            # For fields with non-None defaults: explicitly set means != default
            if tc_val != default_val:
                merged[field_name] = tc_val
            elif gp_val is not None and gp_val != default_val:
                merged[field_name] = gp_val
            # else: keep TrainingConfig value (the default)

    return replace(config, **merged)
