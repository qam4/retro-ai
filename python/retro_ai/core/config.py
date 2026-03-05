"""Configuration parser for retro-ai environments.

Supports loading/saving emulator configurations in JSON
and YAML formats. YAML requires the PyYAML package
(optional dependency).
"""

import json
import os
import types
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class RewardConfig:
    """Configuration for a reward system.

    Attributes:
        mode: Reward mode name, e.g. "survival".
        parameters: Mode-specific parameters dict.
    """

    mode: str = "survival"
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmulatorConfig:
    """Configuration for an emulator environment.

    Attributes:
        emulator_type: Emulator id, e.g. "videopac".
        rom_path: Path to the ROM file.
        bios_path: Optional BIOS file path.
        reward: Reward system configuration.
    """

    emulator_type: str
    rom_path: str
    bios_path: Optional[str] = None
    reward: RewardConfig = field(default_factory=RewardConfig)


class ConfigParser:
    """Parse and serialize emulator configurations.

    Supports JSON natively and YAML when PyYAML is
    installed. All methods are static.
    """

    # ---- Deserialization ----

    @staticmethod
    def from_json(path_or_str: str) -> EmulatorConfig:
        """Load config from a JSON file or string.

        If *path_or_str* is an existing file it is read,
        otherwise it is parsed as a JSON string.

        Raises:
            ValueError: On missing fields or bad JSON.
        """
        data = ConfigParser._load_json(path_or_str)
        return ConfigParser._dict_to_config(data)

    @staticmethod
    def from_yaml(path_or_str: str) -> EmulatorConfig:
        """Load config from a YAML file or string.

        Requires ``pyyaml``. Raises ``ImportError`` with
        install instructions if not available.

        Raises:
            ImportError: If PyYAML is missing.
            ValueError: On missing fields or bad YAML.
        """
        yaml = ConfigParser._import_yaml()
        data = ConfigParser._load_yaml(path_or_str, yaml)
        return ConfigParser._dict_to_config(data)

    # ---- Serialization ----

    @staticmethod
    def to_json(
        config: EmulatorConfig,
        path: Optional[str] = None,
    ) -> str:
        """Serialize config to a JSON string.

        If *path* is given, also writes to that file.

        Returns:
            The JSON string.
        """
        data = ConfigParser._config_to_dict(config)
        json_str: str = json.dumps(data, indent=2)
        if path is not None:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(json_str)
        return json_str

    @staticmethod
    def to_yaml(
        config: EmulatorConfig,
        path: Optional[str] = None,
    ) -> str:
        """Serialize config to a YAML string.

        Requires ``pyyaml``. If *path* is given, also
        writes to that file.

        Returns:
            The YAML string.

        Raises:
            ImportError: If PyYAML is missing.
        """
        yaml = ConfigParser._import_yaml()
        data = ConfigParser._config_to_dict(config)
        yaml_str: str = yaml.dump(
            data,
            default_flow_style=False,
            sort_keys=False,
        )
        if path is not None:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(yaml_str)
        return yaml_str

    # ---- Internal helpers ----

    @staticmethod
    def _import_yaml() -> types.ModuleType:
        """Import ``yaml`` or raise a helpful error."""
        try:
            import yaml

            return yaml  # type: ignore[return-value]
        except ImportError:
            raise ImportError(
                "PyYAML is required for YAML support. "
                "Install it with:  pip install pyyaml"
            )

    @staticmethod
    def _load_json(
        path_or_str: str,
    ) -> Dict[str, Any]:
        """Read JSON from file or parse a string."""
        if os.path.isfile(path_or_str):
            with open(path_or_str, "r", encoding="utf-8") as fh:
                try:
                    data: Dict[str, Any] = json.load(fh)
                    return data
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        "Invalid JSON in file " f"'{path_or_str}': {exc}"
                    ) from exc

        try:
            result: Dict[str, Any] = json.loads(path_or_str)
            return result
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON string: {exc}") from exc

    @staticmethod
    def _load_yaml(
        path_or_str: str,
        yaml: types.ModuleType,
    ) -> Dict[str, Any]:
        """Read YAML from file or parse a string."""
        if os.path.isfile(path_or_str):
            with open(path_or_str, "r", encoding="utf-8") as fh:
                try:
                    data = yaml.safe_load(fh)
                except Exception as exc:
                    raise ValueError(
                        "Invalid YAML in file " f"'{path_or_str}': {exc}"
                    ) from exc
        else:
            try:
                data = yaml.safe_load(path_or_str)
            except Exception as exc:
                raise ValueError(f"Invalid YAML string: {exc}") from exc

        if not isinstance(data, dict):
            raise ValueError(
                "YAML config must be a mapping at "
                "the top level, got "
                f"{type(data).__name__}"
            )
        return data

    @staticmethod
    def _dict_to_config(
        data: Dict[str, Any],
    ) -> EmulatorConfig:
        """Convert dict to EmulatorConfig with validation."""
        if not isinstance(data, dict):
            raise ValueError("Config must be a mapping, got " f"{type(data).__name__}")

        # --- required fields ---
        if "emulator_type" not in data:
            raise ValueError(
                "Missing required field 'emulator_type'."
                " Specify the emulator to use"
                " (e.g. 'videopac', 'mo5')."
            )
        if "rom_path" not in data:
            raise ValueError(
                "Missing required field 'rom_path'."
                " Provide the path to the ROM file."
            )

        emulator_type = data["emulator_type"]
        rom_path = data["rom_path"]

        if not isinstance(emulator_type, str) or not emulator_type.strip():
            raise ValueError(
                "'emulator_type' must be a non-empty" f" string, got {emulator_type!r}."
            )
        if not isinstance(rom_path, str) or not rom_path.strip():
            raise ValueError(
                "'rom_path' must be a non-empty" f" string, got {rom_path!r}."
            )

        # --- optional fields ---
        bios_path = data.get("bios_path")
        if bios_path is not None:
            if not isinstance(bios_path, str):
                tp = type(bios_path).__name__
                raise ValueError("'bios_path' must be a string" f" or null, got {tp}.")

        # --- reward sub-config ---
        reward_data = data.get("reward")
        if reward_data is not None:
            if not isinstance(reward_data, dict):
                tp = type(reward_data).__name__
                raise ValueError("'reward' must be a mapping," f" got {tp}.")
            mode = reward_data.get("mode", "survival")
            if not isinstance(mode, str) or not mode.strip():
                raise ValueError(
                    "'reward.mode' must be a non-empty" f" string, got {mode!r}."
                )
            params = reward_data.get("parameters", {})
            if not isinstance(params, dict):
                tp = type(params).__name__
                raise ValueError("'reward.parameters' must be a" f" mapping, got {tp}.")
            reward = RewardConfig(mode=mode, parameters=params)
        else:
            reward = RewardConfig()

        return EmulatorConfig(
            emulator_type=emulator_type,
            rom_path=rom_path,
            bios_path=bios_path,
            reward=reward,
        )

    @staticmethod
    def _config_to_dict(
        config: EmulatorConfig,
    ) -> Dict[str, Any]:
        """Convert EmulatorConfig to a plain dict."""
        return asdict(config)
