"""retro-ai: Reinforcement learning framework for retro game emulators."""

__version__ = "0.1.0"


# ------------------------------------------------------------------
# Exception hierarchy
# ------------------------------------------------------------------


class RetroAIError(Exception):
    """Base exception for all retro-ai errors."""


class InitializationError(RetroAIError):
    """Raised when an emulator fails to initialize (bad ROM, missing BIOS, etc.)."""


class InvalidActionError(RetroAIError):
    """Raised when an invalid action is passed to step()."""


class StateError(RetroAIError):
    """Raised on save/load state failures (corrupted snapshot, wrong emulator, etc.)."""


class ConfigurationError(RetroAIError):
    """Raised when an emulator configuration is invalid."""


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

from retro_ai.envs.base_env import BaseEnv  # noqa: E402

__all__ = [
    "BaseEnv",
    "RetroAIError",
    "InitializationError",
    "InvalidActionError",
    "StateError",
    "ConfigurationError",
]
