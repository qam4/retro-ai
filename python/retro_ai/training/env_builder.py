"""Shared helper to build the BaseEnv -> preprocessing -> Gymnasium stack.

The three training scripts used to each construct this pipeline inline:

1. Load the GameProfile
2. Build a BaseEnv with its rom/reward config
3. Wrap in PreprocessingPipeline (grayscale/resize/frame_stack/frame_skip)
4. Wrap in PreprocessedEnv (adds frame_maxpool)
5. Wrap in GymnasiumWrapper

Copy-paste drift was inevitable. This module gives every script a single
``build_training_env(profile_name, env_cfg)`` entry point that honors
:class:`~retro_ai.training.run_config.EnvConfig` overrides.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from retro_ai.core.preprocessing import PreprocessedEnv, PreprocessingPipeline
from retro_ai.envs.base_env import BaseEnv
from retro_ai.training.game_profile import GameProfile, GameProfileRegistry
from retro_ai.training.run_config import EnvConfig
from retro_ai.wrappers.gymnasium_wrapper import GymnasiumWrapper


@dataclass
class TrainingEnvStack:
    """The handful of env objects each training script needs to drive.

    ``base`` is the raw :class:`BaseEnv` (for ``read_ram_byte``, ``load_state``,
    ``save_state``). ``gym`` is the fully-wrapped Gymnasium env handed to
    Stable-Baselines3.
    """

    base: BaseEnv
    preprocessed: PreprocessedEnv
    gym: GymnasiumWrapper
    profile: GameProfile


def build_training_env(profile_name: str, env_cfg: EnvConfig) -> TrainingEnvStack:
    """Construct the full training env stack for a given profile.

    Any non-``None`` field on ``env_cfg`` overrides the corresponding field
    on the loaded :class:`GameProfile`.

    Parameters
    ----------
    profile_name : str
        Profile identifier (resolved by :class:`GameProfileRegistry`).
    env_cfg : EnvConfig
        Per-run overrides. ``action_mode`` and ``resize`` always apply;
        ``frame_skip`` / ``frame_stack`` / ``frame_maxpool`` / ``grayscale``
        only apply when they are not ``None``.
    """
    registry = GameProfileRegistry()
    profile = registry.load(profile_name)

    base_config: dict[str, Any] = {}
    if profile.reward_params:
        base_config["reward_params"] = profile.reward_params

    base = BaseEnv(
        emulator_type=profile.emulator_type,
        rom_path=profile.rom_path,
        bios_path=profile.bios_path,
        reward_mode=profile.reward_mode,
        config=base_config or None,
        action_mode=env_cfg.action_mode,
    )

    pipeline = PreprocessingPipeline(
        grayscale=(
            env_cfg.grayscale if env_cfg.grayscale is not None else profile.grayscale
        ),
        resize=env_cfg.resize,
        frame_stack=(
            env_cfg.frame_stack
            if env_cfg.frame_stack is not None
            else profile.frame_stack
        ),
        frame_skip=(
            env_cfg.frame_skip if env_cfg.frame_skip is not None else profile.frame_skip
        ),
    )
    preprocessed = PreprocessedEnv(
        base,
        pipeline,
        frame_maxpool=(
            env_cfg.frame_maxpool
            if env_cfg.frame_maxpool is not None
            else profile.frame_maxpool
        ),
    )
    gym_env = GymnasiumWrapper(preprocessed)

    return TrainingEnvStack(
        base=base,
        preprocessed=preprocessed,
        gym=gym_env,
        profile=profile,
    )


__all__ = ["TrainingEnvStack", "build_training_env"]
