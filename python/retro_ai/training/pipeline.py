"""Training pipeline orchestrator."""

import os
from pathlib import Path
from typing import Optional

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from retro_ai import StateError
from retro_ai.core.logging import StructuredLogger
from retro_ai.core.preprocessing import (
    PreprocessedEnv,
    PreprocessingPipeline,
)
from retro_ai.envs.base_env import BaseEnv
from retro_ai.training.callbacks import (
    CheckpointCallback,
    MetricsCallback,
    StagnationCallback,
)
from retro_ai.training.config import (
    TrainingConfig,
    TrainingConfigParser,
    merge_config_with_profile,
)
from retro_ai.training.game_profile import (
    GameProfileRegistry,
    StartupSequenceWrapper,
)
from retro_ai.training.metrics import MetricsTracker
from retro_ai.wrappers.gymnasium_wrapper import GymnasiumWrapper

ALGORITHM_MAP = {
    "PPO": PPO,
    "DQN": DQN,
}


class TrainingPipeline:
    """Orchestrate end-to-end RL training runs."""

    def __init__(
        self,
        config: TrainingConfig,
        logger: Optional[StructuredLogger] = None,
    ):
        self.config = config
        self._logger = logger or StructuredLogger("training")
        self._metrics: Optional[MetricsTracker] = None
        self._game_profile = None

    # ----------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------

    def run(self) -> Path:
        """Execute a full training run. Returns path to saved model."""
        self._resolve_profile()
        TrainingConfigParser.validate(self.config)
        self._check_low_resolution()
        self._log_run_start()

        os.makedirs(self.config.output_dir, exist_ok=True)
        self._save_config_copy()

        self._metrics = MetricsTracker(
            self.config.output_dir,
            rolling_window=self.config.rolling_window,
        )

        env = self._build_env()
        model = self._build_model(env)
        callbacks = self._build_callbacks()

        try:
            model.learn(
                total_timesteps=self._aligned_timesteps(model),
                callback=callbacks,
            )
        except KeyboardInterrupt:
            self._logger.warning("Training interrupted, saving model")
        finally:
            model_path = self._save_model(model)
            self._metrics.flush_csv()
            self._metrics.write_summary()

        return model_path

    def resume(self, checkpoint_path: str) -> Path:
        """Resume training from a checkpoint."""
        self._resolve_profile()
        TrainingConfigParser.validate(self.config)
        self._check_low_resolution()

        os.makedirs(self.config.output_dir, exist_ok=True)
        csv_path = os.path.join(self.config.output_dir, "metrics.csv")

        self._metrics = MetricsTracker(
            self.config.output_dir,
            rolling_window=self.config.rolling_window,
        )
        self._metrics.load_existing(csv_path)

        env = self._build_env()
        model = self._load_checkpoint(checkpoint_path, env)
        callbacks = self._build_callbacks()

        try:
            model.learn(
                total_timesteps=self._aligned_timesteps(model),
                callback=callbacks,
                reset_num_timesteps=False,
            )
        except KeyboardInterrupt:
            self._logger.warning("Training interrupted, saving model")
        finally:
            model_path = self._save_model(model)
            self._metrics.flush_csv()
            self._metrics.write_summary()

        return model_path

    # ----------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------

    def _resolve_profile(self) -> None:
        """Load and merge game profile if configured."""
        if self.config.game_profile:
            registry = GameProfileRegistry()
            self._game_profile = registry.load(self.config.game_profile)
            self.config = merge_config_with_profile(self.config, self._game_profile)

    def _check_low_resolution(self) -> None:
        """Warn if the configured resize resolution is very low."""
        resize = self.config.resize
        if resize is not None:
            h, w = resize
            if h < 42 or w < 42:
                self._logger.warning(
                    "Low observation resolution %s may degrade agent "
                    "performance. Consider using at least 42×42.",
                    resize,
                )

    def _build_env(self):
        """Build environment(s): single or vectorized via SubprocVecEnv.

        When ``num_envs == 1`` a single unwrapped environment is returned
        (no subprocess overhead).  When ``num_envs > 1`` a
        :class:`SubprocVecEnv` with *N* independent env instances is
        returned, each running its own
        ``BaseEnv → PreprocessedEnv → GymnasiumWrapper`` stack.
        """
        num_envs = self.config.num_envs

        def make_env(rank: int):
            """Return a zero-argument callable that builds one env."""

            def _init():
                config_dict = {}
                gp = self._game_profile
                if gp and hasattr(gp, "joystick_index"):
                    config_dict["joystick_index"] = gp.joystick_index
                if gp and gp.reward_params:
                    config_dict["reward_params"] = gp.reward_params

                base = BaseEnv(
                    emulator_type=self.config.emulator_type,
                    rom_path=self.config.rom_path,
                    bios_path=self.config.bios_path,
                    reward_mode=self.config.reward_mode,
                    config=config_dict or None,
                    observation_mode=self.config.observation_mode,
                    action_mode=self.config.action_mode,
                )
                pipeline = PreprocessingPipeline(
                    grayscale=self.config.grayscale,
                    resize=self.config.resize,
                    frame_stack=self.config.frame_stack,
                    frame_skip=self.config.frame_skip,
                    crop=self.config.crop,
                )
                preprocessed = PreprocessedEnv(base, pipeline)
                env = GymnasiumWrapper(preprocessed)

                # Wrap with startup sequence if profile defines one
                if self._game_profile and self._game_profile.startup_sequence:
                    env = StartupSequenceWrapper(
                        env, self._game_profile.startup_sequence
                    )
                # Add survival bonus if configured
                if self.config.survival_bonus > 0:
                    from retro_ai.wrappers.survival_bonus import (
                        SurvivalBonusWrapper,
                    )

                    env = SurvivalBonusWrapper(env, self.config.survival_bonus)
                # Monitor wrapper records episode rewards/lengths for metrics
                env = Monitor(env)
                return env

            return _init

        if num_envs == 1:
            return make_env(0)()  # no subprocess overhead
        else:
            return SubprocVecEnv([make_env(i) for i in range(num_envs)])

    def _build_model(self, env):
        """Instantiate the SB3 algorithm from config."""
        algo_cls = ALGORITHM_MAP[self.config.algorithm.name]

        policy = self.config.policy
        # Auto-select MlpPolicy for RAM observations (Requirement 6.4)
        if self.config.observation_mode == "ram" and policy != "MlpPolicy":
            self._logger.info(
                "policy_override",
                {
                    "reason": "observation_mode is 'ram'; CNN not applicable",
                    "original_policy": policy,
                    "selected_policy": "MlpPolicy",
                },
            )
            policy = "MlpPolicy"

        kwargs = {
            "policy": policy,
            "env": env,
            "learning_rate": self.config.algorithm.learning_rate,
            "batch_size": self.config.algorithm.batch_size,
            "verbose": 0,
            **self.config.algorithm.extra,
        }

        # Mixed precision support (Requirement 8)
        if self.config.mixed_precision:
            import torch

            if torch.cuda.is_available():
                torch.set_float32_matmul_precision("medium")
                kwargs["policy_kwargs"] = kwargs.get("policy_kwargs", {})
                kwargs["policy_kwargs"]["optimizer_kwargs"] = {"fused": True}
                self._logger.info(
                    "mixed_precision_enabled",
                    {"device": "cuda"},
                )
            else:
                self._logger.warning(
                    "mixed_precision_no_cuda",
                    {
                        "message": "mixed_precision enabled but no CUDA GPU "
                        "available, using FP32"
                    },
                )

        # Scale n_steps for vectorized environments so the effective
        # batch size (num_envs × n_steps) stays consistent.
        num_envs = self.config.num_envs
        if num_envs > 1 and self.config.algorithm.name == "PPO":
            base_n_steps = kwargs.get("n_steps", 2048)
            adjusted_n_steps = max(1, base_n_steps // num_envs)
            kwargs["n_steps"] = adjusted_n_steps
            self._logger.info(
                "n_steps_scaled",
                {
                    "num_envs": num_envs,
                    "base_n_steps": base_n_steps,
                    "adjusted_n_steps": adjusted_n_steps,
                    "effective_batch": num_envs * adjusted_n_steps,
                },
            )

        if self.config.tensorboard:
            tb_dir = os.path.join(self.config.output_dir, "tb")
            kwargs["tensorboard_log"] = tb_dir
        return algo_cls(**kwargs)

    def _build_callbacks(self) -> CallbackList:
        """Assemble the callback list."""
        cbs = []
        cbs.append(
            MetricsCallback(
                metrics=self._metrics,
                log_interval=self.config.log_interval,
                total_timesteps=self.config.total_timesteps,
                logger_inst=self._logger,
            )
        )
        ckpt_dir = os.path.join(self.config.output_dir, "checkpoints")
        cbs.append(
            CheckpointCallback(
                save_path=ckpt_dir,
                interval=self.config.checkpoint_interval,
                max_keep=self.config.max_checkpoints,
                logger_inst=self._logger,
            )
        )
        cbs.append(
            StagnationCallback(
                metrics=self._metrics,
                threshold_steps=self.config.stagnation_threshold,
                logger_inst=self._logger,
            )
        )
        return CallbackList(cbs)

    def _aligned_timesteps(self, model) -> int:
        """Round total_timesteps up to the nearest rollout boundary.

        PPO always completes a full rollout buffer before stopping, so
        the progress bar overshoots if total_timesteps isn't a multiple
        of (n_steps * num_envs).  Rounding up avoids confusing displays
        like 'step 2000/500 (400%)'.
        """
        requested = self.config.total_timesteps
        n_steps = getattr(model, "n_steps", None)
        n_envs = getattr(model, "n_envs", 1)
        if n_steps is None:
            return requested
        rollout_size = n_steps * n_envs
        if rollout_size <= 0:
            return requested
        remainder = requested % rollout_size
        if remainder == 0:
            return requested
        aligned = requested + (rollout_size - remainder)
        self._logger.info(
            "timesteps_aligned",
            {
                "requested": requested,
                "aligned": aligned,
                "rollout_size": rollout_size,
            },
        )
        return aligned

    def _save_model(self, model) -> Path:
        """Save the final model."""
        path = os.path.join(self.config.output_dir, "final_model")
        model.save(path)
        return Path(path + ".zip")

    def _save_config_copy(self) -> None:
        """Save a copy of the config for reproducibility."""
        path = os.path.join(self.config.output_dir, "config.yaml")
        TrainingConfigParser.to_yaml(self.config, path)

    def _log_run_start(self) -> None:
        """Log training run parameters."""
        self._logger.info(
            "training_start",
            {
                "algorithm": self.config.algorithm.name,
                "total_timesteps": self.config.total_timesteps,
                "emulator": self.config.emulator_type,
                "reward_mode": self.config.reward_mode,
                "policy": self.config.policy,
            },
        )

    def _load_checkpoint(self, checkpoint_path, env):
        """Load model from checkpoint, with fallback search."""
        algo_cls = ALGORITHM_MAP[self.config.algorithm.name]

        # Try the specified path first
        if os.path.exists(checkpoint_path):
            try:
                return algo_cls.load(checkpoint_path, env=env)
            except Exception:
                pass

        # Fallback: search checkpoints dir for valid ones
        ckpt_dir = os.path.join(self.config.output_dir, "checkpoints")
        if os.path.isdir(ckpt_dir):
            import glob

            files = sorted(
                glob.glob(os.path.join(ckpt_dir, "*.zip")),
                reverse=True,
            )
            for f in files:
                try:
                    return algo_cls.load(f, env=env)
                except Exception:
                    continue

        raise StateError(f"No valid checkpoint found at {checkpoint_path}")
