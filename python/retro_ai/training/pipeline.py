"""Training pipeline orchestrator."""

import os
from pathlib import Path
from typing import Optional

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import CallbackList

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
                total_timesteps=self.config.total_timesteps,
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
                total_timesteps=self.config.total_timesteps,
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

    def _build_env(self):
        """BaseEnv -> PreprocessedEnv -> GymnasiumWrapper [-> SSW]."""
        config_dict = {}
        gp = self._game_profile
        if gp and hasattr(gp, "joystick_index"):
            config_dict["joystick_index"] = gp.joystick_index

        base = BaseEnv(
            emulator_type=self.config.emulator_type,
            rom_path=self.config.rom_path,
            bios_path=self.config.bios_path,
            reward_mode=self.config.reward_mode,
            config=config_dict or None,
        )
        pipeline = PreprocessingPipeline(
            grayscale=self.config.grayscale,
            resize=self.config.resize,
            frame_stack=self.config.frame_stack,
            frame_skip=self.config.frame_skip,
        )
        preprocessed = PreprocessedEnv(base, pipeline)
        env = GymnasiumWrapper(preprocessed)

        # Wrap with startup sequence if profile defines one
        gp = self._game_profile
        if gp and gp.startup_sequence:
            env = StartupSequenceWrapper(env, gp.startup_sequence)

        return env

    def _build_model(self, env):
        """Instantiate the SB3 algorithm from config."""
        algo_cls = ALGORITHM_MAP[self.config.algorithm.name]
        kwargs = {
            "policy": self.config.policy,
            "env": env,
            "learning_rate": self.config.algorithm.learning_rate,
            "batch_size": self.config.algorithm.batch_size,
            "verbose": 0,
            **self.config.algorithm.extra,
        }
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
            algorithm=self.config.algorithm.name,
            total_timesteps=self.config.total_timesteps,
            emulator=self.config.emulator_type,
            reward_mode=self.config.reward_mode,
            reward_params=self.config.reward_params,
            policy=self.config.policy,
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
