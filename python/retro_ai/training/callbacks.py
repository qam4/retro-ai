"""Stable-Baselines3 training callbacks."""

import glob
import os
import time
from typing import Optional

from stable_baselines3.common.callbacks import BaseCallback

from retro_ai.core.logging import StructuredLogger
from retro_ai.training.metrics import MetricsTracker


class MetricsCallback(BaseCallback):
    """Record episode metrics and flush at configured intervals."""

    def __init__(
        self,
        metrics: MetricsTracker,
        log_interval: int,
        logger_inst: Optional[StructuredLogger] = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self._metrics = metrics
        self._log_interval = log_interval
        self._logger = logger_inst
        self._last_log_step = 0
        self._step_start = time.monotonic()

    def _on_step(self) -> bool:
        # Check for completed episodes in locals
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        if dones is not None and infos is not None:
            for i, done in enumerate(dones):
                if done and i < len(infos):
                    info = infos[i]
                    ep_info = info.get("episode")
                    if ep_info:
                        self._metrics.record_episode(
                            reward=ep_info["r"],
                            length=ep_info["l"],
                            info=info,
                        )

        # Flush and log at interval
        if self.num_timesteps - self._last_log_step >= self._log_interval:
            self._metrics.flush_csv()
            if self._logger:
                elapsed = time.monotonic() - self._step_start
                fps = self._log_interval / elapsed if elapsed > 0 else 0
                rolling = self._metrics.rolling_reward()
                self._logger.info(
                    "training_progress",
                    step=self.num_timesteps,
                    rolling_reward=rolling,
                    fps=round(fps, 1),
                )
            self._last_log_step = self.num_timesteps
            self._step_start = time.monotonic()
        return True


class CheckpointCallback(BaseCallback):
    """Save model checkpoints with rolling deletion."""

    def __init__(
        self,
        save_path: str,
        interval: int,
        max_keep: int = 5,
        logger_inst: Optional[StructuredLogger] = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self._save_path = save_path
        self._interval = interval
        self._max_keep = max_keep
        self._logger = logger_inst
        self._last_save_step = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_save_step >= self._interval:
            self._save_checkpoint()
            self._last_save_step = self.num_timesteps
        return True

    def _save_checkpoint(self) -> None:
        os.makedirs(self._save_path, exist_ok=True)
        name = f"model_step_{self.num_timesteps}"
        path = os.path.join(self._save_path, name)
        try:
            self.model.save(path)  # type: ignore[union-attr]
            if self._logger:
                self._logger.info(
                    "checkpoint_saved",
                    step=self.num_timesteps,
                    path=path,
                )
        except Exception as e:
            if self._logger:
                self._logger.warning(
                    f"checkpoint_save_failed: {e}",
                )
            return
        self._prune_old_checkpoints()

    def _prune_old_checkpoints(self) -> None:
        pattern = os.path.join(self._save_path, "model_step_*.zip")
        files = sorted(glob.glob(pattern))
        while len(files) > self._max_keep:
            oldest = files.pop(0)
            try:
                os.remove(oldest)
            except OSError:
                pass


class StagnationCallback(BaseCallback):
    """Warn when rolling average reward plateaus."""

    def __init__(
        self,
        metrics: MetricsTracker,
        threshold_steps: int,
        logger_inst: Optional[StructuredLogger] = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self._metrics = metrics
        self._threshold = threshold_steps
        self._best_rolling: Optional[float] = None
        self._best_step = 0
        self._warned = False

    def _on_step(self) -> bool:
        rolling = self._metrics.rolling_reward()
        if rolling is None:
            return True
        if self._best_rolling is None or rolling > self._best_rolling:
            self._best_rolling = rolling
            self._best_step = self.num_timesteps
            self._warned = False
        elif (
            not self._warned and self.num_timesteps - self._best_step >= self._threshold
        ):
            if logger_inst := getattr(self, "_logger", None):
                logger_inst.warning(
                    "stagnation_detected",
                    steps_since_improvement=(self.num_timesteps - self._best_step),
                    best_rolling=self._best_rolling,
                )
            self._warned = True
        return True
