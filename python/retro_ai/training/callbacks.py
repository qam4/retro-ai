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
        total_timesteps: int = 0,
        logger_inst: Optional[StructuredLogger] = None,
        verbose: int = 0,
        frame_skip: int = 1,
        num_envs: int = 1,
    ):
        super().__init__(verbose)
        self._metrics = metrics
        self._log_interval = log_interval
        self._total_timesteps = total_timesteps
        self._logger = logger_inst
        self._last_log_step = 0
        self._step_start = time.monotonic()
        self._frame_skip = frame_skip
        self._num_envs = num_envs

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
                        # After auto-reset, the top-level info is from the
                        # new episode.  The terminal step's info is stored
                        # under "terminal_info" by SB3's VecEnv wrappers.
                        terminal_info = info.get("terminal_info", info)
                        self._metrics.record_episode(
                            reward=ep_info["r"],
                            length=ep_info["l"],
                            info=terminal_info,
                        )

        # Flush and log at interval
        if self.num_timesteps - self._last_log_step >= self._log_interval:
            self._metrics.flush_csv()
            if self._logger:
                elapsed = time.monotonic() - self._step_start
                fps = self._log_interval / elapsed if elapsed > 0 else 0
                rolling = self._metrics.rolling_reward()
                rolling_len = self._metrics.rolling_length()
                total = self._total_timesteps
                pct = f" ({100 * self.num_timesteps / total:.0f}%)" if total > 0 else ""
                # emu_fps = total emulator frames/sec across all envs
                # per_env = emulator frames/sec per individual env
                emu_fps = fps * self._frame_skip
                per_env = emu_fps / self._num_envs if self._num_envs > 0 else emu_fps
                reward_str = f"{rolling:.2f}" if rolling is not None else "N/A"
                ep_len_str = f"{rolling_len:.0f}" if rolling_len is not None else "N/A"
                self._logger.info(
                    f"step {self.num_timesteps}/{total}{pct}"
                    f" | reward={reward_str}"
                    f" | ep_len={ep_len_str}"
                    f" | emu_fps={emu_fps:.0f} ({per_env:.0f}/env x{self._num_envs})",
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
                    {"step": self.num_timesteps, "path": path},
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
                    {
                        "steps_since_improvement": (
                            self.num_timesteps - self._best_step
                        ),
                        "best_rolling": self._best_rolling,
                    },
                )
            self._warned = True
        return True


class EpisodeMetricsCallback(BaseCallback):
    """Push domain-specific episode aggregates to TensorBoard.

    Pulls episodes out of an :class:`~retro_ai.training.run_manifest.EpisodeLogger`'s
    in-memory ring buffer, aggregates them over a sliding window via
    :func:`~retro_ai.training.episode_metrics.aggregate`, and records the
    resulting ``{tag: scalar}`` dict through SB3's own logger (so it lands
    in the same TB event file as ``rollout/ep_rew_mean`` et al.).

    Why pull from the logger's ring buffer? The env threads already write
    episodes there as they finish. Maintaining a second buffer inside the
    callback would duplicate memory and invite race conditions — this way
    the logger is the single source of truth.
    """

    def __init__(
        self,
        episode_logger,
        log_interval: int = 10_000,
        window_size: int = 2048,
        min_n: int = 5,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self._episode_logger = episode_logger
        self._log_interval = max(1, log_interval)
        self._window_size = window_size
        self._min_n = min_n
        self._last_log_step = 0
        self._max_level_seen = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_log_step < self._log_interval:
            return True
        self._flush_metrics()
        self._last_log_step = self.num_timesteps
        return True

    def _on_training_end(self) -> None:
        # One more flush so the final window isn't lost.
        self._flush_metrics()

    def _flush_metrics(self) -> None:
        # Imported lazily so the callbacks module doesn't grow a hard
        # dep on episode_metrics for users who don't wire this in.
        from retro_ai.training.episode_metrics import aggregate, infer_max_level

        episodes = self._episode_logger.recent(self._window_size)
        if not episodes:
            return

        # Max level can grow over time; keep track of the largest seen so
        # early empty tags still appear once they start having data.
        self._max_level_seen = max(self._max_level_seen, infer_max_level(episodes))
        metrics = aggregate(episodes, max_level=self._max_level_seen, min_n=self._min_n)
        for tag, value in metrics.items():
            # exclude="stdout" keeps SB3's verbose printer from polluting
            # the console with dozens of tag lines each dump.
            self.logger.record(tag, value, exclude="stdout")
        # Dump under our step so all tags share the same x-coordinate.
        self.logger.dump(step=self.num_timesteps)
