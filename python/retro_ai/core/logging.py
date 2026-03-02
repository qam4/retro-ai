"""Structured logging utilities for retro-ai.

Provides a thin wrapper around Python's :mod:`logging` module with
structured fields (JSON-style) and convenience helpers for common
emulator events.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Optional


class StructuredFormatter(logging.Formatter):
    """Format log records as JSON lines for machine consumption."""

    def format(self, record: logging.LogRecord) -> str:
        entry: Dict[str, Any] = {
            "ts": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "_structured", None)
        if extra:
            entry["data"] = extra
        if record.exc_info and record.exc_info[1]:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


class StructuredLogger:
    """High-level logging helper for retro-ai components.

    Parameters
    ----------
    name : str
        Logger name (e.g. ``"retro_ai.env"``).
    level : int
        Initial log level.  Defaults to ``logging.INFO``.
    json_output : bool
        If ``True``, use :class:`StructuredFormatter` for JSON lines.
    """

    def __init__(
        self,
        name: str = "retro_ai",
        level: int = logging.INFO,
        json_output: bool = False,
    ) -> None:
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)

        if not self._logger.handlers:
            handler = logging.StreamHandler()
            if json_output:
                handler.setFormatter(StructuredFormatter())
            else:
                handler.setFormatter(
                    logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
                )
            self._logger.addHandler(handler)

        self._episode_start: Optional[float] = None
        self._episode_steps: int = 0
        self._episode_reward: float = 0.0

    # ------------------------------------------------------------------
    # Level helpers
    # ------------------------------------------------------------------

    def set_level(self, level: int) -> None:
        """Change the log level at runtime."""
        self._logger.setLevel(level)

    # ------------------------------------------------------------------
    # Structured log methods
    # ------------------------------------------------------------------

    def _log(
        self,
        level: int,
        msg: str,
        data: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        record = self._logger.makeRecord(
            self._logger.name,
            level,
            "(retro_ai)",
            0,
            msg,
            (),
            kwargs.get("exc_info"),
        )
        if data:
            record._structured = data  # type: ignore[attr-defined]
        self._logger.handle(record)

    def info(self, msg: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Log at INFO level."""
        self._log(logging.INFO, msg, data)

    def debug(self, msg: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Log at DEBUG level."""
        self._log(logging.DEBUG, msg, data)

    def warning(self, msg: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Log at WARNING level."""
        self._log(logging.WARNING, msg, data)

    def error(
        self,
        msg: str,
        data: Optional[Dict[str, Any]] = None,
        exc_info: bool = False,
    ) -> None:
        """Log at ERROR level, optionally with traceback."""
        self._log(logging.ERROR, msg, data, exc_info=exc_info)

    # ------------------------------------------------------------------
    # Environment lifecycle events
    # ------------------------------------------------------------------

    def log_env_created(
        self,
        emulator: str,
        rom: str,
        reward_mode: str,
    ) -> None:
        """Log environment creation at INFO."""
        self.info(
            "Environment created",
            {
                "emulator": emulator,
                "rom": rom,
                "reward_mode": reward_mode,
            },
        )

    def log_reset(self, seed: Optional[int] = None) -> None:
        """Log episode reset and start tracking metrics."""
        self._episode_start = time.monotonic()
        self._episode_steps = 0
        self._episode_reward = 0.0
        self.info("Episode reset", {"seed": seed})

    def log_step(self, reward: float, done: bool) -> None:
        """Track per-step metrics; log at DEBUG."""
        self._episode_steps += 1
        self._episode_reward += reward
        self.debug(
            "Step",
            {
                "step": self._episode_steps,
                "reward": reward,
                "done": done,
            },
        )

    def log_episode_end(self) -> None:
        """Log episode summary at INFO."""
        elapsed = 0.0
        if self._episode_start is not None:
            elapsed = time.monotonic() - self._episode_start
        fps = self._episode_steps / elapsed if elapsed > 0 else 0.0
        self.info(
            "Episode finished",
            {
                "steps": self._episode_steps,
                "total_reward": round(self._episode_reward, 4),
                "elapsed_s": round(elapsed, 3),
                "fps": round(fps, 1),
            },
        )

    def log_reward_detail(
        self,
        mode: str,
        raw: float,
        final: float,
    ) -> None:
        """Log reward computation details at DEBUG."""
        self.debug(
            "Reward computed",
            {"mode": mode, "raw": raw, "final": final},
        )
