"""Tests for the structured logging module."""

import json
import logging

from retro_ai.core.logging import StructuredFormatter, StructuredLogger


class TestStructuredFormatter:
    def test_json_output(self):
        fmt = StructuredFormatter()
        record = logging.LogRecord(
            "test", logging.INFO, "", 0, "hello", (), None
        )
        line = fmt.format(record)
        data = json.loads(line)
        assert data["msg"] == "hello"
        assert data["level"] == "INFO"

    def test_extra_data(self):
        fmt = StructuredFormatter()
        record = logging.LogRecord(
            "test", logging.DEBUG, "", 0, "step", (), None
        )
        record._structured = {"step": 1}
        line = fmt.format(record)
        data = json.loads(line)
        assert data["data"]["step"] == 1


class TestStructuredLogger:
    def _make(self, **kw):
        logger = StructuredLogger(
            name=f"test_{id(self)}", level=logging.DEBUG, **kw
        )
        return logger

    def test_set_level(self):
        log = self._make()
        log.set_level(logging.WARNING)
        assert log._logger.level == logging.WARNING

    def test_log_env_created(self):
        log = self._make()
        # Should not raise
        log.log_env_created("videopac", "game.bin", "survival")

    def test_episode_lifecycle(self):
        log = self._make()
        log.log_reset(seed=42)
        assert log._episode_steps == 0
        assert log._episode_reward == 0.0

        log.log_step(1.0, False)
        assert log._episode_steps == 1
        assert log._episode_reward == 1.0

        log.log_step(2.0, True)
        assert log._episode_steps == 2
        assert log._episode_reward == 3.0

        # Should not raise
        log.log_episode_end()

    def test_reward_detail(self):
        log = self._make()
        log.log_reward_detail("survival", 1.0, 1.0)
