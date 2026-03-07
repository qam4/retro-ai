"""Tests for mixed precision support in TrainingPipeline._build_model().

Covers Requirements 8.1, 8.2, 8.3, 8.4.
"""

from unittest.mock import MagicMock, patch

import pytest

from retro_ai.training.config import TrainingConfig
from retro_ai.training.pipeline import TrainingPipeline


class TestMixedPrecisionNoCuda:
    """When mixed_precision=True but no CUDA GPU, log warning and use FP32."""

    def test_warning_logged_when_no_cuda(self):
        config = TrainingConfig(
            emulator_type="videopac",
            rom_path="/dummy.bin",
            mixed_precision=True,
        )
        pipeline = TrainingPipeline(config)
        pipeline._logger = MagicMock()

        # Create a minimal mock env that SB3 won't choke on
        mock_env = MagicMock()
        mock_env.observation_space = MagicMock()
        mock_env.action_space = MagicMock()

        with patch("retro_ai.training.pipeline.ALGORITHM_MAP") as algo_map:
            mock_algo_cls = MagicMock()
            algo_map.__getitem__ = MagicMock(return_value=mock_algo_cls)

            with patch("torch.cuda.is_available", return_value=False):
                pipeline._build_model(mock_env)

        # Verify warning was logged
        pipeline._logger.warning.assert_called_once_with(
            "mixed_precision_no_cuda",
            {
                "message": "mixed_precision enabled but no CUDA GPU "
                "available, using FP32"
            },
        )

        # Verify fused optimizer was NOT set
        call_kwargs = mock_algo_cls.call_args[1]
        assert "policy_kwargs" not in call_kwargs

    def test_no_warning_when_mixed_precision_disabled(self):
        config = TrainingConfig(
            emulator_type="videopac",
            rom_path="/dummy.bin",
            mixed_precision=False,
        )
        pipeline = TrainingPipeline(config)
        pipeline._logger = MagicMock()

        mock_env = MagicMock()

        with patch("retro_ai.training.pipeline.ALGORITHM_MAP") as algo_map:
            mock_algo_cls = MagicMock()
            algo_map.__getitem__ = MagicMock(return_value=mock_algo_cls)
            pipeline._build_model(mock_env)

        pipeline._logger.warning.assert_not_called()


class TestMixedPrecisionWithCuda:
    """When mixed_precision=True and CUDA available, configure AMP."""

    def test_cuda_path_sets_precision_and_fused_optimizer(self):
        config = TrainingConfig(
            emulator_type="videopac",
            rom_path="/dummy.bin",
            mixed_precision=True,
        )
        pipeline = TrainingPipeline(config)
        pipeline._logger = MagicMock()

        mock_env = MagicMock()

        with patch("retro_ai.training.pipeline.ALGORITHM_MAP") as algo_map:
            mock_algo_cls = MagicMock()
            algo_map.__getitem__ = MagicMock(return_value=mock_algo_cls)

            with patch("torch.cuda.is_available", return_value=True) as _, \
                 patch("torch.set_float32_matmul_precision") as mock_precision:
                pipeline._build_model(mock_env)

                mock_precision.assert_called_once_with("medium")

        # Verify fused optimizer was set
        call_kwargs = mock_algo_cls.call_args[1]
        assert "policy_kwargs" in call_kwargs
        assert call_kwargs["policy_kwargs"]["optimizer_kwargs"] == {"fused": True}

        # Verify info log
        pipeline._logger.info.assert_any_call(
            "mixed_precision_enabled",
            {"device": "cuda"},
        )
