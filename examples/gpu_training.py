#!/usr/bin/env python3
"""GPU-accelerated training with SB3 on CUDA.

Enables mixed precision, reward clipping, and sticky actions
as recommended in docs/training_speedup_ideas.md (D1, B2, B3).

Usage:
    python examples/gpu_training.py \
        --rom roms/game.bin --bios roms/videopac.bin

    # With multiple parallel envs (A2-style throughput):
    python examples/gpu_training.py \
        --rom roms/game.bin --bios roms/videopac.bin --num-envs 8
"""

import argparse

from retro_ai.training.config import AlgorithmConfig, TrainingConfig
from retro_ai.training.pipeline import TrainingPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU-accelerated training")
    parser.add_argument("--emulator", default="videopac", choices=["videopac", "mo5"])
    parser.add_argument("--rom", required=True)
    parser.add_argument("--bios", default=None)
    parser.add_argument("--reward-mode", default="survival")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--output", default="output/gpu_run")
    parser.add_argument("--profile", default=None, help="Game profile name")
    parser.add_argument("--algorithm", default="PPO", choices=["PPO", "DQN"])
    args = parser.parse_args()

    config = TrainingConfig(
        algorithm=AlgorithmConfig(name=args.algorithm),
        total_timesteps=args.timesteps,
        emulator_type=args.emulator,
        rom_path=args.rom,
        bios_path=args.bios,
        reward_mode=args.reward_mode,
        output_dir=args.output,
        game_profile=args.profile,
        # GPU settings (D1)
        device="auto",  # auto-detects CUDA
        mixed_precision=True,
        # Parallel envs for throughput
        num_envs=args.num_envs,
        # Sticky actions (B3) — standard 25% repeat probability
        sticky_actions=0.25,
        # Reward clipping (B2) — standard [-1, +1]
        reward_clip=1.0,
        # Logging
        tensorboard=True,
    )

    pipeline = TrainingPipeline(config)
    model_path = pipeline.run()
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
