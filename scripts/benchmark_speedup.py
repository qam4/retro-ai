#!/usr/bin/env python3
"""Benchmark training speed for speedup comparisons.

Runs a short training session and records wall-clock time, FPS, and
GPU utilization. Results are appended to benchmarks/speedup_results.json.

Usage:
    # CPU baseline (no optimizations)
    python scripts/benchmark_speedup.py --name cpu-baseline --device cpu

    # GPU only
    python scripts/benchmark_speedup.py --name gpu-only --device cuda

    # GPU + mixed precision
    python scripts/benchmark_speedup.py --name gpu-fp16 --device cuda --mixed-precision
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path


def get_system_info():
    """Collect system/hardware info for the benchmark record."""
    info = {
        "python_version": None,
        "torch_version": None,
        "cuda_available": False,
        "gpu_name": None,
        "gpu_memory_mb": None,
        "cpu_count": os.cpu_count(),
    }
    try:
        import sys

        info["python_version"] = sys.version.split()[0]
    except Exception:
        pass
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_mb"] = round(
                torch.cuda.get_device_properties(0).total_mem / 1e6
            )
    except Exception:
        pass
    return info


def run_benchmark(args):
    """Run a short training session and measure throughput."""
    from retro_ai.training.config import AlgorithmConfig, TrainingConfig
    from retro_ai.training.pipeline import TrainingPipeline

    config = TrainingConfig(
        algorithm=AlgorithmConfig(
            name=args.algorithm,
            learning_rate=3e-4,
            batch_size=64,
        ),
        total_timesteps=args.timesteps,
        game_profile="course_automobile",
        reward_mode="memory",
        policy=args.policy,
        grayscale=True,
        resize=(84, 84),
        frame_stack=4,
        frame_skip=4,
        observation_mode=args.obs_mode,
        num_envs=args.num_envs,
        device=args.device,
        mixed_precision=args.mixed_precision,
        torch_compile=args.torch_compile,
        vec_env_type=args.vec_env_type,
        sticky_actions=args.sticky_actions,
        reward_clip=args.reward_clip,
        output_dir=f"output/bench_{args.name}",
        tensorboard=False,
        checkpoint_interval=args.timesteps + 1,  # no checkpoints during bench
        log_interval=1000,
    )

    print(f"=== Benchmark: {args.name} ===")
    print(f"  device={args.device}, num_envs={args.num_envs}, "
          f"mixed_precision={args.mixed_precision}, vec_env={args.vec_env_type}, "
          f"obs={args.obs_mode}, policy={args.policy}")
    print(f"  timesteps={args.timesteps}, algorithm={args.algorithm}")
    print()

    pipeline = TrainingPipeline(config)

    t0 = time.monotonic()
    pipeline.run()
    wall_clock = time.monotonic() - t0

    fps = args.timesteps / wall_clock

    result = {
        "name": args.name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "wall_clock_seconds": round(wall_clock, 2),
        "fps": round(fps, 1),
        "total_timesteps": args.timesteps,
        "config": {
            "algorithm": args.algorithm,
            "device": args.device,
            "num_envs": args.num_envs,
            "mixed_precision": args.mixed_precision,
            "torch_compile": args.torch_compile,
            "vec_env_type": args.vec_env_type,
            "obs_mode": args.obs_mode,
            "policy": args.policy,
            "sticky_actions": args.sticky_actions,
            "reward_clip": args.reward_clip,
        },
        "system": get_system_info(),
    }

    print(f"\n=== Results: {args.name} ===")
    print(f"  Wall clock: {wall_clock:.1f}s")
    print(f"  FPS: {fps:.1f}")

    # Append to results file
    results_path = Path("benchmarks/speedup_results.json")
    existing = []
    if results_path.exists():
        with open(results_path) as f:
            existing = json.load(f)
    existing.append(result)
    with open(results_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"  Saved to {results_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Benchmark training speed")
    parser.add_argument("--name", required=True, help="Name for this benchmark run")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument("--sticky-actions", type=float, default=0.0)
    parser.add_argument("--reward-clip", type=float, default=0.0)
    parser.add_argument("--timesteps", type=int, default=10000)
    parser.add_argument("--vec-env-type", default="subproc", choices=["subproc", "threaded"])
    parser.add_argument("--obs-mode", default="framebuffer", choices=["framebuffer", "ram"])
    parser.add_argument("--policy", default="CnnPolicy", choices=["CnnPolicy", "MlpPolicy"])
    parser.add_argument("--algorithm", default="PPO", choices=["PPO", "DQN", "SBX_PPO", "SBX_DQN"])
    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
