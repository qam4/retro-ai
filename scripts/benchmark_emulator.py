#!/usr/bin/env python3
"""Benchmark raw emulator throughput — no neural network, just stepping.

This measures the ceiling: how fast can we feed observations to a model
if the model were infinitely fast?

Usage:
    RETRO_AI_ROM_DIR=roms PYTHONPATH=python:build/ci-linux \
      python3.9 scripts/benchmark_emulator.py --num-envs 8
"""

import argparse
import concurrent.futures
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def make_env():
    """Build one env with the standard preprocessing stack."""
    from retro_ai import BaseEnv
    from retro_ai.core.preprocessing import PreprocessedEnv, PreprocessingPipeline
    from retro_ai.wrappers.gymnasium_wrapper import GymnasiumWrapper
    from retro_ai.training.game_profile import GameProfileRegistry

    registry = GameProfileRegistry()
    profile = registry.load("course_automobile")

    config_dict = {}
    if hasattr(profile, "joystick_index"):
        config_dict["joystick_index"] = profile.joystick_index
    if profile.reward_params:
        config_dict["reward_params"] = profile.reward_params

    base = BaseEnv(
        emulator_type=profile.emulator_type,
        rom_path=profile.rom_path,
        bios_path=profile.bios_path,
        reward_mode="memory",
        config=config_dict or None,
        observation_mode="framebuffer",
        action_mode="discrete",
    )
    pipeline = PreprocessingPipeline(
        grayscale=True, resize=(84, 84), frame_stack=4, frame_skip=4,
    )
    preprocessed = PreprocessedEnv(base, pipeline)
    return GymnasiumWrapper(preprocessed)


def bench_single(steps):
    """Benchmark a single env with random actions."""
    env = make_env()
    obs, _ = env.reset()
    n_actions = env.action_space.n
    t0 = time.monotonic()
    for _ in range(steps):
        action = np.random.randint(n_actions)
        obs, reward, done, truncated, info = env.step(action)
        if done or truncated:
            obs, _ = env.reset()
    wall = time.monotonic() - t0
    env.close()
    return steps / wall


def bench_threaded(num_envs, steps_per_env):
    """Benchmark N envs stepping in parallel using threads (GIL released)."""
    envs = [make_env() for _ in range(num_envs)]
    for env in envs:
        env.reset()

    total_steps = num_envs * steps_per_env

    def step_env(env, n):
        n_actions = env.action_space.n
        for _ in range(n):
            action = np.random.randint(n_actions)
            obs, reward, done, truncated, info = env.step(action)
            if done or truncated:
                env.reset()

    t0 = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_envs) as pool:
        futures = [pool.submit(step_env, env, steps_per_env) for env in envs]
        concurrent.futures.wait(futures)
    wall = time.monotonic() - t0

    for env in envs:
        env.close()

    return total_steps / wall


def main():
    parser = argparse.ArgumentParser(description="Benchmark raw emulator speed")
    parser.add_argument("--steps", type=int, default=5000, help="Steps per env")
    parser.add_argument("--max-envs", type=int, default=32, help="Max parallel envs to test")
    args = parser.parse_args()

    print("=== Raw Emulator Throughput (no neural network) ===\n")

    # Single env baseline
    single_fps = bench_single(args.steps)
    print(f"  1 env (sequential): {single_fps:.0f} FPS")

    results = [{"num_envs": 1, "fps": round(single_fps, 1), "scaling": 1.0}]

    # Scale up
    env_counts = [2, 4, 8, 12, 16, 24, 32]
    env_counts = [n for n in env_counts if n <= args.max_envs]

    for n in env_counts:
        fps = bench_threaded(n, args.steps)
        scaling = fps / single_fps
        print(f"  {n:2d} envs (threaded):  {fps:.0f} FPS  ({scaling:.2f}x)")
        results.append({"num_envs": n, "fps": round(fps, 1), "scaling": round(scaling, 2)})

    # Save
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "steps_per_env": args.steps,
        "results": results,
    }
    out_path = Path("benchmarks/emulator_throughput.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
