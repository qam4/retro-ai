#!/usr/bin/env python3
"""Quick smoke test for a game profile.

Verifies that the environment can be created, reset, and stepped through
with the training pipeline's full wrapper chain (BaseEnv → PreprocessedEnv
→ GymnasiumWrapper → StartupSequenceWrapper).

Usage:
    RETRO_AI_ROM_DIR=roms PYTHONPATH=build/ci-linux:python \
        python scripts/smoke_test_game.py <profile_name_or_path> [--steps N] [--action-mode MODE]

Examples:
    python scripts/smoke_test_game.py course_automobile
    python scripts/smoke_test_game.py satellite_attack_memory --steps 500
    python scripts/smoke_test_game.py game_profiles/videopac_satellite_attack.yaml --action-mode discrete
"""

import argparse
import sys
import time

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test a game profile")
    parser.add_argument("profile", help="Game profile name or YAML path")
    parser.add_argument(
        "--steps", type=int, default=200, help="Steps to run (default: 200)"
    )
    parser.add_argument(
        "--action-mode",
        default="multi_discrete",
        help="Action mode (default: multi_discrete)",
    )
    parser.add_argument(
        "--num-envs", type=int, default=1, help="Number of parallel envs (default: 1)"
    )
    args = parser.parse_args()

    from retro_ai.training.config import TrainingConfig, TrainingConfigParser
    from retro_ai.training.game_profile import GameProfileRegistry
    from retro_ai.training.pipeline import TrainingPipeline

    # Load profile
    registry = GameProfileRegistry()
    try:
        profile = registry.load(args.profile)
    except Exception as e:
        print(f"FAIL: Could not load profile '{args.profile}': {e}")
        return 1

    print(f"Game:         {profile.display_name or profile.name}")
    print(f"Emulator:     {profile.emulator_type}")
    print(f"Reward mode:  {profile.reward_mode}")
    print(f"Action mode:  {args.action_mode}")
    print(f"Steps:        {args.steps}")
    print()

    # Build a minimal training config
    config = TrainingConfig(
        game_profile=profile.name,
        total_timesteps=args.steps,
        action_mode=args.action_mode,
        num_envs=args.num_envs,
        output_dir="/tmp/retro_ai_smoke",
        grayscale=profile.grayscale,
        resize=profile.resize,
        frame_stack=profile.frame_stack,
        frame_skip=profile.frame_skip,
        reward_mode=profile.reward_mode,
    )

    pipeline = TrainingPipeline(config)
    pipeline._resolve_profile()

    # Build env
    try:
        env = pipeline._build_env()
    except Exception as e:
        print(f"FAIL: Could not build env: {e}")
        return 1

    print(f"Action space: {env.action_space}")
    print(f"Obs space:    {env.observation_space.shape}")
    print()

    # Reset
    try:
        if args.num_envs > 1:
            obs = env.reset()
        else:
            obs, info = env.reset()
    except Exception as e:
        print(f"FAIL: reset() failed: {e}")
        return 1

    obs_shape = obs.shape
    print(f"Reset OK, obs shape: {obs_shape}")

    # Step loop
    total_reward = 0.0
    nonzero_rewards = 0
    episodes = 0
    t0 = time.perf_counter()

    for i in range(args.steps):
        action = env.action_space.sample()
        try:
            if args.num_envs > 1:
                obs, reward, done, info = env.step(action)
                total_reward += reward.sum()
                nonzero_rewards += (reward != 0).sum()
                episodes += done.sum()
            else:
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
                if reward != 0:
                    nonzero_rewards += 1
                if done or truncated:
                    episodes += 1
                    obs, info = env.reset()
        except Exception as e:
            print(f"FAIL: step() failed at step {i}: {e}")
            env.close()
            return 1

    elapsed = time.perf_counter() - t0
    fps = args.steps / elapsed

    env.close()

    print(f"Steps:          {args.steps}")
    print(f"Episodes:       {episodes}")
    print(f"Total reward:   {total_reward:.1f}")
    print(f"Nonzero rewards:{nonzero_rewards}")
    print(f"Wall clock:     {elapsed:.2f}s")
    print(f"FPS:            {fps:.0f}")
    print()
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
