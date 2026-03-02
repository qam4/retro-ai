#!/usr/bin/env python3
"""Basic training loop without any RL framework.

Usage:
    python examples/basic_training.py --rom roms/game.bin --bios roms/videopac.bin
"""

import argparse
import random

from retro_ai import BaseEnv


def random_agent(env: BaseEnv, episodes: int = 5, max_steps: int = 1000) -> None:
    """Run a random agent for a few episodes and print stats."""
    for ep in range(episodes):
        obs, info = env.reset(seed=ep)
        total_reward = 0.0
        for step in range(max_steps):
            action = random.randint(0, env.get_action_space()["shape"][0] - 1)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            if done or truncated:
                break
        print(f"Episode {ep + 1}: steps={step + 1}, reward={total_reward:.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Basic retro-ai training loop")
    parser.add_argument("--emulator", default="videopac", choices=["videopac", "mo5"])
    parser.add_argument("--rom", required=True, help="Path to ROM file")
    parser.add_argument("--bios", default=None, help="Path to BIOS file (Videopac)")
    parser.add_argument("--reward-mode", default="survival")
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    env = BaseEnv(
        emulator_type=args.emulator,
        rom_path=args.rom,
        bios_path=args.bios,
        reward_mode=args.reward_mode,
    )
    print(f"Observation space: {env.get_observation_space()}")
    print(f"Action space: {env.get_action_space()}")
    print(f"Reward modes: {env.available_reward_modes()}")
    print()

    random_agent(env, episodes=args.episodes)


if __name__ == "__main__":
    main()
