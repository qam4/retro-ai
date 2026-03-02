#!/usr/bin/env python3
"""Demonstrate switching between reward modes at runtime.

Usage:
    python examples/custom_rewards.py --rom roms/game.bin --bios roms/videopac.bin
"""

import argparse
import random

from retro_ai import BaseEnv


def run_episode(env: BaseEnv, mode: str, max_steps: int = 500) -> float:
    """Run one episode with the given reward mode, return total reward."""
    env.set_reward_mode(mode)
    obs, info = env.reset()
    total = 0.0
    for _ in range(max_steps):
        action = random.randint(0, env.get_action_space()["shape"][0] - 1)
        obs, reward, done, truncated, info = env.step(action)
        total += reward
        if done or truncated:
            break
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Reward mode comparison")
    parser.add_argument("--emulator", default="videopac", choices=["videopac", "mo5"])
    parser.add_argument("--rom", required=True)
    parser.add_argument("--bios", default=None)
    args = parser.parse_args()

    env = BaseEnv(
        emulator_type=args.emulator,
        rom_path=args.rom,
        bios_path=args.bios,
    )

    modes = env.available_reward_modes()
    print(f"Available reward modes: {modes}\n")

    for mode in modes:
        reward = run_episode(env, mode)
        print(f"  {mode:12s} -> total reward: {reward:.2f}")


if __name__ == "__main__":
    main()
