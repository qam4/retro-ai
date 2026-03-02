#!/usr/bin/env python3
"""Run multiple emulators side by side.

Usage:
    python examples/multi_emulator.py \
        --videopac-rom roms/game.bin --videopac-bios roms/videopac.bin \
        --mo5-rom roms/mo5game.k7
"""

import argparse
import random

from retro_ai import BaseEnv


def run_random(env: BaseEnv, label: str, steps: int = 200) -> None:
    """Run random actions and print a summary."""
    obs, _ = env.reset()
    total_reward = 0.0
    for s in range(steps):
        action = random.randint(0, env.get_action_space()["shape"][0] - 1)
        obs, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        if done or truncated:
            break
    print(f"  [{label}] steps={s + 1}, reward={total_reward:.1f}, obs_shape={obs.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-emulator demo")
    parser.add_argument("--videopac-rom", default=None)
    parser.add_argument("--videopac-bios", default=None)
    parser.add_argument("--mo5-rom", default=None)
    args = parser.parse_args()

    envs = []
    if args.videopac_rom:
        envs.append(
            (
                "Videopac",
                BaseEnv("videopac", args.videopac_rom, bios_path=args.videopac_bios),
            )
        )
    if args.mo5_rom:
        envs.append(("MO5", BaseEnv("mo5", args.mo5_rom)))

    if not envs:
        print("Provide at least one ROM. See --help.")
        return

    for label, env in envs:
        print(f"\n{label}:")
        print(f"  obs_space = {env.get_observation_space()}")
        print(f"  act_space = {env.get_action_space()}")
        run_random(env, label)


if __name__ == "__main__":
    main()
