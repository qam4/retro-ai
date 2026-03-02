#!/usr/bin/env python3
"""Train an agent using Gymnasium + Stable-Baselines3.

Requirements:
    pip install gymnasium stable-baselines3

Usage:
    python examples/gymnasium_integration.py \
        --rom roms/game.bin --bios roms/videopac.bin
"""

import argparse

from retro_ai import BaseEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Gymnasium + SB3 training")
    parser.add_argument("--emulator", default="videopac", choices=["videopac", "mo5"])
    parser.add_argument("--rom", required=True)
    parser.add_argument("--bios", default=None)
    parser.add_argument("--reward-mode", default="survival")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--save", default="retro_ai_agent")
    args = parser.parse_args()

    # Late imports so the script fails fast with a clear message
    try:
        from retro_ai.wrappers.gymnasium_wrapper import GymnasiumWrapper
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError as e:
        raise SystemExit(
            f"Missing dependency: {e}\n"
            "Install with: pip install gymnasium stable-baselines3"
        )

    def make_env():
        base = BaseEnv(
            emulator_type=args.emulator,
            rom_path=args.rom,
            bios_path=args.bios,
            reward_mode=args.reward_mode,
        )
        return GymnasiumWrapper(base, render_mode="rgb_array")

    env = DummyVecEnv([make_env])

    model = PPO("CnnPolicy", env, verbose=1, n_steps=128, batch_size=64)
    print(f"Training for {args.timesteps} timesteps...")
    model.learn(total_timesteps=args.timesteps)
    model.save(args.save)
    print(f"Model saved to {args.save}")

    env.close()


if __name__ == "__main__":
    main()
