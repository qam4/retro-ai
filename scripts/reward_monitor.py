#!/usr/bin/env python3
"""Monitor reward signals frame-by-frame during an episode.

Runs an episode (agent or random) and logs every reward event with
context: score, lives, Y position, action taken. Outputs a CSV for
analysis and prints a summary.

Usage:
    # Random agent
    python scripts/reward_monitor.py --profile yeti --mode random

    # Trained agent
    python scripts/reward_monitor.py --profile yeti --mode agent \
        --model output/yeti/ppo_5M/final_model.zip

    # Scripted actions (go right)
    python scripts/reward_monitor.py --profile yeti --mode script \
        --script "right"
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np


def make_env(profile_name, action_mode="joystick"):
    """Build a single env from a game profile name."""
    from retro_ai.envs.base_env import BaseEnv
    from retro_ai.core.preprocessing import PreprocessedEnv, PreprocessingPipeline
    from retro_ai.wrappers.gymnasium_wrapper import GymnasiumWrapper
    from retro_ai.training.game_profile import GameProfileRegistry

    registry = GameProfileRegistry()
    profile = registry.load(profile_name)

    config_dict = {}
    if hasattr(profile, "joystick_index"):
        config_dict["joystick_index"] = getattr(profile, "joystick_index", 0)
    if profile.reward_params:
        config_dict["reward_params"] = profile.reward_params

    base = BaseEnv(
        emulator_type=profile.emulator_type,
        rom_path=profile.rom_path,
        bios_path=getattr(profile, "bios_path", None),
        reward_mode=profile.reward_mode,
        config=config_dict or None,
        action_mode=action_mode,
    )

    pipeline = PreprocessingPipeline(
        grayscale=profile.grayscale,
        resize=profile.resize,
        frame_stack=profile.frame_stack,
        frame_skip=profile.frame_skip,
    )
    preprocessed = PreprocessedEnv(base, pipeline,
                                   frame_maxpool=profile.frame_maxpool)
    env = GymnasiumWrapper(preprocessed)
    return env, base, profile


# RAM addresses for Yeti (could be generalized via profile)
YETI_ADDRS = {
    "score_hi": 11093,
    "score_lo": 11094,
    "bonus_hi": 11010,
    "bonus_lo": 11011,
    "lives": 11095,
    "y_pos": 11089,
    "x_pos": 11090,
}


def read_game_state(base_env):
    """Read game state from RAM."""
    iface = base_env._interface
    state = {}
    for name, addr in YETI_ADDRS.items():
        state[name] = iface.read_ram_byte(addr)
    # Compute derived values
    state["score"] = (state["score_hi"] << 8) | state["score_lo"]
    state["bonus"] = (state["bonus_hi"] << 8) | state["bonus_lo"]
    return state


SCRIPTS = {
    "right": lambda step: np.array([0, 1, 0]),       # go right
    "right_up": lambda step: np.array([1, 1, 0]),     # climb right
    "noop": lambda step: np.array([0, 0, 0]),         # stand still
    "random": None,                                    # handled separately
}


def run_episode(env, base_env, mode="random", model=None, script="right",
                max_steps=5000):
    """Run one episode, logging reward and game state each step."""
    obs, info = env.reset()
    state = read_game_state(base_env)

    rows = []
    total_reward = 0
    prev_score = state["score"]
    prev_bonus = state["bonus"]
    prev_y = state["y_pos"]

    for step in range(max_steps):
        # Choose action
        if mode == "agent" and model is not None:
            action, _ = model.predict(obs, deterministic=True)
        elif mode == "script" and script in SCRIPTS:
            action = SCRIPTS[script](step)
        else:
            action = env.action_space.sample()

        obs, reward, done, truncated, info = env.step(action)
        state = read_game_state(base_env)

        # Decompose reward changes
        score_delta = state["score"] - prev_score
        bonus_delta = state["bonus"] - prev_bonus
        y_delta = state["y_pos"] - prev_y

        row = {
            "step": step + 1,
            "action": action.tolist() if hasattr(action, "tolist") else action,
            "reward": round(reward, 4),
            "total_reward": round(total_reward + reward, 4),
            "score": state["score"],
            "score_delta": score_delta,
            "bonus": state["bonus"],
            "bonus_delta": bonus_delta,
            "lives": state["lives"],
            "y_pos": state["y_pos"],
            "y_delta": y_delta,
            "x_pos": state["x_pos"],
            "done": done,
        }
        rows.append(row)
        total_reward += reward

        # Print significant events
        if score_delta != 0:
            print(f"  Step {step+1}: SCORE +{score_delta} (total={state['score']}), reward={reward:.3f}")
        if y_delta < -5:
            print(f"  Step {step+1}: CLIMBED y_delta={y_delta} (Y={state['y_pos']})")
        if y_delta > 5:
            print(f"  Step {step+1}: FELL y_delta={y_delta} (Y={state['y_pos']})")
        if state["lives"] < rows[-2]["lives"] if len(rows) > 1 else False:
            print(f"  Step {step+1}: LIFE LOST (lives={state['lives']})")

        prev_score = state["score"]
        prev_bonus = state["bonus"]
        prev_y = state["y_pos"]

        if done or truncated:
            print(f"  Step {step+1}: EPISODE END (reward={reward:.3f}, total={total_reward:.1f})")
            break

    return rows, total_reward


def main():
    parser = argparse.ArgumentParser(description="Monitor reward signals")
    parser.add_argument("--profile", required=True, help="Game profile name")
    parser.add_argument("--mode", default="random",
                        choices=["random", "agent", "script"])
    parser.add_argument("--model", help="Path to trained model (for agent mode)")
    parser.add_argument("--script", default="right",
                        choices=list(SCRIPTS.keys()))
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--output", help="CSV output path")
    parser.add_argument("--action-mode", default="joystick")
    args = parser.parse_args()

    env, base_env, profile = make_env(args.profile, args.action_mode)

    model = None
    if args.mode == "agent":
        from stable_baselines3 import PPO
        model = PPO.load(args.model, env=env)

    print(f"=== Reward Monitor: {profile.name} ({args.mode}) ===")
    rows, total = run_episode(env, base_env, mode=args.mode, model=model,
                              script=args.script, max_steps=args.max_steps)

    print(f"\n=== Summary ===")
    print(f"  Steps: {len(rows)}")
    print(f"  Total reward: {total:.2f}")
    rewards = [r["reward"] for r in rows]
    nonzero = [r for r in rewards if abs(r) > 0.001]
    print(f"  Non-zero reward steps: {len(nonzero)}/{len(rows)}")
    if nonzero:
        print(f"  Reward range: {min(nonzero):.4f} to {max(nonzero):.4f}")

    # Save CSV
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Saved to {args.output}")

    env.close()


if __name__ == "__main__":
    main()
