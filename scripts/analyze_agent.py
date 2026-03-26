#!/usr/bin/env python3
"""Analyze a trained agent's behavior on Yeti.

Generates:
  1. Trajectory plot (path on game background)
  2. Position heatmap (20 episodes)
  3. Action distribution pie chart
  4. Reward timeline
  5. Summary stats

Usage:
    python scripts/analyze_agent.py --model output/yeti/ppo_5M/final_model.zip
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def make_env():
    from retro_ai.envs.base_env import BaseEnv
    from retro_ai.core.preprocessing import PreprocessedEnv, PreprocessingPipeline
    from retro_ai.wrappers.gymnasium_wrapper import GymnasiumWrapper
    from retro_ai.training.game_profile import GameProfileRegistry

    registry = GameProfileRegistry()
    profile = registry.load("yeti")
    config_dict = {}
    if profile.reward_params:
        config_dict["reward_params"] = profile.reward_params
    base = BaseEnv(
        emulator_type=profile.emulator_type,
        rom_path=profile.rom_path,
        reward_mode=profile.reward_mode,
        config=config_dict or None,
        action_mode="joystick",
    )
    pipeline = PreprocessingPipeline(
        grayscale=profile.grayscale, resize=profile.resize,
        frame_stack=profile.frame_stack, frame_skip=profile.frame_skip,
    )
    preprocessed = PreprocessedEnv(base, pipeline, frame_maxpool=profile.frame_maxpool)
    env = GymnasiumWrapper(preprocessed)
    return env, base


YETI_ADDRS = {
    "score_hi": 11093, "score_lo": 11094,
    "bonus_hi": 11010, "bonus_lo": 11011,
    "lives": 11095, "y_pos": 11089, "x_pos": 11090,
}

ACTION_NAMES = {
    (0,0,0): "noop", (1,0,0): "up", (2,0,0): "down",
    (0,1,0): "right", (0,2,0): "left",
    (0,0,1): "jump", (1,0,1): "up+jump", (2,0,1): "down+jump",
    (0,1,1): "right+jump", (0,2,1): "left+jump",
    (1,1,0): "up+right", (1,2,0): "up+left",
    (2,1,0): "down+right", (2,2,0): "down+left",
    (1,1,1): "up+right+jump", (1,2,1): "up+left+jump",
    (2,1,1): "down+right+jump", (2,2,1): "down+left+jump",
}


def read_state(base_env):
    iface = base_env._interface
    state = {k: iface.read_ram_byte(v) for k, v in YETI_ADDRS.items()}
    state["score"] = (state["score_hi"] << 8) | state["score_lo"]
    state["bonus"] = (state["bonus_hi"] << 8) | state["bonus_lo"]
    return state


def run_episodes(env, base_env, model, n_episodes=20, max_steps=5000):
    episodes = []
    for ep in range(n_episodes):
        obs, info = env.reset()
        bg = base_env._last_raw_obs.copy() if ep == 0 else None
        steps = []
        total_reward = 0
        for step in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            state = read_state(base_env)
            a = tuple(int(x) for x in action) if hasattr(action, '__iter__') else (action,)
            steps.append({
                "action": a, "reward": reward, "total_reward": total_reward + reward,
                **state,
            })
            total_reward += reward
            if done or truncated:
                break
        episodes.append({"steps": steps, "reward": total_reward, "bg": bg})
    return episodes


def plot_trajectory(episodes, output_dir):
    """Plot best episode trajectory on game background."""
    best = max(episodes, key=lambda e: e["reward"])
    bg = episodes[0]["bg"]
    if bg is None:
        return

    img = Image.fromarray(bg)
    draw = ImageDraw.Draw(img)
    positions = [(s["x_pos"], s["y_pos"]) for s in best["steps"]]

    for i in range(1, len(positions)):
        t = i / len(positions)
        r, g = int(255 * t), int(255 * (1 - t))
        draw.line([(positions[i-1][0]*4, positions[i-1][1]),
                    (positions[i][0]*4, positions[i][1])],
                   fill=(r, g, 0), width=1)

    draw.ellipse([positions[0][0]*4-3, positions[0][1]-3,
                   positions[0][0]*4+3, positions[0][1]+3], fill=(0, 255, 0))
    draw.ellipse([positions[-1][0]*4-3, positions[-1][1]-3,
                   positions[-1][0]*4+3, positions[-1][1]+3], fill=(255, 0, 0))

    img.save(os.path.join(output_dir, "trajectory.png"))


def plot_heatmap(episodes, output_dir):
    """Plot position heatmap across all episodes."""
    bg = episodes[0]["bg"]
    if bg is None:
        return

    counts = np.zeros((200, 320), dtype=np.float32)
    for ep in episodes:
        for s in ep["steps"]:
            x, y = s["x_pos"] * 4, s["y_pos"]
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < 200 and 0 <= nx < 320:
                        counts[ny, nx] += 1

    if counts.max() > 0:
        counts = counts / counts.max()

    base = bg.astype(np.float32) * 0.3
    rgb = base.copy()
    for y in range(200):
        for x in range(320):
            v = counts[y, x]
            if v > 0.5:
                rgb[y, x] = [255, 255 * (v - 0.5) * 2, 0]
            elif v > 0.1:
                rgb[y, x] = [255 * v * 2, 50 * v, 0]
            elif v > 0.01:
                rgb[y, x, 0] = max(base[y, x, 0], 80 * v * 10)
                rgb[y, x, 2] = max(base[y, x, 2], 150 * v * 10)

    Image.fromarray(rgb.clip(0, 255).astype(np.uint8)).save(
        os.path.join(output_dir, "heatmap.png"))


def plot_actions(episodes, output_dir):
    """Plot action distribution as a text summary."""
    action_counts = {}
    total = 0
    for ep in episodes:
        for s in ep["steps"]:
            a = s["action"]
            name = ACTION_NAMES.get(a, str(a))
            action_counts[name] = action_counts.get(name, 0) + 1
            total += 1

    lines = ["Action Distribution:"]
    for name, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / total
        bar = "█" * int(pct / 2)
        lines.append(f"  {name:20s} {pct:5.1f}% {bar}")

    text = "\n".join(lines)
    with open(os.path.join(output_dir, "actions.txt"), "w") as f:
        f.write(text)
    print(text)


def plot_rewards(episodes, output_dir):
    """Plot reward timeline for best episode."""
    best = max(episodes, key=lambda e: e["reward"])
    steps = best["steps"]

    lines = ["Reward Events (best episode):"]
    for i, s in enumerate(steps):
        if abs(s["reward"]) > 0.001:
            lines.append(f"  Step {i+1:4d}: reward={s['reward']:+.2f}  "
                        f"score={s['score']}  Y={s['y_pos']}  "
                        f"action={ACTION_NAMES.get(s['action'], str(s['action']))}")

    lines.append(f"\nTotal: {best['reward']:.1f} in {len(steps)} steps")
    lines.append(f"Score events: {sum(1 for s in steps if abs(s['reward']) > 0.001)}")
    lines.append(f"Y range: {min(s['y_pos'] for s in steps)}-{max(s['y_pos'] for s in steps)}")
    lines.append(f"X range: {min(s['x_pos'] for s in steps)}-{max(s['x_pos'] for s in steps)}")

    # Action breakdown when reward > 0
    reward_actions = {}
    for s in steps:
        if s["reward"] > 0.001:
            name = ACTION_NAMES.get(s["action"], str(s["action"]))
            reward_actions[name] = reward_actions.get(name, 0) + 1
    if reward_actions:
        lines.append("\nActions when reward > 0:")
        for name, count in sorted(reward_actions.items(), key=lambda x: -x[1]):
            lines.append(f"  {name}: {count}")

    text = "\n".join(lines)
    with open(os.path.join(output_dir, "rewards.txt"), "w") as f:
        f.write(text)
    print(text)


def print_summary(episodes):
    rewards = [e["reward"] for e in episodes]
    lengths = [len(e["steps"]) for e in episodes]
    max_y = [min(s["y_pos"] for s in e["steps"]) for e in episodes]
    max_x = [max(s["x_pos"] for s in e["steps"]) for e in episodes]

    print(f"\n=== Summary ({len(episodes)} episodes) ===")
    print(f"  Reward: {np.mean(rewards):.1f} ± {np.std(rewards):.1f} "
          f"(min={min(rewards):.1f}, max={max(rewards):.1f})")
    print(f"  Length: {np.mean(lengths):.0f} ± {np.std(lengths):.0f}")
    print(f"  Best Y (lowest): {min(max_y)} (182=bottom)")
    print(f"  Max X: {max(max_x)}")

    # How many episodes climbed above Y=160?
    climbed = sum(1 for y in max_y if y < 160)
    print(f"  Episodes that climbed (Y<160): {climbed}/{len(episodes)}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Yeti agent")
    parser.add_argument("--model", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--output", default="output/yeti/analysis")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    from stable_baselines3 import PPO
    env, base_env = make_env()
    model = PPO.load(args.model, env=env)

    print(f"Running {args.episodes} episodes...")
    episodes = run_episodes(env, base_env, model, args.episodes)

    print_summary(episodes)
    plot_trajectory(episodes, args.output)
    plot_heatmap(episodes, args.output)
    plot_actions(episodes, args.output)
    plot_rewards(episodes, args.output)

    print(f"\nSaved to {args.output}/")
    env.close()


if __name__ == "__main__":
    main()
