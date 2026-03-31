#!/usr/bin/env python3
"""Plot training curves and compare runs.

Usage:
    # Plot reward curve from metrics CSV
    python scripts/plot_training.py curve output/yeti/ppo_5M/metrics.csv

    # Compare multiple runs (overlay reward curves)
    python scripts/plot_training.py compare \
        output/yeti/ppo_5M/metrics.csv \
        output/yeti/ppo_5M_impala_fullres/metrics.csv \
        --labels "Nature 84x84" "IMPALA 320x200"

    # Overlay all trajectories from N eval episodes
    python scripts/plot_training.py trajectories \
        --model output/yeti/ppo_5M/final_model.zip \
        --episodes 20 --output output/yeti/trajectories.png
"""

import argparse
import csv
import os
import sys

import numpy as np
from PIL import Image, ImageDraw


# ── Colors for multi-run comparison ──────────────────────────────────────────

COLORS = [
    (0, 100, 200),  # blue
    (200, 50, 0),  # red
    (0, 160, 60),  # green
    (180, 100, 0),  # orange
    (120, 0, 180),  # purple
    (0, 160, 160),  # teal
    (180, 0, 120),  # magenta
    (100, 100, 100),  # gray
]

COLORS_LIGHT = [
    (180, 210, 240),
    (240, 190, 180),
    (180, 230, 190),
    (240, 220, 180),
    (210, 180, 230),
    (180, 230, 230),
    (230, 180, 210),
    (210, 210, 210),
]


# ── Data loading ─────────────────────────────────────────────────────────────


def load_metrics(csv_path):
    """Load (episode, reward) from a metrics.csv file."""
    episodes, rewards = [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            rewards.append(float(row["reward"]))
    return np.array(episodes), np.array(rewards)


def parse_log_rewards(log_path):
    """Extract (step, reward) from a training output.log (fallback)."""
    import re

    steps, rewards = [], []
    with open(log_path) as f:
        for line in f:
            m = re.search(r"step (\d+)/\d+.*reward=([0-9.]+)", line)
            if m:
                steps.append(int(m.group(1)))
                rewards.append(float(m.group(2)))
    return np.array(steps), np.array(rewards)


def load_data(path):
    """Auto-detect CSV vs log format and load reward data."""
    if path.endswith(".csv"):
        eps, rews = load_metrics(path)
        return eps, rews, "episode"
    else:
        steps, rews = parse_log_rewards(path)
        return steps, rews, "step"


def smooth(values, window):
    """Rolling average with same-length output (NaN-padded start)."""
    if window <= 1 or len(values) < window:
        return values.copy()
    kernel = np.ones(window) / window
    smoothed = np.convolve(values, kernel, mode="valid")
    pad = np.full(window - 1, np.nan)
    return np.concatenate([pad, smoothed])


# ── Drawing helpers ──────────────────────────────────────────────────────────


def draw_axes(draw, w, h, margin, x_label, y_label, title=None):
    """Draw axis box and labels."""
    draw.rectangle([margin, margin, w - margin, h - margin], outline=(0, 0, 0))
    draw.text((w // 2 - 30, h - 18), x_label, fill=(80, 80, 80))
    draw.text((5, h // 2), y_label, fill=(80, 80, 80))
    if title:
        draw.text((margin + 5, 5), title, fill=(0, 0, 0))


def draw_curve(
    draw, xs, ys, x_range, y_range, margin, plot_w, plot_h, h, color, width=1
):
    """Draw a polyline on the plot area."""
    x_min, x_max = x_range
    y_min, y_max = y_range
    points = []
    for x, y in zip(xs, ys):
        if np.isnan(y):
            continue
        px = margin + int((x - x_min) / max(1, x_max - x_min) * plot_w)
        py = h - margin - int((y - y_min) / max(0.01, y_max - y_min) * plot_h)
        points.append((px, py))
    for i in range(1, len(points)):
        draw.line([points[i - 1], points[i]], fill=color, width=width)


def draw_legend(draw, labels, colors, x, y):
    """Draw a simple legend."""
    for i, (label, color) in enumerate(zip(labels, colors)):
        ly = y + i * 16
        draw.rectangle([x, ly, x + 12, ly + 10], fill=color)
        draw.text((x + 16, ly - 2), label, fill=(0, 0, 0))


def draw_y_ticks(draw, margin, h, y_min, y_max, plot_h, n_ticks=5):
    """Draw Y-axis tick labels."""
    for i in range(n_ticks + 1):
        val = y_min + (y_max - y_min) * i / n_ticks
        py = h - margin - int(i / n_ticks * plot_h)
        draw.text((2, py - 6), f"{val:.0f}", fill=(120, 120, 120))
        if i > 0 and i < n_ticks:
            # light grid line
            draw.line(
                [(margin + 1, py), (margin + int(plot_h * 2), py)],
                fill=(230, 230, 230),
                width=1,
            )


# ── Commands ─────────────────────────────────────────────────────────────────


def cmd_curve(args):
    """Plot a single reward curve."""
    xs, ys, x_type = load_data(args.path)
    if len(xs) == 0:
        print("No data found")
        return

    w, h = 900, 450
    margin = 60
    plot_w = w - 2 * margin
    plot_h = h - 2 * margin
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    x_min, x_max = xs[0], xs[-1]
    y_min = 0
    y_max = max(ys) * 1.1

    window = max(1, len(ys) // 30)
    ys_smooth = smooth(ys, window)

    title = os.path.basename(os.path.dirname(args.path))
    draw_axes(draw, w, h, margin, x_type, "reward", title)
    draw_y_ticks(draw, margin, h, y_min, y_max, plot_h)

    # Raw (light)
    draw_curve(
        draw,
        xs,
        ys,
        (x_min, x_max),
        (y_min, y_max),
        margin,
        plot_w,
        plot_h,
        h,
        COLORS_LIGHT[0],
        width=1,
    )
    # Smoothed
    draw_curve(
        draw,
        xs,
        ys_smooth,
        (x_min, x_max),
        (y_min, y_max),
        margin,
        plot_w,
        plot_h,
        h,
        COLORS[0],
        width=2,
    )

    out = args.output or args.path.replace(".csv", "_curve.png").replace(
        ".log", "_curve.png"
    )
    img.save(out)
    print(f"Saved {out}")


def cmd_compare(args):
    """Overlay reward curves from multiple runs."""
    if len(args.paths) == 0:
        print("No paths provided")
        return

    labels = args.labels or [os.path.basename(os.path.dirname(p)) for p in args.paths]
    while len(labels) < len(args.paths):
        labels.append(f"run{len(labels)}")

    # Load all data
    all_data = []
    for path in args.paths:
        xs, ys, x_type = load_data(path)
        if len(xs) > 0:
            all_data.append((xs, ys, x_type))
        else:
            print(f"Warning: no data in {path}")
            all_data.append((np.array([0]), np.array([0]), "episode"))

    w, h = 1000, 500
    margin = 65
    legend_w = 160
    plot_w = w - 2 * margin - legend_w
    plot_h = h - 2 * margin
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Global ranges
    x_min = min(d[0][0] for d in all_data)
    x_max = max(d[0][-1] for d in all_data)
    y_min = 0
    y_max = max(max(d[1]) for d in all_data) * 1.1

    draw.rectangle([margin, margin, margin + plot_w, h - margin], outline=(0, 0, 0))
    draw_y_ticks(draw, margin, h, y_min, y_max, plot_h)
    draw.text((margin + plot_w // 2 - 30, h - 18), "episode", fill=(80, 80, 80))
    draw.text((5, h // 2), "reward", fill=(80, 80, 80))

    for i, (xs, ys, _) in enumerate(all_data):
        ci = i % len(COLORS)
        window = max(1, len(ys) // 30)
        ys_smooth = smooth(ys, window)

        # Raw (light)
        draw_curve(
            draw,
            xs,
            ys,
            (x_min, x_max),
            (y_min, y_max),
            margin,
            plot_w,
            plot_h,
            h,
            COLORS_LIGHT[ci],
            width=1,
        )
        # Smoothed
        draw_curve(
            draw,
            xs,
            ys_smooth,
            (x_min, x_max),
            (y_min, y_max),
            margin,
            plot_w,
            plot_h,
            h,
            COLORS[ci],
            width=2,
        )

    draw_legend(draw, labels, COLORS[: len(labels)], margin + plot_w + 10, margin + 10)

    out = args.output or "compare.png"
    img.save(out)
    print(f"Saved {out}")


def cmd_trajectories(args):
    """Overlay all episode trajectories on the game background.

    Runs N eval episodes and draws every trajectory on one image.
    Color encodes episode index (rainbow). Start=green dot, end=red dot.
    """
    env, base, model = _build_eval_env(args.profile, args.model)

    # RAM addresses (Yeti-specific, but read from profile if possible)
    x_addr = 11090
    y_addr = 11089

    # Run episodes, collect trajectories
    trajectories = []
    bg = None
    for ep in range(args.episodes):
        obs, _ = env.reset()
        if bg is None:
            bg = base._last_raw_obs.copy()
        positions = []
        total_reward = 0
        for step in range(args.max_steps):
            action, _ = model.predict(obs, deterministic=(not args.stochastic))
            obs, reward, done, truncated, _ = env.step(action)
            x = base._interface.read_ram_byte(x_addr)
            y = base._interface.read_ram_byte(y_addr)
            positions.append((x, y))
            total_reward += reward
            if done or truncated:
                break
        trajectories.append({"positions": positions, "reward": total_reward})
        print(
            f"  Episode {ep+1}/{args.episodes}: "
            f"reward={total_reward:.1f}, steps={len(positions)}"
        )

    env.close()

    # Draw all trajectories on background
    if bg is None:
        print("No background frame captured")
        return

    # Dim the background
    img = Image.fromarray((bg.astype(np.float32) * 0.4).clip(0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)

    n = len(trajectories)
    for i, traj in enumerate(trajectories):
        # Rainbow color per episode: hue = i/n * 360
        hue = i / max(1, n)
        r, g, b = _hsv_to_rgb(hue, 0.8, 0.9)
        color = (int(r * 255), int(g * 255), int(b * 255))
        alpha_color = (int(r * 200), int(g * 200), int(b * 200))

        positions = traj["positions"]
        for j in range(1, len(positions)):
            x0, y0 = positions[j - 1][0] * 4, positions[j - 1][1]
            x1, y1 = positions[j][0] * 4, positions[j][1]
            draw.line([(x0, y0), (x1, y1)], fill=alpha_color, width=1)

        # Start dot (green) and end dot (red)
        if positions:
            sx, sy = positions[0][0] * 4, positions[0][1]
            ex, ey = positions[-1][0] * 4, positions[-1][1]
            draw.ellipse([sx - 2, sy - 2, sx + 2, sy + 2], fill=(0, 255, 0))
            draw.ellipse([ex - 2, ey - 2, ex + 2, ey + 2], fill=(255, 0, 0))

    # Stats annotation
    rewards = [t["reward"] for t in trajectories]
    draw.text(
        (5, 5),
        f"{n} episodes | reward: {np.mean(rewards):.1f} "
        f"(min={min(rewards):.1f}, max={max(rewards):.1f})",
        fill=(255, 255, 255),
    )

    out = args.output or "trajectories.png"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    img.save(out)
    print(f"Saved {out}")


def _hsv_to_rgb(h, s, v):
    """Convert HSV [0-1] to RGB [0-1]."""
    import colorsys

    return colorsys.hsv_to_rgb(h, s, v)


def _build_eval_env(profile_name, model_path):
    """Build env matching a saved model's observation space."""
    from retro_ai.envs.base_env import BaseEnv
    from retro_ai.core.preprocessing import PreprocessedEnv, PreprocessingPipeline
    from retro_ai.wrappers.gymnasium_wrapper import GymnasiumWrapper
    from retro_ai.training.game_profile import GameProfileRegistry
    from stable_baselines3 import PPO

    registry = GameProfileRegistry()
    profile = registry.load(profile_name)

    # Detect model's expected observation shape to set resize correctly
    import zipfile, json

    resize = profile.resize
    with zipfile.ZipFile(model_path) as z:
        with z.open("data") as f:
            data = json.load(f)
            shape = data.get("observation_space", {}).get("_shape")
            if shape and len(shape) == 3:
                _, h, w = shape
                if profile.resize is None and (h != 200 or w != 320):
                    resize = (h, w)
                elif profile.resize is not None:
                    resize = (h, w)

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
        grayscale=profile.grayscale,
        resize=resize,
        frame_stack=profile.frame_stack,
        frame_skip=profile.frame_skip,
    )
    preprocessed = PreprocessedEnv(base, pipeline, frame_maxpool=profile.frame_maxpool)
    env = GymnasiumWrapper(preprocessed)
    model = PPO.load(model_path, env=env)
    return env, base, model


def cmd_trajectory_video(args):
    """Animated video: one dot per episode at each timestep.

    All episodes play simultaneously. Each dot is colored by episode index
    (rainbow). Trails fade behind the current position.
    """
    import imageio

    env, base, model = _build_eval_env(args.profile, args.model)

    x_addr = 11090
    y_addr = 11089

    # Collect trajectories
    trajectories = []
    bg = None
    for ep in range(args.episodes):
        obs, _ = env.reset()
        if bg is None:
            bg = base._last_raw_obs.copy()
        positions = []
        for step in range(args.max_steps):
            action, _ = model.predict(obs, deterministic=(not args.stochastic))
            obs, reward, done, truncated, _ = env.step(action)
            x = base._interface.read_ram_byte(x_addr)
            y = base._interface.read_ram_byte(y_addr)
            positions.append((x, y))
            if done or truncated:
                break
        trajectories.append(positions)
        print(f"  Episode {ep+1}/{args.episodes}: {len(positions)} steps")

    env.close()
    if bg is None:
        print("No background frame captured")
        return

    # Build colors per episode
    n = len(trajectories)
    colors = []
    for i in range(n):
        r, g, b = _hsv_to_rgb(i / max(1, n), 0.9, 0.95)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))

    max_len = max(len(t) for t in trajectories)
    bg_dim = (bg.astype(np.float32) * 0.35).clip(0, 255).astype(np.uint8)
    trail_len = args.trail

    out = args.output or "trajectory_video.mp4"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    writer = imageio.get_writer(out, fps=args.fps, codec="libx264")

    for t in range(max_len):
        frame = Image.fromarray(bg_dim.copy())
        draw = ImageDraw.Draw(frame)

        for i, positions in enumerate(trajectories):
            if t >= len(positions):
                # Episode ended — draw final position as X
                px, py = positions[-1][0] * 4, positions[-1][1]
                draw.line([(px - 2, py - 2), (px + 2, py + 2)], fill=(100, 100, 100))
                draw.line([(px - 2, py + 2), (px + 2, py - 2)], fill=(100, 100, 100))
                continue

            # Draw trail
            start = max(0, t - trail_len)
            for j in range(start, t):
                x0, y0 = positions[j][0] * 4, positions[j][1]
                x1, y1 = positions[j + 1][0] * 4, positions[j + 1][1]
                fade = int(255 * (j - start) / max(1, trail_len))
                c = (
                    colors[i][0] * fade // 255,
                    colors[i][1] * fade // 255,
                    colors[i][2] * fade // 255,
                )
                draw.line([(x0, y0), (x1, y1)], fill=c, width=1)

            # Draw current dot
            px, py = positions[t][0] * 4, positions[t][1]
            r = 3
            draw.ellipse([px - r, py - r, px + r, py + r], fill=colors[i])

        # Timestamp
        draw.text((5, 5), f"t={t}", fill=(255, 255, 255))
        draw.text((5, 190), f"{n} episodes", fill=(200, 200, 200))

        writer.append_data(np.array(frame))

    writer.close()
    print(f"Saved {out} ({max_len} frames)")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Plot training curves and trajectories"
    )
    sub = parser.add_subparsers(dest="command")

    # curve
    p_curve = sub.add_parser("curve", help="Plot single reward curve")
    p_curve.add_argument("path", help="metrics.csv or output.log")
    p_curve.add_argument("--output", "-o", help="Output PNG path")

    # compare
    p_cmp = sub.add_parser("compare", help="Overlay multiple reward curves")
    p_cmp.add_argument("paths", nargs="+", help="metrics.csv files")
    p_cmp.add_argument("--labels", nargs="+", help="Legend labels")
    p_cmp.add_argument("--output", "-o", help="Output PNG path")

    # trajectories
    p_traj = sub.add_parser("trajectories", help="Overlay eval trajectories on game bg")
    p_traj.add_argument("--model", required=True, help="Model .zip path")
    p_traj.add_argument("--profile", default="yeti", help="Game profile name")
    p_traj.add_argument("--episodes", type=int, default=20)
    p_traj.add_argument("--max-steps", type=int, default=5000)
    p_traj.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy instead of deterministic",
    )
    p_traj.add_argument("--output", "-o", help="Output PNG path")

    # trajectory video
    p_vid = sub.add_parser(
        "trajectory-video", help="Animated video of eval trajectories"
    )
    p_vid.add_argument("--model", required=True, help="Model .zip path")
    p_vid.add_argument("--profile", default="yeti", help="Game profile name")
    p_vid.add_argument("--episodes", type=int, default=10)
    p_vid.add_argument("--max-steps", type=int, default=5000)
    p_vid.add_argument("--stochastic", action="store_true")
    p_vid.add_argument("--fps", type=int, default=15)
    p_vid.add_argument(
        "--trail",
        type=int,
        default=30,
        help="Trail length in frames behind current dot",
    )
    p_vid.add_argument("--output", "-o", help="Output MP4 path")

    args = parser.parse_args()
    if args.command == "curve":
        cmd_curve(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "trajectories":
        cmd_trajectories(args)
    elif args.command == "trajectory-video":
        cmd_trajectory_video(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
