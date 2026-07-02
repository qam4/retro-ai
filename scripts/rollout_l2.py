#!/usr/bin/env python3
"""Roll out trained Yeti *level-2* policies from the level-2 start state,
record video, and report how far down the tower each snapshot gets.

Level 2 boots from a save-state (``level2_start.sav``), NOT a game reset, so
the from-reset renderers (``render_from_reset.py``) can't be used — they'd
show level 1. This script reproduces the curriculum env's reset exactly
(load_state -> notify_state_loaded -> 5 noop settle frames) and then drives
the policy, tracking the agent's descent.

Geometry (see yeti_map.LEVEL2): the agent spawns on floor 1 (y=30) and must
DESCEND. Floors are at y = {1:30, 2:54, 3:78, 4:102, 5:126, 6:150}; both
fruits sit on floor 5 (y~126), the princess on floor 6 (y~168). "Deepest
floor" = the lowest (highest-y) floor the agent stood on during the episode.

Two uses:

1. Sweep candidate checkpoints to find promising ones::

     env RETRO_AI_ROM_DIR=roms PYTHONPATH=python:build/ci-linux \\
       python3 scripts/rollout_l2.py \\
         --snapshots-dir output/mo5/yeti/training/yeti_curriculum_l2_v3_10m/snapshots \\
         --steps 100000,6300000,7600000,9900000,10000000 \\
         --episodes 20 --out output/mo5/yeti/videos/l2_v3_probe --video-best 1

2. Inspect a single model with video of every kept episode::

     ... python3 scripts/rollout_l2.py \\
         --model .../snapshots/model_9900000_steps.zip \\
         --episodes 8 --out debug/l2_9900k --video
"""
from __future__ import annotations

import argparse
import glob
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
from retro_ai.training.env_builder import build_training_env
from retro_ai.training.run_config import EnvConfig
from retro_ai.training.yeti_map import agent_floor_from_pixel_y, get_level_map
from stable_baselines3 import PPO

# RAM addresses (level-independent player state).
X_ADDR = 11090
Y_ADDR = 11089
LIVES_ADDR = 11095
POSE_ADDR = 11092  # sprite pose; surface set below
# Death flag (0x2AFC): 32 alive / 65 dead, cause-agnostic. Same name as the
# mo5_rl.cpp / game-profile `death_flag_addr`/`death_flag_value`.
DEATH_FLAG_ADDR, DEATH_FLAG_VALUE = 11004, 65
BONUS_HI, BONUS_LO = 11010, 11011
SCORE_HI, SCORE_LO = 11093, 11094
# Poses where the agent is on a surface (grounded floor or ladder). Only
# these count toward "deepest floor reached" so a fall-through doesn't
# inflate depth. See experiments/003-yeti-training.md "run 3".
SURFACE_POSES = frozenset({0, 1, 2, 3, 4, 5, 8})
# Level-2 fruit-presence tilemap cells (byte != 0 => fruit still on map).
L2_FRUIT_PRESENCE = {1: 11950, 2: 11975}

# RAM x is in 4px units (x_px = X*4); y is already in pixels. Frame is 320x200.
X_SCALE = 4
FRAME_W, FRAME_H = 320, 200

# joystick action = (vertical, horizontal, fire); 0/1/2 -> none/up|right/down|left.
ACTION_NAMES = {
    (0, 0, 0): "noop",
    (1, 0, 0): "up",
    (2, 0, 0): "down",
    (0, 1, 0): "right",
    (0, 2, 0): "left",
    (0, 0, 1): "jump",
    (1, 0, 1): "up+jump",
    (2, 0, 1): "down+jump",
    (0, 1, 1): "right+jump",
    (0, 2, 1): "left+jump",
    (1, 1, 0): "up+right",
    (1, 2, 0): "up+left",
    (2, 1, 0): "down+right",
    (2, 2, 0): "down+left",
    (1, 1, 1): "up+right+jump",
    (1, 2, 1): "up+left+jump",
    (2, 1, 1): "down+right+jump",
    (2, 2, 1): "down+left+jump",
}


@dataclass
class EpisodeResult:
    ep: int
    length: int
    deepest_floor: int
    max_y: int
    fruits: int
    end_reason: str
    final_x: int
    final_y: int
    frames: Optional[list]
    positions: list  # list of (x_px, y) per step
    actions: list  # list of (v, h, f) per step
    bg: Optional[np.ndarray]  # start-frame background (only ep 0)


def _load_font(size):
    """A crisp scalable font, falling back gracefully to PIL's default."""
    from PIL import ImageFont

    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        pass
    try:
        import matplotlib

        return ImageFont.truetype(
            f"{matplotlib.get_data_path()}/fonts/ttf/DejaVuSans.ttf", size
        )
    except Exception:
        pass
    try:
        return ImageFont.load_default(size)  # Pillow >= 10.1
    except Exception:
        return ImageFont.load_default()


def _draw_keypad(d, cx, cy, action, gap, a, r, font=None) -> None:
    """Draw a small D-pad + jump button; highlight pressed inputs.
    action = (vertical, horizontal, fire): v 1=up/2=down, h 1=right/2=left."""
    v, h, f = (action + (0, 0, 0))[:3]
    ON, OFF, OUT = (0, 255, 0), (60, 60, 60), (140, 140, 140)
    up, down = v == 1, v == 2
    right, left = h == 1, h == 2
    tris = {
        up: [(cx, cy - gap - a), (cx - a, cy - gap + a), (cx + a, cy - gap + a)],
        down: [(cx, cy + gap + a), (cx - a, cy + gap - a), (cx + a, cy + gap - a)],
        left: [(cx - gap - a, cy), (cx - gap + a, cy - a), (cx - gap + a, cy + a)],
        right: [(cx + gap + a, cy), (cx + gap - a, cy - a), (cx + gap - a, cy + a)],
    }
    for active, pts in [
        (up, tris[up]),
        (down, tris[down]),
        (left, tris[left]),
        (right, tris[right]),
    ]:
        d.polygon(pts, fill=ON if active else OFF, outline=OUT)
    # Jump button to the right of the cross.
    jx = cx + gap * 2 + r + 4
    d.ellipse([jx - r, cy - r, jx + r, cy + r], fill=ON if f else OFF, outline=OUT)
    if font is not None:
        d.text(
            (jx - r + 2, cy - r + 1),
            "J",
            font=font,
            fill=(0, 0, 0) if f else (180, 180, 180),
        )


def _draw_hud(
    rgb,
    step,
    floor,
    x,
    y,
    fruits,
    deepest,
    scale=3,
    lives=None,
    bonus=None,
    score=None,
    action=None,
    end="",
) -> np.ndarray:
    """Upscale the game frame, then draw a HUD in a strip ADDED BELOW it
    (never over the game's own HUD) with a crisp font at full resolution
    (avoids the aliasing from upscaling tiny text). The game's score/bonus
    counters are blank after load_state — a separate known MO5 bug — so we
    surface the real RAM values here."""
    frame = np.asarray(rgb, dtype=np.uint8)
    h, w = frame.shape[:2]
    W = w * scale
    big = Image.fromarray(frame).resize((W, h * scale), Image.NEAREST)
    fs = max(11, 6 * scale)  # font size scales with the frame
    banner = fs * 2 + 10  # TOP banner (scrubbers rarely sit here)
    footer = fs + 14  # blank BOTTOM margin so the player's
    # scrubber overlays this, not the game
    canvas = Image.new("RGB", (W, banner + h * scale + footer), (0, 0, 0))
    canvas.paste(big, (0, banner))
    d = ImageDraw.Draw(canvas)
    font = _load_font(fs)
    d.text(
        (6, 3),
        f"step {step:>4}  floor {floor}  deepest {deepest}",
        font=font,
        fill=(0, 255, 0),
    )
    game = f"  lives {lives} bonus {bonus} score {score}" if lives is not None else ""
    d.text(
        (6, 3 + fs + 4),
        f"pos ({x:>3},{y:>3})  fruits {fruits}{game}",
        font=font,
        fill=(255, 255, 255),
    )
    # D-pad + jump indicator on the right of the top banner (inputs pressed).
    if action is not None:
        gap = max(8, fs // 2)
        a = max(4, fs // 4)
        r = max(6, fs // 3)
        kp_cy = banner // 2
        kp_cx = W - (gap * 3 + r * 2 + 30)
        _draw_keypad(
            d, kp_cx, kp_cy, tuple(action), gap, a, r, font=_load_font(max(9, fs - 6))
        )
    return np.asarray(canvas, dtype=np.uint8)


def _run_episode(
    stack,
    model,
    start_state: bytes,
    ep: int,
    max_steps: int,
    settle: int,
    deterministic: bool,
    keep_frames: bool,
    scale: int = 3,
) -> EpisodeResult:
    base, gym_env, iface = stack.base, stack.gym, stack.base._interface

    iface.load_state(start_state)
    stack.preprocessed.notify_state_loaded()
    obs = None
    for _ in range(settle):
        obs, _, _, _, _ = gym_env.step([0, 0, 0])

    def y():
        return iface.read_ram_byte(Y_ADDR)

    def x():
        return iface.read_ram_byte(X_ADDR)

    def present():
        return {k: iface.read_ram_byte(a) != 0 for k, a in L2_FRUIT_PRESENCE.items()}

    start_present = present()
    prev_lives = iface.read_ram_byte(LIVES_ADDR)
    max_y = y()
    deepest_floor = agent_floor_from_pixel_y(max_y, level=2) or 1

    bg = None
    if ep == 0 and base._last_raw_obs is not None:
        bg = np.asarray(base._last_raw_obs, dtype=np.uint8).copy()

    frames = [] if keep_frames else None
    positions = []
    actions = []
    steps = 0
    end_reason = "max_steps"
    while steps < max_steps:
        obs_chw = np.transpose(obs, (2, 0, 1))
        action, _ = model.predict(obs_chw, deterministic=deterministic)
        obs, _, done, trunc, _ = gym_env.step(action)
        steps += 1

        cx, cy = x(), y()
        pose = iface.read_ram_byte(POSE_ADDR)
        positions.append((cx * X_SCALE, cy))
        actions.append(tuple(int(a) for a in np.ravel(action)))
        if cy > max_y:
            max_y = cy
        f = agent_floor_from_pixel_y(cy, level=2)
        # Only credit a floor the agent is actually STANDING on (grounded/
        # ladder pose) — a fall passing through a floor line doesn't count.
        if f is not None and f > deepest_floor and pose in SURFACE_POSES:
            deepest_floor = f

        now = present()
        fruits = sum(1 for k in start_present if start_present[k] and not now[k])
        lives = iface.read_ram_byte(LIVES_ADDR)

        if keep_frames and base._last_raw_obs is not None:
            bonus = (iface.read_ram_byte(BONUS_HI) << 8) | iface.read_ram_byte(BONUS_LO)
            score = (iface.read_ram_byte(SCORE_HI) << 8) | iface.read_ram_byte(SCORE_LO)
            frames.append(
                _draw_hud(
                    base._last_raw_obs,
                    steps,
                    f if f else "-",
                    cx,
                    cy,
                    fruits,
                    deepest_floor,
                    scale=scale,
                    lives=lives,
                    bonus=bonus,
                    score=score,
                    action=tuple(int(a) for a in np.ravel(action)),
                )
            )

        # Death via the cause-agnostic 0x2AFC flag (the lives byte is inert
        # on L2, so the old lives-based check missed falls-to-death).
        if iface.read_ram_byte(DEATH_FLAG_ADDR) == DEATH_FLAG_VALUE:
            end_reason = "death"
            break
        if lives < prev_lives and prev_lives > 0:
            end_reason = "death"
            break
        prev_lives = lives
        if done or trunc:
            end_reason = "done" if done else "trunc"
            break

    now = present()
    fruits = sum(1 for k in start_present if start_present[k] and not now[k])
    return EpisodeResult(
        ep=ep,
        length=steps,
        deepest_floor=deepest_floor,
        max_y=max_y,
        fruits=fruits,
        end_reason=end_reason,
        final_x=x(),
        final_y=y(),
        frames=frames,
        positions=positions,
        actions=actions,
        bg=bg,
    )


def _write_heatmap(results, label, out_dir) -> None:
    """Overlay a position heatmap on the start-frame background."""
    bg = next((r.bg for r in results if r.bg is not None), None)
    if bg is None:
        return
    counts = np.zeros((FRAME_H, FRAME_W), dtype=np.float32)
    for r in results:
        for xpx, yy in r.positions:
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    ny, nx = yy + dy, xpx + dx
                    if 0 <= ny < FRAME_H and 0 <= nx < FRAME_W:
                        counts[ny, nx] += 1
    if counts.max() > 0:
        counts /= counts.max()
    base = bg.astype(np.float32) * 0.35
    rgb = base.copy()
    for yy in range(FRAME_H):
        for xx in range(FRAME_W):
            v = counts[yy, xx]
            if v > 0.5:
                rgb[yy, xx] = [255, 255 * (v - 0.5) * 2, 0]
            elif v > 0.1:
                rgb[yy, xx] = [255 * v * 2, 50 * v, 0]
            elif v > 0.01:
                rgb[yy, xx, 0] = max(base[yy, xx, 0], 80 * v * 10)
                rgb[yy, xx, 2] = max(base[yy, xx, 2], 150 * v * 10)
    path = os.path.join(out_dir, f"{label}_heatmap.png")
    Image.fromarray(rgb.clip(0, 255).astype(np.uint8)).save(path)
    print(f"    heatmap  -> {path}")


def _write_trajectory(results, label, out_dir) -> None:
    """Draw the deepest episode's path on the start background (green=start,
    fading to red=end)."""
    bg = next((r.bg for r in results if r.bg is not None), None)
    if bg is None:
        return
    best = max(results, key=lambda r: (r.deepest_floor, r.max_y))
    pos = best.positions
    if len(pos) < 2:
        return
    img = Image.fromarray(bg.copy())
    d = ImageDraw.Draw(img)
    for i in range(1, len(pos)):
        t = i / len(pos)
        d.line(
            [pos[i - 1], pos[i]], fill=(int(255 * t), int(255 * (1 - t)), 0), width=1
        )
    d.ellipse(
        [pos[0][0] - 3, pos[0][1] - 3, pos[0][0] + 3, pos[0][1] + 3], fill=(0, 255, 0)
    )
    d.ellipse(
        [pos[-1][0] - 3, pos[-1][1] - 3, pos[-1][0] + 3, pos[-1][1] + 3],
        fill=(255, 0, 0),
    )
    path = os.path.join(out_dir, f"{label}_trajectory_ep{best.ep:03d}.png")
    img.save(path)
    print(f"    traj     -> {path}")


def _print_action_dist(results, label) -> None:
    from collections import Counter

    c = Counter()
    xs = []
    for r in results:
        for a in r.actions:
            c[ACTION_NAMES.get(a, str(a))] += 1
        xs.extend(px // X_SCALE for (px, _y) in r.positions)  # agent_x (RAM units)
    total = sum(c.values()) or 1
    # Horizontal intent tally.
    left = sum(n for name, n in c.items() if "left" in name)
    right = sum(n for name, n in c.items() if "right" in name)
    print(
        f"    actions [{label}] (top): "
        + ", ".join(f"{name} {100 * n / total:.0f}%" for name, n in c.most_common(6))
    )
    print(
        f"    horizontal: left {100 * left / total:.0f}%  "
        f"right {100 * right / total:.0f}%"
    )
    if xs:
        xs.sort()
        n = len(xs)

        def p(q):
            return xs[min(n - 1, int(n * q))]

        # First F1->F2 gap ~ agent_x 8-11; descent ladder ~ agent_x 18.
        crossed = sum(
            1
            for r in results
            if max((px // X_SCALE for (px, _y) in r.positions), default=0) >= 18
        )
        print(
            f"    agent_x: max={max(xs)} p50={p(.5)} p90={p(.9)} p99={p(.99)}  "
            f"(gap~8-11, F1->F2 ladder~18)  episodes reaching ladder(x>=18): "
            f"{crossed}/{len(results)}"
        )


def _model_label(path: str) -> str:
    m = re.search(r"model_(\d+)_steps", os.path.basename(path))
    if m:
        return f"{int(m.group(1)) // 1000}k"
    return os.path.splitext(os.path.basename(path))[0]


def _resolve_models(args) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for m in args.model or []:
        out.append((_model_label(m), m))
    if args.snapshots_dir:
        if args.steps:
            for s in args.steps.split(","):
                s = s.strip()
                p = os.path.join(args.snapshots_dir, f"model_{int(s)}_steps.zip")
                out.append((_model_label(p), p))
        else:
            for p in sorted(
                glob.glob(os.path.join(args.snapshots_dir, "model_*_steps.zip"))
            ):
                out.append((_model_label(p), p))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", action="append", help="model .zip (repeatable)")
    p.add_argument("--snapshots-dir", help="dir of model_<step>_steps.zip")
    p.add_argument(
        "--steps", help="comma-separated step numbers within --snapshots-dir"
    )
    p.add_argument("--state", default="output/mo5/yeti/level2/level2_start.sav")
    p.add_argument("--profile", default="yeti_fruit_level2")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument(
        "--stall-threshold",
        type=int,
        default=1000,
        help="frames of no progress before env truncates; high = let it run",
    )
    p.add_argument("--settle-frames", type=int, default=5)
    p.add_argument(
        "--deterministic",
        action="store_true",
        help="greedy policy (default: stochastic, matching training rollouts)",
    )
    p.add_argument("--out", required=True)
    p.add_argument("--video", action="store_true", help="save video for every episode")
    p.add_argument(
        "--video-best",
        type=int,
        default=0,
        help="save video only for the N deepest episodes per model",
    )
    p.add_argument(
        "--heatmap",
        action="store_true",
        help="write position heatmap + trajectory + action distribution",
    )
    # frame_skip=4 => 1 recorded frame per 4 emulator frames; emulator is
    # ~50 Hz, so 12 fps ~= real-time playback.
    p.add_argument("--fps", type=int, default=12)
    p.add_argument(
        "--scale",
        type=int,
        default=3,
        help="integer upscale for saved videos (HUD readability)",
    )
    args = p.parse_args()

    models = _resolve_models(args)
    if not models:
        raise SystemExit("Provide --model or --snapshots-dir")

    with open(args.state, "rb") as f:
        start_state = f.read()

    env_cfg = EnvConfig(
        profile=args.profile,
        action_mode="joystick",
        max_steps=args.max_steps,
        stall_threshold=args.stall_threshold,
        resize=(84, 84),
    )
    stack = build_training_env(args.profile, env_cfg)
    stack.base.reset(seed=0)

    lvl = get_level_map(2)
    os.makedirs(args.out, exist_ok=True)
    print(f"L2 floors (y): {lvl.floor_top_y}  fruits on floor 5")
    print(
        f"state={args.state}  episodes={args.episodes}  "
        f"deterministic={args.deterministic}\n"
    )

    summary = []
    for label, path in models:
        if not os.path.exists(path):
            print(f"[{label}] MISSING {path}")
            continue
        model = PPO.load(path, device="auto")
        keep_all = args.video
        results: List[EpisodeResult] = []
        for ep in range(args.episodes):
            keep = keep_all or args.video_best > 0
            r = _run_episode(
                stack,
                model,
                start_state,
                ep,
                args.max_steps,
                args.settle_frames,
                args.deterministic,
                keep,
                scale=args.scale,
            )
            results.append(r)

        deepest = max(r.deepest_floor for r in results)
        mean_floor = sum(r.deepest_floor for r in results) / len(results)
        best_fruits = max(r.fruits for r in results)
        reached_f = {
            fl: sum(1 for r in results if r.deepest_floor >= fl) / len(results)
            for fl in (2, 3, 4, 5)
        }
        print(
            f"[{label}] deepest_floor={deepest} mean={mean_floor:.2f} "
            f"fruits={best_fruits}  reach: "
            + " ".join(f"F{fl}>={reached_f[fl]:.0%}" for fl in (2, 3, 4, 5))
        )

        if args.heatmap:
            _print_action_dist(results, label)
            _write_heatmap(results, label, args.out)
            _write_trajectory(results, label, args.out)

        # Save videos.
        to_save = []
        if keep_all:
            to_save = results
        elif args.video_best > 0:
            to_save = sorted(
                results, key=lambda r: (r.deepest_floor, r.max_y), reverse=True
            )[: args.video_best]
        for r in to_save:
            if not r.frames:
                continue
            name = (
                f"{label}_ep{r.ep:03d}_floor{r.deepest_floor}_y{r.max_y}"
                f"_fr{r.fruits}_{r.end_reason}_len{r.length}.mp4"
            )
            out_path = os.path.join(args.out, name)
            # Frames are already upscaled with a crisp HUD by _draw_hud.
            # macro_block_size=1 keeps exact dimensions (no silent padding).
            imageio.mimsave(out_path, r.frames, fps=args.fps, macro_block_size=1)
            print(f"    video -> {out_path}")
        summary.append((label, deepest, mean_floor, best_fruits))
        del model

    print("\n=== summary (deepest floor reached, floor 5 = fruits) ===")
    for label, deepest, mean_floor, best_fruits in summary:
        print(
            f"  {label:>10}: deepest F{deepest}  mean F{mean_floor:.2f}  "
            f"fruits {best_fruits}"
        )


if __name__ == "__main__":
    main()
