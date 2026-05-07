#!/usr/bin/env python3
"""Replay a saved state forward and dump PNGs or an MP4.

Load an emulator save-state and advance N frames under a fixed action
(default: noop). Writes per-frame PNGs so you can flip through them,
or an MP4 if imageio-ffmpeg is installed and --video is passed.

The tool is meant for debugging: "what does this state look like when
you just let it run?" Useful for inspecting the handful of save-states
produced by Go-Explore or the curriculum that we can't easily reason
about from RAM alone.

Usage
-----

Replay a single state (PNGs per frame, all-noops, 120 frames)::

    python scripts/play_state.py \\
        path/to/state.pkl \\
        --out debug/playback \\
        --frames 120

Replay with a fixed action every frame (e.g. hold UP+RIGHT)::

    python scripts/play_state.py state.pkl --out debug/playback \\
        --action 1 1 0

Load from a cell inside a Go-Explore archive.pkl or curriculum
checkpoints.pkl by index::

    python scripts/play_state.py \\
        output/.../go_explore_v9/archive.pkl \\
        --cell-index 7 \\
        --out debug/playback_cell7

Write an MP4 instead of per-frame PNGs::

    python scripts/play_state.py state.pkl --out debug/pb.mp4 --video
"""

from __future__ import annotations

import argparse
import os
import pickle
from typing import Any, List, Tuple

import numpy as np
from PIL import Image
from retro_ai.training.env_builder import build_training_env
from retro_ai.training.run_config import EnvConfig


def _load_state_bytes(path: str, cell_index: int | None) -> bytes:
    """Return raw state bytes from a pickle file.

    Supports:
      - raw bytes pickle (``pickle.dump(bytes(state))`` style)
      - curriculum format: ``{"checkpoints": [[state, ...], ...]}``
      - Go-Explore archive format: ``{cell_key: {"state": bytes, ...}}``
    """
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)
    if isinstance(data, dict):
        # Go-Explore archive
        if "checkpoints" not in data:
            cells = list(data.values())
            if cell_index is None:
                cell_index = 0
            return bytes(cells[cell_index]["state"])
        # Curriculum format — flatten across CPs
        all_states: List[bytes] = []
        for cp_states in data["checkpoints"]:
            for s in cp_states:
                all_states.append(bytes(s))
        if cell_index is None:
            cell_index = 0
        return all_states[cell_index]
    raise SystemExit(f"unknown state file format at {path!r}")


def _fixed_action(args_action: List[int] | None) -> List[int]:
    if args_action is None:
        return [0, 0, 0]
    if len(args_action) != 3:
        raise SystemExit(f"--action needs exactly 3 integers, got {args_action!r}")
    return [int(v) for v in args_action]


def _read_ram_summary(iface) -> Tuple[int, int, int, int, int]:
    x = iface.read_ram_byte(11090)
    y = iface.read_ram_byte(11089)
    fr = iface.read_ram_byte(11055)
    lives = iface.read_ram_byte(11095)
    bonus = (iface.read_ram_byte(11010) << 8) | iface.read_ram_byte(11011)
    return x, y, fr, lives, bonus


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "state_path",
        help="Pickle file holding one state or an archive of states.",
    )
    p.add_argument(
        "--cell-index",
        type=int,
        default=None,
        help="If state_path is an archive/curriculum dict, pick this index.",
    )
    p.add_argument(
        "--profile",
        default="yeti_fruit",
        help="Game profile used to build the env.",
    )
    p.add_argument(
        "--frames",
        type=int,
        default=120,
        help="How many frames to step forward (default 120 = 3 seconds).",
    )
    p.add_argument(
        "--action",
        type=int,
        nargs=3,
        default=None,
        help="Fixed 3-int action every frame (default: [0, 0, 0] = noop).",
    )
    p.add_argument(
        "--settle-frames",
        type=int,
        default=5,
        help="Noop frames after load before starting playback (default 5).",
    )
    p.add_argument(
        "--out",
        required=True,
        help=(
            "Output directory for PNGs, OR an .mp4 path if --video is set. "
            "Directory is created if missing."
        ),
    )
    p.add_argument(
        "--video",
        action="store_true",
        help="Write a single MP4 instead of per-frame PNGs.",
    )
    p.add_argument(
        "--fps",
        # MO5 runs at 50 Hz (PAL); default to real-time playback.
        type=int,
        default=50,
        help="MP4 framerate (MO5 runs at 50 Hz; default 50 = real-time).",
    )
    args = p.parse_args()

    state = _load_state_bytes(args.state_path, args.cell_index)
    action = _fixed_action(args.action)

    env_cfg = EnvConfig(
        profile=args.profile,
        action_mode="joystick",
        max_steps=100_000,  # we don't want training-max-steps to cut us off
        stall_threshold=100_000,
        resize=(84, 84),
    )
    stack = build_training_env(args.profile, env_cfg)
    base = stack.base
    base.reset(seed=0)
    base._interface.load_state(state)

    # Settle.
    obs: Any = None
    for _ in range(args.settle_frames):
        obs, _, _, _, _ = base.step([0, 0, 0])

    frames: List[np.ndarray] = []
    # Capture the post-settle frame too so frame 0 of the playback is
    # the "starting point" the policy/user would see.
    if obs is not None:
        frames.append(obs.astype(np.uint8).copy())

    iface = base._interface
    print(f"{'frame':>5}  {'x':>3} {'y':>3} {'fr':>2} {'lv':>2}  {'bonus':>5}")
    x, y, fr, lv, bonus = _read_ram_summary(iface)
    print(f"{0:>5}  {x:>3} {y:>3} {fr:>2} {lv:>2}  {bonus:>5}  (post-settle)")

    for i in range(1, args.frames + 1):
        obs, _, done, truncated, _ = base.step(action)
        frames.append(obs.astype(np.uint8).copy())
        x, y, fr, lv, bonus = _read_ram_summary(iface)
        if i % 10 == 0 or done or truncated:
            flag = " DONE" if done else (" TRUNC" if truncated else "")
            print(f"{i:>5}  {x:>3} {y:>3} {fr:>2} {lv:>2}  {bonus:>5}{flag}")
        # Do NOT break on done/truncated — we want to see what the
        # emulator does through and past the terminal event.

    # Write outputs.
    if args.video:
        import imageio.v2 as imageio  # type: ignore

        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        imageio.mimsave(args.out, frames, fps=args.fps)
        print(f"wrote {args.out}  ({len(frames)} frames)")
    else:
        os.makedirs(args.out, exist_ok=True)
        for i, f in enumerate(frames):
            Image.fromarray(f).save(os.path.join(args.out, f"f{i:04d}.png"))
        print(f"wrote {len(frames)} PNGs to {args.out}/")


if __name__ == "__main__":
    main()
