#!/usr/bin/env python3
"""Print a per-frame RAM trace for a save state under a fixed action.

Companion to ``scripts/play_state.py`` — while that one dumps
framebuffer PNGs, this one dumps every RAM variable we currently care
about, one row per frame. Useful for understanding what signals fire
when, in cases where death / stall / life-loss semantics are unclear.

The action defaults to noop. Pass ``--action 1 1 0`` (or any 3 ints)
to hold a fixed input.

Usage
-----

Trace a single state for 60 noop frames::

    python scripts/trace_state.py state.pkl --frames 60

Trace a cell inside an archive::

    python scripts/trace_state.py output/.../archive.pkl \\
        --cell-index 7 --frames 60

Trace while holding UP::

    python scripts/trace_state.py state.pkl --action 1 0 0 --frames 60
"""

from __future__ import annotations

import argparse
import pickle
from typing import List


# Yeti-specific addresses.
ADDRS = {
    "x": 11090,
    "y": 11089,
    "fruits_rem": 11055,
    "lives": 11095,
    "bonus_hi": 11010,
    "bonus_lo": 11011,
    "score_hi": 11093,
    "score_lo": 11094,
    "fruit1": 0x2FAD,
    "fruit2": 0x2F00,
    "fruit3": 0x2E68,
    "fruit4": 0x2DD8,
}


def _load_state_bytes(path: str, cell_index: int | None) -> bytes:
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)
    if isinstance(data, dict):
        if "checkpoints" not in data:
            cells = list(data.values())
            return bytes(cells[cell_index or 0]["state"])
        all_states: List[bytes] = []
        for cp_states in data["checkpoints"]:
            for s in cp_states:
                all_states.append(bytes(s))
        return all_states[cell_index or 0]
    raise SystemExit(f"unknown format at {path!r}")


def _read_all(iface):
    r = {name: iface.read_ram_byte(addr) for name, addr in ADDRS.items()}
    r["bonus"] = (r["bonus_hi"] << 8) | r["bonus_lo"]
    r["score"] = (r["score_hi"] << 8) | r["score_lo"]
    del r["bonus_hi"]
    del r["bonus_lo"]
    del r["score_hi"]
    del r["score_lo"]
    return r


def main() -> None:
    from retro_ai.training.env_builder import build_training_env
    from retro_ai.training.run_config import EnvConfig

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("state_path")
    p.add_argument("--cell-index", type=int, default=None)
    p.add_argument("--profile", default="yeti_fruit")
    p.add_argument("--frames", type=int, default=60)
    p.add_argument("--settle-frames", type=int, default=5)
    p.add_argument("--action", type=int, nargs=3, default=[0, 0, 0])
    args = p.parse_args()

    state = _load_state_bytes(args.state_path, args.cell_index)

    env_cfg = EnvConfig(
        profile=args.profile,
        action_mode="joystick",
        max_steps=1_000_000,
        stall_threshold=1_000_000,
        resize=(84, 84),
    )
    stack = build_training_env(args.profile, env_cfg)
    base = stack.base
    base.reset(seed=0)
    base._interface.load_state(state)
    for _ in range(args.settle_frames):
        base.step([0, 0, 0])

    iface = base._interface
    initial = _read_all(iface)
    prev_bonus = initial["bonus"]
    stall_c = 0
    # Header.
    cols = [
        "f",
        "x",
        "y",
        "fr",
        "lives",
        "bonus",
        "score",
        "f1",
        "f2",
        "f3",
        "f4",
        "done",
        "trunc",
        "bonus_stall",
    ]
    print("  ".join(f"{c:>5s}" for c in cols))

    def print_row(f: int, r: dict, done: int, trunc: int, stall: int) -> None:
        vals = [
            f,
            r["x"],
            r["y"],
            r["fruits_rem"],
            r["lives"],
            r["bonus"],
            r["score"],
            r["fruit1"],
            r["fruit2"],
            r["fruit3"],
            r["fruit4"],
            done,
            trunc,
            stall,
        ]
        print("  ".join(f"{v:>5d}" for v in vals))

    print_row(0, initial, 0, 0, 0)

    for f in range(1, args.frames + 1):
        _obs, _r, done, truncated, _info = base.step(args.action)
        cur = _read_all(iface)
        if cur["bonus"] != prev_bonus:
            stall_c = 0
            prev_bonus = cur["bonus"]
        else:
            stall_c += 1
        print_row(f, cur, int(done), int(truncated), stall_c)


if __name__ == "__main__":
    main()
