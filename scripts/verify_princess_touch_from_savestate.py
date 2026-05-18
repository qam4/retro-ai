#!/usr/bin/env python3
"""Load the user's near-princess save state and walk RIGHT until either
the agent touches the princess or dies. Per-frame RAM trace is dumped.

The state in `debug/Yeti (1984) (Loriciels)_slot0.sav` is a v3 Crayon
save written by the SDL frontend after manually navigating to floor 5,
all four fruits collected, ~32 px left of the princess centre. There
are no snowballs between the agent and princess on floor 5, so the
expected outcome is a princess touch within ~30 frames.

Detection rule under test (mirrors `scripts/train_segment.py`):

    rising edge of byte 11050 (level-cleared flag): 0 -> 1

Verified empirically in `scripts/probe_princess_flag_long_baseline.py`:
the flag stayed at 0 across 26k frames of varied non-touch gameplay.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make sure the repo root is on the path when running as a script.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "build" / "ci-linux"))

from retro_ai.training.env_builder import build_training_env  # noqa: E402
from retro_ai.training.run_config import EnvConfig  # noqa: E402

# Yeti RAM addresses (see scripts/walk_to_princess.py for cross-ref).
X_ADDR = 11090
Y_ADDR = 11089
LIVES_ADDR = 11095
FRUITS_ADDR = 11055
BONUS_HI = 11010
BONUS_LO = 11011
SCORE_HI = 11093
SCORE_LO = 11094
PRINCESS_FLAG_ADDR = 11050


def read_state(iface) -> dict[str, int]:
    return {
        "x": iface.read_ram_byte(X_ADDR),
        "y": iface.read_ram_byte(Y_ADDR),
        "fr": iface.read_ram_byte(FRUITS_ADDR),
        "lv": iface.read_ram_byte(LIVES_ADDR),
        "bonus": (iface.read_ram_byte(BONUS_HI) << 8) | iface.read_ram_byte(BONUS_LO),
        "score": (iface.read_ram_byte(SCORE_HI) << 8) | iface.read_ram_byte(SCORE_LO),
        "flag": iface.read_ram_byte(PRINCESS_FLAG_ADDR),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--state",
        type=Path,
        default=ROOT / "debug" / "Yeti (1984) (Loriciels)_slot0.sav",
        help="Path to the .sav near-princess state",
    )
    parser.add_argument(
        "--max-frames", type=int, default=120, help="Max frames to walk right"
    )
    args = parser.parse_args()

    cfg = EnvConfig(
        profile="yeti_fruit",
        action_mode="joystick",
        max_steps=10_000,
        stall_threshold=10_000,
        resize=(84, 84),
    )
    stack = build_training_env("yeti_fruit", cfg)
    base = stack.base
    gym_env = stack.gym
    base.reset(seed=0)

    data = args.state.read_bytes()
    print(f"loading state: {args.state} ({len(data)} bytes)")
    base._interface.load_state(data)

    # Settle a few frames after load (frame_stack isn't reset on load_state).
    iface = base._interface
    for _ in range(5):
        gym_env.step([0, 0, 0])

    init = read_state(iface)
    print(f"initial state: {init}")
    print(f"agent centre: px=({init['x'] * 4 + 8}, {init['y'] + 8})")
    print("princess centre: px=(312, 60)")
    dx = 312 - (init["x"] * 4 + 8)
    print(f"distance to princess centre: dx={dx}")
    print()

    # Walk right with joystick action [no-fire, dx=+1, dy=0].
    action = [0, 1, 0]
    print(
        f"{'fr':>4} {'x':>3} {'y':>3} {'frR':>3} {'lv':>2} "
        f"{'bonus':>5} {'score':>5} {'flag':>4}"
    )
    prev = init
    print(
        f"{0:>4} {prev['x']:>3} {prev['y']:>3} {prev['fr']:>3} "
        f"{prev['lv']:>2} {prev['bonus']:>5} {prev['score']:>5} "
        f"{prev['flag']:>4}"
    )

    for f in range(1, args.max_frames + 1):
        gym_env.step(action)
        cur = read_state(iface)

        # Print only when something interesting changes.
        changed = (cur != prev) or (f % 10 == 0)
        if changed:
            print(
                f"{f:>4} {cur['x']:>3} {cur['y']:>3} {cur['fr']:>3} "
                f"{cur['lv']:>2} {cur['bonus']:>5} {cur['score']:>5} "
                f"{cur['flag']:>4}"
            )

        # Detection rule under test: rising edge of the level-cleared flag.
        princess_touched = cur["flag"] == 1 and prev["flag"] == 0
        if princess_touched:
            print()
            print(f"*** PRINCESS TOUCH DETECTED at frame {f} ***")
            print(
                f"    flag rising edge: {prev['flag']} -> {cur['flag']}, "
                f"score {prev['score']} -> {cur['score']}, "
                f"bonus {prev['bonus']} -> {cur['bonus']}"
            )
            return 0

        if cur["lv"] < prev["lv"]:
            print()
            print(
                f"!!! LIFE LOST at frame {f} (lives {prev['lv']} -> "
                f"{cur['lv']}). agent died before touching princess."
            )
            return 1

        prev = cur

    print()
    print(f"!!! Walked {args.max_frames} frames without touching the princess.")
    print(f"    final state: {prev}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
