#!/usr/bin/env python3
"""Filter a Go-Explore-style archive.pkl to keep only viable states.

Reads ``archive.pkl`` (a dict mapping cell keys to
``{"state": bytes, ...}`` entries), runs each state through
:func:`retro_ai.training.state_validator.validate_state`, and writes a
new archive containing only the viable entries.

Usage
-----

::

    python scripts/filter_archive.py \\
        output/.../go_explore_fruit/archive.pkl \\
        --profile yeti_fruit \\
        --out output/.../go_explore_fruit/archive_validated.pkl

Without ``--out`` the filtered archive is written alongside the input as
``<stem>_validated.pkl``.
"""

from __future__ import annotations

import argparse
import os
import pickle
from collections import Counter

from retro_ai.training.env_builder import build_training_env
from retro_ai.training.run_config import EnvConfig
from retro_ai.training.state_validator import validate_state


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("archive_path", help="Path to the source archive.pkl.")
    p.add_argument(
        "--profile",
        default="yeti_fruit",
        help="Game profile used to build the validation env.",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output archive path. Defaults to <input>_validated.pkl alongside.",
    )
    p.add_argument("--settle-frames", type=int, default=5)
    p.add_argument("--probe-frames", type=int, default=120)
    args = p.parse_args()

    with open(args.archive_path, "rb") as f:
        archive = pickle.load(f)
    if not isinstance(archive, dict):
        raise SystemExit(
            f"Unexpected archive format at {args.archive_path!r}: "
            f"expected dict, got {type(archive).__name__}"
        )

    # Build the validation env once.
    env_cfg = EnvConfig(
        profile=args.profile,
        action_mode="joystick",
        max_steps=1000,
        stall_threshold=15,
        resize=(84, 84),
    )
    stack = build_training_env(args.profile, env_cfg)
    base = stack.base
    base.reset(seed=0)

    def load_state(state_bytes: bytes) -> None:
        base._interface.load_state(state_bytes)

    def step_noop() -> bool:
        _, _, done, _, _ = base.step([0, 0, 0])
        return bool(done)

    filtered: dict = {}
    reasons: Counter = Counter()
    for key, entry in archive.items():
        state_bytes = entry["state"]
        result = validate_state(
            state_bytes=state_bytes,
            load_state=load_state,
            step_noop=step_noop,
            settle_frames=args.settle_frames,
            probe_frames=args.probe_frames,
        )
        reasons[result.reason] += 1
        if result.viable:
            filtered[key] = entry

    out_path = args.out
    if out_path is None:
        stem, ext = os.path.splitext(args.archive_path)
        out_path = f"{stem}_validated{ext}"

    with open(out_path, "wb") as f:
        pickle.dump(filtered, f)

    total = len(archive)
    kept = len(filtered)
    print(f"Read   {total:5d} cells from {args.archive_path}")
    print(f"Kept   {kept:5d} viable cells ({100 * kept / total:.0f}%)")
    print("Reasons:")
    for reason in sorted(reasons):
        print(f"  {reason:15s} {reasons[reason]:5d}")


if __name__ == "__main__":
    main()
