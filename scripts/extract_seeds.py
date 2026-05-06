#!/usr/bin/env python3
"""Extract per-CP save-state bags from an archive.

Bridges the gap between Go-Explore archive formats (dict keyed by
cell) and the curriculum/segment format (dict with ``checkpoints``: a
list[list[bytes]], one list per CP level 0..4).

Usage
-----

Extract all CPs from a Go-Explore archive (new or old format),
validate, and write the curriculum-format file::

    python scripts/extract_seeds.py \\
        output/mo5/yeti/go_explore_v9/archive.pkl \\
        --out output/mo5/yeti/seeds/v9_checkpoints.pkl

Then point ``train_segment.py`` at it::

    segment:
      checkpoints: output/mo5/yeti/seeds/v9_checkpoints.pkl
      segment: 1

Validation uses
:func:`retro_ai.training.state_validator.validate_state` by default;
states that fail are dropped. Pass ``--no-validate`` to skip.

Archive formats supported (auto-detected):
  - ``cell_key[2]`` is a ``frozenset`` of floor numbers (v9+ scheme).
  - ``cell_key[2]`` is an int ``fruits_remaining`` (older Go-Explore).
  - Already in curriculum format (``{"checkpoints": list[list[bytes]]}``) —
    pass-through with optional validation.
"""

from __future__ import annotations

import argparse
import os
import pickle
from typing import Dict, List


def _cp_for_cell(cell_key):
    """Return the CP level (number of fruits collected) for a cell key."""
    if len(cell_key) < 3:
        return None
    v = cell_key[2]
    if isinstance(v, (frozenset, set, list, tuple)):
        return len(v)
    if isinstance(v, int) and 0 <= v <= 4:
        return 4 - v
    return None


def _load_archive_as_cp_buckets(path: str) -> Dict[int, List[bytes]]:
    """Read ``path`` and return ``{cp_level: [state_bytes, ...]}``."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    buckets: Dict[int, List[bytes]] = {i: [] for i in range(5)}
    if isinstance(data, dict) and "checkpoints" in data:
        # Curriculum format — already bucketed.
        for cp, states in enumerate(data["checkpoints"]):
            buckets[cp] = [bytes(s) for s in states]
        return buckets
    # Go-Explore archive format.
    for cell_key, entry in data.items():
        cp = _cp_for_cell(cell_key)
        if cp is None:
            continue
        buckets[cp].append(bytes(entry["state"]))
    return buckets


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "archive_path",
        help="Source archive (Go-Explore or curriculum format).",
    )
    p.add_argument(
        "--out",
        required=True,
        help="Output curriculum-format checkpoints.pkl.",
    )
    p.add_argument(
        "--profile",
        default="yeti_fruit",
        help="Game profile used to build the validation env.",
    )
    p.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip viability validation (keep every state from the archive).",
    )
    p.add_argument(
        "--bonus-hi",
        type=int,
        default=11010,
        help="RAM high byte for the countdown (Yeti default).",
    )
    p.add_argument(
        "--bonus-lo",
        type=int,
        default=11011,
        help="RAM low byte for the countdown (Yeti default).",
    )
    args = p.parse_args()

    print(f"Reading {args.archive_path}", flush=True)
    buckets = _load_archive_as_cp_buckets(args.archive_path)
    for cp in sorted(buckets):
        print(f"  CP{cp}: {len(buckets[cp])} raw states")

    if not args.no_validate:
        # Build a single env for validation.
        from retro_ai.training.env_builder import build_training_env
        from retro_ai.training.run_config import EnvConfig
        from retro_ai.training.state_validator import validate_state

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

        def _load(state_bytes):
            base._interface.load_state(state_bytes)

        def _step_noop():
            base.step([0, 0, 0])

        def _read_bonus():
            i = base._interface
            return (i.read_ram_byte(args.bonus_hi) << 8) | i.read_ram_byte(
                args.bonus_lo
            )

        print("\nValidating states…")
        kept: Dict[int, List[bytes]] = {i: [] for i in range(5)}
        for cp in sorted(buckets):
            for st in buckets[cp]:
                result = validate_state(
                    state_bytes=st,
                    load_state=_load,
                    step_noop=_step_noop,
                    read_counter=_read_bonus,
                )
                if result.viable:
                    kept[cp].append(st)
            print(f"  CP{cp}: kept {len(kept[cp])}/{len(buckets[cp])}")
        buckets = kept

    # Write out in curriculum format.
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out_data = {
        "checkpoints": [buckets[i] for i in range(5)],
        "stats": {"saves": [len(buckets[i]) for i in range(5)]},
    }
    with open(args.out, "wb") as f:
        pickle.dump(out_data, f)
    print(f"\nWrote {args.out}")
    print(f"  final sizes: {[len(buckets[i]) for i in range(5)]}")


if __name__ == "__main__":
    main()
