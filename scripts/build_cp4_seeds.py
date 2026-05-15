#!/usr/bin/env python3
"""Build a CP4 seed pool from v8's collected_states + v9's CP4 cells.

Same approach as ``build_cp3_seeds.py``: combine sources, dedupe by
hash, validate (probe=120), then optionally rank and cap by post-
settle bonus per (x, y) bucket.

Unlike CP3, CP4 states are dominated by a single position
(post-fruit-pickup). Quality-filtering by bonus per-position
captures diversity if any exists; no remaining-fruit grouping is
needed (CP4 = no fruits remaining = single category).
"""
from __future__ import annotations

import argparse
import hashlib
import pickle
from collections import Counter

from retro_ai.training.env_builder import build_training_env
from retro_ai.training.run_config import EnvConfig
from retro_ai.training.state_validator import validate_state


FRUITS_ADDR = 11055
BONUS_HI = 11010
BONUS_LO = 11011
X_ADDR = 11090
Y_ADDR = 11089


def _hash(b: bytes) -> str:
    return hashlib.blake2b(b, digest_size=8).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--v9", default="output/mo5/yeti/seeds/v9_v3_cp3enriched.pkl")
    p.add_argument(
        "--collected",
        default="output/mo5/yeti/training/segment_3to4_v1/collected_states.pkl",
    )
    p.add_argument("--out", required=True)
    p.add_argument("--probe-frames", type=int, default=120)
    p.add_argument(
        "--top-per-position",
        type=int,
        default=20,
        help="Per (x, y) bucket, keep at most this many seeds (highest bonus).",
    )
    args = p.parse_args()

    # Source 1: v9's CP4 from the latest enriched archive.
    with open(args.v9, "rb") as f:
        v9_data = pickle.load(f)
    v9_cp4 = [bytes(s) for s in v9_data["checkpoints"][4]]

    # Source 2: collected_states from segment_3to4_v1 (CP4 reaches).
    with open(args.collected, "rb") as f:
        coll = [bytes(s) for s in pickle.load(f)["states"]]

    print(f"v9 CP4: {len(v9_cp4)}")
    print(f"collected (raw): {len(coll)}")

    # Dedupe.
    seen = set()
    combined = []
    for s in v9_cp4 + coll:
        h = _hash(s)
        if h in seen:
            continue
        seen.add(h)
        combined.append(s)
    print(f"after dedupe: {len(combined)}")

    cfg = EnvConfig(
        profile="yeti_fruit",
        action_mode="joystick",
        max_steps=1000,
        stall_threshold=15,
        resize=(84, 84),
    )
    stack = build_training_env("yeti_fruit", cfg)
    base = stack.base
    base.reset(seed=0)

    def _load(s):
        base._interface.load_state(s)

    def _step_noop():
        _, _, done, _, _ = base.step([0, 0, 0])
        return bool(done)

    valid = []
    drift = 0
    rejected = 0
    for s in combined:
        result = validate_state(
            state_bytes=s,
            load_state=_load,
            step_noop=_step_noop,
            probe_frames=args.probe_frames,
        )
        if not result.viable:
            rejected += 1
            continue
        # Confirm fruits_remaining == 0 (CP4) after settle.
        fr = base._interface.read_ram_byte(FRUITS_ADDR)
        if fr != 0:
            drift += 1
            continue
        valid.append(s)

    print(f"validated CP4: {len(valid)}")
    print(f"  rejected: {rejected}")
    print(f"  drifted off CP4 in settle: {drift}")

    # Quality cap per (x, y) bucket by post-settle bonus.
    if args.top_per_position > 0:
        scored = []
        for s in valid:
            base._interface.load_state(s)
            for _ in range(5):
                base.step([0, 0, 0])
            x = base._interface.read_ram_byte(X_ADDR)
            y = base._interface.read_ram_byte(Y_ADDR)
            bonus = (
                base._interface.read_ram_byte(BONUS_HI) << 8
            ) | base._interface.read_ram_byte(BONUS_LO)
            scored.append((bonus, (x, y), s))

        groups = {}
        for bonus, pos, s in scored:
            groups.setdefault(pos, []).append((bonus, s))
        print()
        print(f"  applying top-{args.top_per_position}-per-position cap:")
        kept = []
        for pos in sorted(groups):
            entries = sorted(groups[pos], key=lambda e: -e[0])
            chosen = entries[: args.top_per_position]
            print(
                f"    pos={pos}: kept {len(chosen)}/{len(entries)} "
                f"(bonus {chosen[-1][0] if chosen else '-'}-"
                f"{chosen[0][0] if chosen else '-'})"
            )
            kept.extend([s for _b, s in chosen])
        valid = kept
        print(f"  final CP4 pool: {len(valid)}")

    # Write out: copy lower-CP entries from v9, replace CP4.
    out = {
        "checkpoints": list(v9_data["checkpoints"]),
        "stats": {"saves": list(v9_data.get("stats", {}).get("saves", [0] * 5))},
    }
    out["checkpoints"][4] = valid
    out["stats"]["saves"][4] = len(valid)
    with open(args.out, "wb") as f:
        pickle.dump(out, f)
    print(f"\nWrote {args.out}: per-CP sizes = "
          f"{[len(b) for b in out['checkpoints']]}")


if __name__ == "__main__":
    main()
