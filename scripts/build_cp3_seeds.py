#!/usr/bin/env python3
"""Build a CP3 seed pool by combining go-explore CP3 states with
collected_states from a per-segment training run.

Both sources are validated via state_validator with the standard
probe (5 settle, 120 probe) before being merged. Duplicates by
state-byte hash are de-duplicated.
"""
from __future__ import annotations

import argparse
import hashlib
import pickle

from retro_ai.training.env_builder import build_training_env
from retro_ai.training.run_config import EnvConfig
from retro_ai.training.state_validator import validate_state


FRUITS_ADDR = 11055


def _hash(b: bytes) -> str:
    return hashlib.blake2b(b, digest_size=8).hexdigest()


def _load_v9_cp3(path: str):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return [bytes(s) for s in data["checkpoints"][3]]


def _load_collected(path: str):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return [bytes(s) for s in data["states"]]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--v9", default="output/mo5/yeti/seeds/v9_checkpoints_v2.pkl")
    p.add_argument(
        "--collected",
        default="output/mo5/yeti/training/segment_2to3_v7/collected_states.pkl",
    )
    p.add_argument("--out", required=True)
    p.add_argument("--probe-frames", type=int, default=120)
    p.add_argument(
        "--top-per-group",
        type=int,
        default=50,
        help=(
            "Cap the number of seeds kept per remaining-fruit group "
            "to the top-N by post-settle bonus. 0 to keep all."
        ),
    )
    args = p.parse_args()

    v9 = _load_v9_cp3(args.v9)
    coll = _load_collected(args.collected)
    print(f"v9 CP3 (already validated): {len(v9)}")
    print(f"collected states (raw):     {len(coll)}")

    # Combine, dedupe by hash.
    seen = set()
    combined = []
    for s in v9 + coll:
        h = _hash(s)
        if h in seen:
            continue
        seen.add(h)
        combined.append(s)
    print(f"after dedupe:               {len(combined)}")

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

    valid_cp3 = []
    rejected = 0
    drift = 0
    for s in combined:
        # Run validator (load + 5 settle + 120 probe).
        result = validate_state(
            state_bytes=s,
            load_state=_load,
            step_noop=_step_noop,
            probe_frames=args.probe_frames,
        )
        if not result.viable:
            rejected += 1
            continue
        # After settle, confirm fruits_remaining == 1 (CP3).
        # validate_state already loaded + settled.
        fr = base._interface.read_ram_byte(FRUITS_ADDR)
        if fr != 1:
            drift += 1
            continue
        valid_cp3.append(s)

    print(f"validated CP3:              {len(valid_cp3)}")
    print(f"  rejected by validator:    {rejected}")
    print(f"  drifted off CP3 in settle: {drift}")

    # Quality filter: re-load each seed, snapshot bonus + remaining
    # fruit subset. Cap each remaining-fruit group at the top
    # ``top_per_group`` by bonus (= more game-time available, better
    # starting state for the next segment).
    if args.top_per_group > 0:
        FRUIT_PRESENCE = {1: 0x2FAD, 2: 0x2F00, 3: 0x2E68, 4: 0x2DD8}
        BONUS_HI = 11010
        BONUS_LO = 11011

        scored = []  # (bonus, remaining_tuple, state_bytes)
        for s in valid_cp3:
            base._interface.load_state(s)
            for _ in range(5):
                base.step([0, 0, 0])
            bonus = (
                base._interface.read_ram_byte(BONUS_HI) << 8
            ) | base._interface.read_ram_byte(BONUS_LO)
            fp = tuple(
                base._interface.read_ram_byte(FRUIT_PRESENCE[i]) != 0
                for i in (1, 2, 3, 4)
            )
            remaining = tuple(j + 1 for j, p in enumerate(fp) if p)
            scored.append((bonus, remaining, s))

        groups: dict = {}
        for bonus, rem, s in scored:
            groups.setdefault(rem, []).append((bonus, s))

        kept = []
        print()
        print(
            f"  applying top-{args.top_per_group}-per-group selection "
            f"(by post-settle bonus):"
        )
        for rem in sorted(groups):
            entries = sorted(groups[rem], key=lambda e: -e[0])
            chosen = entries[: args.top_per_group]
            print(
                f"    remaining={rem}: kept {len(chosen)}/{len(entries)} "
                f"(bonus range {chosen[-1][0] if chosen else '-'}-"
                f"{chosen[0][0] if chosen else '-'})"
            )
            kept.extend([s for _b, s in chosen])
        valid_cp3 = kept
        print(f"  final CP3 pool: {len(valid_cp3)}")

    # Write out as curriculum format: same shape as v9_v2, but only
    # populate CP3 (other levels copied through from v9 unchanged).
    with open(args.v9, "rb") as f:
        v9_data = pickle.load(f)
    out = {
        "checkpoints": list(v9_data["checkpoints"]),
        "stats": {"saves": list(v9_data.get("stats", {}).get("saves", [0] * 5))},
    }
    out["checkpoints"][3] = valid_cp3
    out["stats"]["saves"][3] = len(valid_cp3)
    with open(args.out, "wb") as f:
        pickle.dump(out, f)
    print(f"Wrote {args.out}: per-CP sizes = "
          f"{[len(b) for b in out['checkpoints']]}")


if __name__ == "__main__":
    main()
