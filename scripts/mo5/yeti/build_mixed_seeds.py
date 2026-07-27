#!/usr/bin/env python3
"""Build a "mixed" seed pool combining seeds from multiple checkpoint
buckets into a single bucket.

Use case: train_segment.py samples ``data["checkpoints"][segment]``
uniformly per episode. To train one policy on a mix of CP3 and CP4
states, write both pools into the same bucket so each episode gets
sampled from the union.

Default writes the union of CP3 and CP4 seeds into bucket 3 of the
output pool. Other buckets are zeroed out so any accidental misuse
fails loudly.

Why we do this: segment_3toP_v1 (warm-started from segment_4toP_v2)
catastrophically forgot CP4->princess after 5M steps because every
episode started from CP3 (where picking F4 is the easy local
optimum). Mixing keeps the harder CP4->princess practice in the
distribution.
"""

from __future__ import annotations

import argparse
import hashlib
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _hash(b: bytes) -> str:
    return hashlib.blake2b(b, digest_size=8).hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--cp3-pool",
        type=Path,
        default=ROOT / "output/mo5/yeti/seeds/v9_v3_cp3enriched.pkl",
    )
    p.add_argument(
        "--cp4-pool",
        type=Path,
        default=ROOT / "output/mo5/yeti/seeds/v9_v5_cp4_user_seed.pkl",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "output/mo5/yeti/seeds/v10_cp3_plus_cp4_mix.pkl",
    )
    p.add_argument(
        "--target-bucket",
        type=int,
        default=3,
        help="Index of the output pool bucket to write the union into. "
        "Use 3 so train_segment.py with segment:3 picks it up.",
    )
    p.add_argument(
        "--cp3-replicas",
        type=int,
        default=1,
        help="How many copies of CP3 seeds to put in the union. Default 1.",
    )
    p.add_argument(
        "--cp4-replicas",
        type=int,
        default=8,
        help=(
            "How many copies of CP4 seeds to put in the union. Default 8 "
            "balances 8x8=64 CP4 entries against ~128 CP3 entries (≈ 1:2 "
            "CP4:CP3 sample weight)."
        ),
    )
    args = p.parse_args()

    with args.cp3_pool.open("rb") as f:
        cp3 = pickle.load(f)["checkpoints"][3]
    with args.cp4_pool.open("rb") as f:
        cp4 = pickle.load(f)["checkpoints"][4]
    print(f"CP3 source: {args.cp3_pool} ({len(cp3)} seeds)")
    print(f"CP4 source: {args.cp4_pool} ({len(cp4)} seeds)")

    union: list[bytes] = []
    seen: set[str] = set()
    for state in cp3:
        b = bytes(state)
        h = _hash(b)
        if h in seen:
            continue
        seen.add(h)
        for _ in range(args.cp3_replicas):
            union.append(b)
    for state in cp4:
        b = bytes(state)
        h = _hash(b)
        if h in seen:
            # CP4 seeds shouldn't collide with CP3 in practice.
            continue
        seen.add(h)
        for _ in range(args.cp4_replicas):
            union.append(b)

    print(
        f"union: {len(union)} entries "
        f"(unique pre-replica = CP3 {len(cp3)} + CP4 {len(cp4)} = {len(cp3)+len(cp4)})"
    )

    # Build the output pool: empty buckets except for target.
    checkpoints = [[] for _ in range(5)]
    checkpoints[args.target_bucket] = union
    saves = [0] * 5
    saves[args.target_bucket] = len(union)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as f:
        pickle.dump({"checkpoints": checkpoints, "stats": {"saves": saves}}, f)
    print(f"wrote {args.out}: per-bucket sizes = {[len(b) for b in checkpoints]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
