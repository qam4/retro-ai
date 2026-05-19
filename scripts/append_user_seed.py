#!/usr/bin/env python3
"""Append a user-curated CP4 save state to the seed pool.

The default in/out is the v9-derived pool in
``output/mo5/yeti/seeds/v9_v4_cp4enriched.pkl``. The user's save
file written by the Crayon SDL frontend gets validated (probe=120)
to confirm it's not in a transient/dying state, then prepended to
the CP4 bucket (slot index 4) of the pool.

We don't run the per-position cap here: the user's seed is the
unique near-princess high-quality state and we want it kept
verbatim.
"""

from __future__ import annotations

import argparse
import hashlib
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "build" / "ci-linux"))

from retro_ai.training.env_builder import build_training_env  # noqa: E402
from retro_ai.training.run_config import EnvConfig  # noqa: E402
from retro_ai.training.state_validator import validate_state  # noqa: E402

FRUITS_ADDR = 11055
LIVES_ADDR = 11095
X_ADDR = 11090
Y_ADDR = 11089
BONUS_HI = 11010
BONUS_LO = 11011


def _hash(b: bytes) -> str:
    return hashlib.blake2b(b, digest_size=8).hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--state",
        type=Path,
        default=ROOT / "debug" / "Yeti (1984) (Loriciels)_slot0.sav",
        help="User save state file to append",
    )
    p.add_argument(
        "--in-pool",
        type=Path,
        default=ROOT / "output/mo5/yeti/seeds/v9_v4_cp4enriched.pkl",
    )
    p.add_argument(
        "--out-pool",
        type=Path,
        default=ROOT / "output/mo5/yeti/seeds/v9_v5_cp4_user_seed.pkl",
    )
    p.add_argument("--probe-frames", type=int, default=120)
    args = p.parse_args()

    state_bytes = args.state.read_bytes()
    h = _hash(state_bytes)
    print(f"loading {args.state} ({len(state_bytes)} bytes, hash={h})")

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

    print(f"validating with probe_frames={args.probe_frames}...")
    result = validate_state(
        state_bytes=state_bytes,
        load_state=_load,
        step_noop=_step_noop,
        probe_frames=args.probe_frames,
    )
    if not result.viable:
        print(f"VALIDATION FAILED: {result.reason}")
        return 1

    iface = base._interface
    fr = iface.read_ram_byte(FRUITS_ADDR)
    lv = iface.read_ram_byte(LIVES_ADDR)
    bonus = (iface.read_ram_byte(BONUS_HI) << 8) | iface.read_ram_byte(BONUS_LO)
    x = iface.read_ram_byte(X_ADDR)
    y = iface.read_ram_byte(Y_ADDR)
    print(
        f"validated. post-settle state: x={x} y={y} (px={x * 4 + 8}, {y + 8}) "
        f"fruits_remaining={fr} lives={lv} bonus={bonus}"
    )

    if fr != 0:
        print("ERROR: post-settle fruits_remaining != 0; not a CP4 state")
        return 1

    with args.in_pool.open("rb") as f:
        pool = pickle.load(f)
    cp4 = list(pool["checkpoints"][4])

    # Dedupe by hash.
    existing_hashes = {_hash(bytes(s)) for s in cp4}
    if h in existing_hashes:
        print(f"already in pool ({h}); nothing to do")
        return 0

    cp4.insert(0, state_bytes)
    pool["checkpoints"] = list(pool["checkpoints"])
    pool["checkpoints"][4] = cp4
    pool.setdefault("stats", {}).setdefault("saves", [0] * 5)
    pool["stats"]["saves"][4] = len(cp4)

    args.out_pool.parent.mkdir(parents=True, exist_ok=True)
    with args.out_pool.open("wb") as f:
        pickle.dump(pool, f)
    print(f"wrote {args.out_pool}")
    print(f"  per-CP sizes: {[len(b) for b in pool['checkpoints']]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
