#!/usr/bin/env python3
"""Dump the 84x84 grayscale observation the agent actually sees, for a
CP0 state (all 4 fruits present) and a CP3 state (3 collected), so we
can judge whether fruit presence is visible at the policy's input
resolution.

Writes side-by-side upscaled PNGs to debug/.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "build" / "ci-linux"))

from retro_ai.training.env_builder import build_training_env  # noqa: E402
from retro_ai.training.run_config import EnvConfig  # noqa: E402


def save_obs(obs, raw, tag):
    # obs is HxWxC (frame stack). Take the last frame.
    arr = np.asarray(obs)
    if arr.ndim == 3:
        frame = arr[:, :, -1]
    else:
        frame = arr
    img = Image.fromarray(frame.astype(np.uint8), mode="L")
    img = img.resize((84 * 6, 84 * 6), Image.NEAREST)
    out = ROOT / "debug" / f"obs_{tag}.png"
    img.save(out)
    print(f"  wrote {out} (obs frame {frame.shape})")
    if raw is not None:
        rawimg = Image.fromarray(np.asarray(raw, dtype=np.uint8))
        rawout = ROOT / "debug" / f"raw_{tag}.png"
        rawimg.save(rawout)
        print(f"  wrote {rawout} (raw {np.asarray(raw).shape})")


def main() -> int:
    cfg = EnvConfig(
        profile="yeti_fruit",
        action_mode="joystick",
        max_steps=2000,
        stall_threshold=2000,
        resize=(84, 84),
    )
    stack = build_training_env("yeti_fruit", cfg)
    base = stack.base
    gym_env = stack.gym

    # CP0: fresh reset, all fruits present.
    obs, _ = gym_env.reset(seed=0)
    for _ in range(60):
        obs, _, _, _, _ = gym_env.step([0, 0, 0])
    ram = base._interface.read_ram()
    print(f"CP0: fruits_remaining={ram[11055]} pos=({ram[11090]},{ram[11089]})")
    save_obs(obs, base._last_raw_obs, "cp0_4fruits")

    # CP3: load from the v3 checkpoint pool.
    ckpt = ROOT / "output/mo5/yeti/training/yeti_curriculum_v3/checkpoints.pkl"
    if ckpt.exists():
        cp = pickle.load(ckpt.open("rb"))["checkpoints"]
        if cp[3]:
            _b, seed = cp[3][0]
            base._interface.load_state(bytes(seed))
            for _ in range(5):
                obs, _, _, _, _ = gym_env.step([0, 0, 0])
            ram = base._interface.read_ram()
            print(
                f"CP3: fruits_remaining={ram[11055]} "
                f"pos=({ram[11090]},{ram[11089]})"
            )
            save_obs(obs, base._last_raw_obs, "cp3_1fruit")
        else:
            print("CP3 pool empty")
    else:
        print(f"no checkpoint file at {ckpt}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
