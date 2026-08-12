#!/usr/bin/env python3
"""Is the emulator's episode start faithful, and do two core builds agree?

Background (full write-up: experiments/003-yeti/core_provenance_2b0a45d.md).
``MO5RLInterface::reset()`` boots the emulator only on the FIRST reset; every
later episode is restored from a cached ``startup_state_``. So the save/load
path is on the critical path of ALL training and eval episodes, and a change to
state-restore semantics silently changes what every policy is trained against.
That is how commit ``2b0a45d`` turned the L1 champion from 97.5% princess into
0% without touching physics.

This probe removes the policy from the loop: it drives a FIXED action sequence
and snapshots full RAM (and optionally every frame) per step, so any difference
is the emulator's, not the agent's.

Three modes
-----------
``capture``    Record episode 1 (real boot) and episode 2 (startup-state
               restore) to an ``.npz``. Run it once per core build.

``selfcheck``  Boot vs restore WITHIN one build. On a correct core these are
               bit-identical; divergence means the cached startup state does not
               reproduce a real boot, which invalidates every policy trained on
               it. Needs only ONE build, so this is the cheap standing guard --
               exits non-zero on failure.

``compare``    Two captures from two different core builds, reported separately
               for the boot path and the restore path. This is what attributes a
               regression to the restore rather than to emulator dynamics.

Examples
--------
Capture from two builds and attribute the difference::

    env PYTHONPATH=python:out/oldcore-2b0a45d/build/ci-linux RETRO_AI_ROM_DIR=roms \\
      python3 scripts/mo5/yeti/core_determinism_probe.py capture --out debug/ram_old.npz
    env PYTHONPATH=python:build/ci-linux RETRO_AI_ROM_DIR=roms \\
      python3 scripts/mo5/yeti/core_determinism_probe.py capture --out debug/ram_new.npz
    python3 scripts/mo5/yeti/core_determinism_probe.py compare \\
      --a debug/ram_old.npz --b debug/ram_new.npz

Guard the current build (should print OK and exit 0)::

    env PYTHONPATH=python:build/ci-linux RETRO_AI_ROM_DIR=roms \\
      python3 scripts/mo5/yeti/core_determinism_probe.py selfcheck \\
        --out debug/ram_head.npz
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

import numpy as np

HUD_ROWS = 16  # Yeti's HUD strip occupies rows 0..15; play area starts at y=16.


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------


def capture(
    out_path: str,
    steps: int,
    profile: str,
    with_frames: bool,
    action: Optional[list] = None,
) -> dict:
    """Record RAM (and optionally frames) per step for episode 1 and episode 2.

    Episode 1 is the only real emulator boot; episode 2 exercises the cached
    startup-state restore. Actions are fixed, so the two episodes are
    comparable and any difference belongs to the emulator.
    """
    from retro_ai.training.env_builder import build_training_env
    from retro_ai.training.run_config import EnvConfig

    stack = build_training_env(
        profile,
        EnvConfig(
            profile=profile,
            action_mode="joystick",
            max_steps=max(steps + 1, 1000),
            resize=(84, 84),
        ),
    )
    iface = stack.base._interface
    act = np.array(action if action else [0, 0, 0])

    payload: dict = {}
    for episode in (1, 2):
        stack.gym.reset()
        rams, frames = [], []
        for _ in range(steps):
            stack.gym.step(act)
            rams.append(np.frombuffer(bytes(iface.read_ram()), dtype=np.uint8).copy())
            if with_frames:
                frames.append(np.array(stack.base._last_raw_obs, copy=True))
        payload[f"ram_ep{episode}"] = np.stack(rams)
        if with_frames:
            payload[f"frames_ep{episode}"] = np.stack(frames)
        print(
            f"  episode {episode} "
            f"({'real boot' if episode == 1 else 'startup-state restore'}): "
            f"{len(rams)} steps x {rams[0].size} RAM bytes"
        )

    payload["meta_steps"] = np.array([steps])
    payload["meta_action"] = act
    np.savez_compressed(out_path, **payload)
    print(f"wrote {out_path}")
    return payload


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _describe(a: np.ndarray, b: np.ndarray, indent: str = "   ") -> bool:
    """Print how two per-step RAM series differ. Returns True when identical."""
    diff = a != b
    per_step = diff.sum(axis=1)
    steps = np.nonzero(per_step)[0]
    if len(steps) == 0:
        print(f"{indent}IDENTICAL for all {len(a)} steps")
        return True

    first = int(steps[0])
    addrs = np.nonzero(diff[first])[0]
    print(f"{indent}DIVERGES at step {first} ({len(addrs)} byte(s)):")
    for x in addrs[:6]:
        print(f"{indent}  0x{x:04X} ({x}): {a[first][x]} vs {b[first][x]}")
    print(f"{indent}steps affected: {len(steps)}/{len(a)}")
    ever = np.nonzero(diff.any(axis=0))[0]
    print(f"{indent}addresses that ever differ: {len(ever)}")
    frozen = [
        x for x in ever if len(np.unique(a[:, x])) == 1 or len(np.unique(b[:, x])) == 1
    ]
    if frozen:
        print(
            f"{indent}FROZEN on one side (inert counter/RNG -- the signature of a "
            f"broken restore): " + ", ".join(f"0x{x:04X}" for x in frozen[:6])
        )
    return False


def _describe_frames(a: np.ndarray, b: np.ndarray, indent: str = "   ") -> None:
    """Split pixel differences into HUD chrome vs play area."""
    n = min(len(a), len(b))
    first_hud = first_play = None
    for i in range(n):
        d = (a[i] != b[i]).any(axis=2)
        if first_hud is None and d[:HUD_ROWS].any():
            first_hud = i
        if first_play is None and d[HUD_ROWS:].any():
            first_play = i
        if first_hud is not None and first_play is not None:
            break
    print(f"{indent}first HUD-strip (y<{HUD_ROWS}) pixel difference : {first_hud}")
    print(f"{indent}first PLAY-area (y>={HUD_ROWS}) pixel difference: {first_play}")
    if first_play is not None:
        d = (a[first_play] != b[first_play]).any(axis=2)
        ys, xs = np.where(d[HUD_ROWS:])
        print(
            f"{indent}  play-area bbox at step {first_play}: "
            f"y=[{ys.min() + HUD_ROWS},{ys.max() + HUD_ROWS}] x=[{xs.min()},{xs.max()}]"
        )


def selfcheck(payload) -> bool:
    """Boot vs restore within one build. True when the restore is faithful."""
    print("\n=== selfcheck: episode 1 (boot) vs episode 2 (restore), same build ===")
    ok = _describe(payload["ram_ep1"], payload["ram_ep2"])
    if ok:
        print("\nOK — the cached startup state reproduces a real boot bit-exactly.")
    else:
        print(
            "\nFAIL — reset() does NOT reproduce a real boot. Every episode after\n"
            "the first trains against a state the game never actually reaches, so\n"
            "policies trained here are tied to this bug (see H-AM in\n"
            "experiments/003-yeti-training.md)."
        )
    return ok


def compare(a_payload, b_payload, label_a: str, label_b: str) -> None:
    print(f"\n=== compare: A={label_a}  B={label_b} ===")
    print("\nepisode 1 — REAL BOOT (tests emulator dynamics):")
    boot_same = _describe(a_payload["ram_ep1"], b_payload["ram_ep1"])
    print("\nepisode 2 — STARTUP-STATE RESTORE (tests the save/load path):")
    restore_same = _describe(a_payload["ram_ep2"], b_payload["ram_ep2"])

    if "frames_ep2" in a_payload and "frames_ep2" in b_payload:
        print("\nepisode 2 — rendered frames:")
        _describe_frames(a_payload["frames_ep2"], b_payload["frames_ep2"])

    print("\nverdict:")
    if boot_same and restore_same:
        print("  the two builds are equivalent on both paths.")
    elif boot_same and not restore_same:
        print(
            "  cold-boot dynamics are IDENTICAL, only the RESTORE differs.\n"
            "  => physics did not change; the state-restore semantics did. Run\n"
            "     selfcheck on each build to see WHICH one restores faithfully."
        )
    elif not boot_same:
        print("  cold-boot dynamics differ => a genuine emulator behaviour change.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See experiments/003-yeti/core_provenance_2b0a45d.md",
    )
    sub = p.add_subparsers(dest="mode", required=True)

    for name in ("capture", "selfcheck"):
        sp = sub.add_parser(name)
        sp.add_argument("--out", required=True, help="output .npz")
        sp.add_argument("--steps", type=int, default=150)
        sp.add_argument("--profile", default="yeti_fruit")
        sp.add_argument(
            "--frames",
            action="store_true",
            help="also record every raw frame (enables the pixel report)",
        )

    sp = sub.add_parser("compare")
    sp.add_argument("--a", required=True)
    sp.add_argument("--b", required=True)

    args = p.parse_args()

    if args.mode == "compare":
        compare(np.load(args.a), np.load(args.b), args.a, args.b)
        return 0

    payload = capture(args.out, args.steps, args.profile, args.frames)
    if args.mode == "selfcheck":
        return 0 if selfcheck(payload) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
