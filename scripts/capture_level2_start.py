#!/usr/bin/env python3
"""Capture a save-state at the START OF LEVEL 2.

Runs the champion from reset (level 1), rides PAST the princess touch and
the victory animation, and saves the emulator state the moment level 2
loads -- detected by the bonus resetting to 1000 (the user's observation:
princess touch -> victory music -> bonus added to score -> next level,
bonus resets to 1000).

Outputs to <out>/:
  level2_start.sav   raw emulator save-state (the level-2 CP0 seed)
  level2_start.png   a frame of the level-2 layout (for fruit/nav mapping)
  transition.mp4     the run through the transition (to eyeball the layout)

Example:
  CUDA_VISIBLE_DEVICES= RETRO_AI_ROM_DIR=roms PYTHONPATH=python:build/ci-linux \\
    python scripts/capture_level2_start.py \\
      --model output/mo5/yeti/champions/v15_phase2_4500k/final_model.zip \\
      --out output/mo5/yeti/level2
"""
from __future__ import annotations

import argparse
import os

import imageio.v2 as imageio
import numpy as np
from retro_ai.training.env_builder import build_training_env
from retro_ai.training.run_config import EnvConfig
from stable_baselines3 import PPO

FRUITS_ADDR = 11055
LIVES_ADDR = 11095
PRINCESS_FLAG_ADDR = 11050
BONUS_HI = 11010
BONUS_LO = 11011
PLAYER_X = 11090
PLAYER_Y = 11089


def _pick_controllable(candidates, gym_env, base, iface, pre, probe_steps=10):
    """Pick the earliest candidate where player control is actually live.

    The level-2 intro animation is input-frozen: a state saved during it is
    "wedged" (loading it yields an uncontrollable agent that just animates and
    dies on a timer -- the original capture bug). We validate each candidate by
    loading it and checking that holding RIGHT changes the player's x. We keep
    the EARLIEST candidate that responds, which is the moment control is handed
    over -- before the agent has moved and (for a short intro) while bonus is
    still 1000.

    ``candidates`` is a list of (frame_idx, state_bytes, bonus, x, y, lives,
    raw_frame). Returns the chosen tuple, or None if none are controllable.
    """
    for cand in candidates:
        f, state, b, x, y, lv, _raw = cand
        iface.load_state(state)
        if pre is not None and hasattr(pre, "notify_state_loaded"):
            pre.notify_state_loaded()
        for _ in range(5):
            gym_env.step([0, 0, 0])
        x0 = iface.read_ram_byte(PLAYER_X)
        live = False
        for _ in range(probe_steps):
            gym_env.step([0, 1, 0])  # hold right
            if iface.read_ram_byte(PLAYER_X) != x0:
                live = True
                break
        print(
            f"   candidate +{f:3d}: bonus={b} x={x} y={y} lives={lv} "
            f"control={'LIVE' if live else 'frozen'}",
            flush=True,
        )
        if live:
            return cand
    return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--episodes", type=int, default=15)
    p.add_argument(
        "--ride-steps",
        type=int,
        default=1500,
        help="max steps to keep stepping after princess, waiting "
        "for the level-2 load (bonus->1000)",
    )
    p.add_argument(
        "--settle-window",
        type=int,
        default=300,
        help="frames to step (NOOP) after level-2 load while "
        "snapshotting candidate states",
    )
    p.add_argument(
        "--settle-stride",
        type=int,
        default=4,
        help="snapshot a candidate every N frames in the window",
    )
    args = p.parse_args()

    # Huge stall/step limits so the env doesn't truncate during the
    # (input-frozen) victory animation.
    env_cfg = EnvConfig(
        profile="yeti_fruit",
        action_mode="joystick",
        max_steps=10_000,
        stall_threshold=10**9,
        resize=(84, 84),
    )
    stack = build_training_env("yeti_fruit", env_cfg)
    base, gym_env, iface = stack.base, stack.gym, stack.base._interface
    pre = getattr(stack, "preprocessed", None)
    model = PPO.load(args.model, device="auto")
    os.makedirs(args.out, exist_ok=True)

    def bonus():
        return (iface.read_ram_byte(BONUS_HI) << 8) | iface.read_ram_byte(BONUS_LO)

    def act(obs):
        a, _ = model.predict(np.transpose(obs, (2, 0, 1)), deterministic=False)
        return a

    for ep in range(args.episodes):
        obs, _ = gym_env.reset()
        prev_pr = iface.read_ram_byte(PRINCESS_FLAG_ADDR)
        prev_lives = iface.read_ram_byte(LIVES_ADDR)
        frames = []
        touched = False

        # Phase 1: play to the princess.
        for _ in range(2000):
            obs, _, done, trunc, _ = gym_env.step(act(obs))
            raw = base._last_raw_obs
            if raw is not None:
                frames.append(np.asarray(raw, dtype=np.uint8))
            pr = iface.read_ram_byte(PRINCESS_FLAG_ADDR)
            lives = iface.read_ram_byte(LIVES_ADDR)
            if pr == 1 and prev_pr == 0:
                touched = True
                print(f"ep{ep}: princess touched (bonus={bonus()})", flush=True)
                break
            prev_pr = pr
            if lives < prev_lives and prev_lives > 0:
                break
            prev_lives = lives
            if done or trunc:
                break
        if not touched:
            print(f"ep{ep}: no princess this episode, retrying", flush=True)
            continue

        # Phase 2: ride through the victory animation into level 2.
        # Ignore done/trunc here (the emulator keeps running); stop when
        # the bonus resets to 1000 (level 2 loaded).
        prev_b = bonus()
        captured = False
        for _ in range(args.ride_steps):
            try:
                obs, _, done, trunc, _ = gym_env.step(act(obs))
            except Exception as e:
                print(
                    f"  step raised after princess ({e}); env likely "
                    f"auto-terminates on princess -- need base stepping.",
                    flush=True,
                )
                break
            raw = base._last_raw_obs
            if raw is not None:
                frames.append(np.asarray(raw, dtype=np.uint8))
            b = bonus()
            if b == 1000 and prev_b != 1000:
                # Level 2 just loaded. We are now inside the input-frozen
                # level-intro animation -- saving here is the ORIGINAL BUG
                # (the state is wedged: on reload the agent only animates and
                # dies on a timer, control never activates). Instead, step
                # NOOPs through the intro and snapshot a sequence of candidate
                # states; afterwards validate each (load -> hold right -> did x
                # move?) and keep the EARLIEST controllable one. NOOP stepping
                # guarantees the agent has not moved yet.
                candidates = []
                for f in range(args.settle_window):
                    obs, _, _, _, _ = gym_env.step([0, 0, 0])
                    raw = base._last_raw_obs
                    raw_arr = (
                        np.asarray(raw, dtype=np.uint8) if raw is not None else None
                    )
                    if raw_arr is not None:
                        frames.append(raw_arr)
                    if f % args.settle_stride == 0:
                        candidates.append(
                            (
                                f,
                                base._interface.save_state(),
                                bonus(),
                                iface.read_ram_byte(PLAYER_X),
                                iface.read_ram_byte(PLAYER_Y),
                                iface.read_ram_byte(LIVES_ADDR),
                                raw_arr,
                            )
                        )
                print(
                    f"ep{ep}: level 2 loaded; validating "
                    f"{len(candidates)} candidates for live control...",
                    flush=True,
                )
                chosen = _pick_controllable(candidates, gym_env, base, iface, pre)
                if chosen is None:
                    print(
                        f"ep{ep}: no controllable candidate in "
                        f"{args.settle_window}-frame window; retrying",
                        flush=True,
                    )
                    break
                cf, cstate, cb, cx, cy, clv, craw = chosen
                with open(os.path.join(args.out, "level2_start.sav"), "wb") as fh:
                    fh.write(cstate)
                if craw is not None:
                    imageio.imwrite(os.path.join(args.out, "level2_start.png"), craw)
                imageio.mimsave(
                    os.path.join(args.out, "transition.mp4"), frames, fps=50
                )
                print(
                    f"ep{ep}: LEVEL 2 captured. chosen=+{cf} bonus={cb} "
                    f"x={cx} y={cy} lives={clv} ({len(cstate)} bytes) "
                    f"-> {args.out}",
                    flush=True,
                )
                captured = True
                break
            prev_b = b
        if captured:
            return
        print(
            f"ep{ep}: bonus never reset to 1000 within ride window "
            f"(last bonus={bonus()}); retrying",
            flush=True,
        )

    print("Failed to capture level-2 start within the episode budget.")


if __name__ == "__main__":
    main()
