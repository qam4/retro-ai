#!/usr/bin/env python3
"""Capture a save-state at the START OF LEVEL 3 (and a video of the
level-2 -> level-3 transition).

Generalizes capture_level2_start.py to the next level. The difference:
the level-2 policy does NOT reach the princess from a cold level-2 reset
(0% from reset in v7/v8 -- see experiments/003-yeti-training.md), it only
completes the final leg from a SEED. So instead of playing from reset, we
LOAD a high-value seed each attempt (a waypoint pool near the princess, or
the both-fruits CP2 pool, from a training run's checkpoints.pkl), run the
policy to the princess, then ride PAST the victory animation and save the
emulator state the moment level 3 loads -- detected the same way
(bonus resetting to 1000; user's observation: princess -> victory music ->
bonus added to score -> next level, bonus resets to 1000).

Outputs to <out>/:
  level3_start.sav   raw emulator save-state (the level-3 CP0 seed)
  level3_start.png   a frame of the level-3 layout
  transition.mp4     the run through princess + victory + level 3 (to eyeball)

Example (run=yeti_curriculum_l2_v8_anneal_10m):
  CUDA_VISIBLE_DEVICES= RETRO_AI_ROM_DIR=roms PYTHONPATH=python:build/ci-linux \\
    python scripts/mo5/yeti/capture_level3_start.py \\
      --model .../final_model.zip \\
      --seed-checkpoints .../checkpoints.pkl \\
      --seed-pool L56_bot \\
      --out output/mo5/yeti/level3
"""
from __future__ import annotations

import argparse
import os
import pickle
import random

import imageio.v2 as imageio
import numpy as np
from retro_ai.games import yeti
from retro_ai.training.env_builder import build_training_env
from retro_ai.training.run_config import EnvConfig
from stable_baselines3 import PPO

PLAYER_X = yeti.X_ADDR
PLAYER_Y = yeti.Y_ADDR
PRINCESS_FLAG_ADDR = yeti.PRINCESS_FLAG_ADDR


def _load_seed_pool(pkl_path: str, pool: str):
    """Return a list of raw state-byte blobs from a checkpoints.pkl pool.

    ``pool`` is either a waypoint id (e.g. "L56_bot") or "cpN" for the
    fruit-checkpoint pool N (e.g. "cp2" = both-fruits states on L2).
    Entries are (source_cp, bonus, state_bytes); we keep the bytes.
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    if pool.lower().startswith("cp"):
        idx = int(pool[2:])
        states = data["checkpoints"][idx]
    else:
        states = data.get("waypoints", {})[pool][0]
    blobs = [bytes(s[2]) for s in states]
    if not blobs:
        raise ValueError(f"seed pool {pool!r} is empty in {pkl_path}")
    return blobs


def _pick_controllable(candidates, gym_env, iface, pre, probe_steps=10):
    """Pick the earliest candidate where player control is actually live.

    The level-intro animation is input-frozen: a state saved during it is
    "wedged" (loading it yields an uncontrollable agent that just animates and
    dies on a timer). We validate each candidate by loading it and checking
    that holding RIGHT changes the player's x, keeping the EARLIEST responsive
    one -- the moment control is handed over. (Same logic as
    capture_level2_start._pick_controllable.)
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
    p.add_argument("--seed-checkpoints", required=True, help="checkpoints.pkl")
    p.add_argument(
        "--seed-pool",
        default="L56_bot",
        help="waypoint id (e.g. L56_bot) or cpN (e.g. cp2) to seed from",
    )
    p.add_argument("--out", required=True)
    p.add_argument("--profile", default="yeti_fruit_level2")
    p.add_argument("--attempts", type=int, default=60)
    p.add_argument("--settle", type=int, default=5, help="NOOP steps after load")
    p.add_argument(
        "--to-princess-steps",
        type=int,
        default=800,
        help="max steps per attempt to reach the princess from the seed",
    )
    p.add_argument(
        "--ride-steps",
        type=int,
        default=1500,
        help="max steps after princess, waiting for level-3 load (bonus->1000)",
    )
    p.add_argument("--settle-window", type=int, default=300)
    p.add_argument("--settle-stride", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed)

    env_cfg = EnvConfig(
        profile=args.profile,
        action_mode="joystick",
        max_steps=10_000,
        stall_threshold=10**9,
        resize=(84, 84),
    )
    stack = build_training_env(args.profile, env_cfg)
    base, gym_env, iface = stack.base, stack.gym, stack.base._interface
    pre = getattr(stack, "preprocessed", None)
    model = PPO.load(args.model, device="auto")
    os.makedirs(args.out, exist_ok=True)

    seeds = _load_seed_pool(args.seed_checkpoints, args.seed_pool)
    print(
        f"Loaded {len(seeds)} seed states from pool {args.seed_pool!r}; "
        f"profile={args.profile}",
        flush=True,
    )

    def bonus():
        return yeti.read_bonus(iface)

    def act(obs):
        a, _ = model.predict(np.transpose(obs, (2, 0, 1)), deterministic=False)
        return a

    # Boot the emulator once (expensive startup); subsequent attempts just
    # load_state a seed (the seed-pool rollout pattern in yeti_rollout).
    gym_env.reset()

    for attempt in range(args.attempts):
        state = random.choice(seeds)
        iface.load_state(state)
        if pre is not None and hasattr(pre, "notify_state_loaded"):
            pre.notify_state_loaded()
        obs = None
        for _ in range(args.settle):
            obs, _, _, _, _ = gym_env.step([0, 0, 0])

        prev_pr = iface.read_ram_byte(PRINCESS_FLAG_ADDR)
        frames = []
        touched = False

        # Phase 1: from the seed, play to the princess.
        for _ in range(args.to_princess_steps):
            obs, _, done, trunc, _ = gym_env.step(act(obs))
            raw = base._last_raw_obs
            if raw is not None:
                frames.append(np.asarray(raw, dtype=np.uint8))
            pr = iface.read_ram_byte(PRINCESS_FLAG_ADDR)
            if pr == 1 and prev_pr == 0:
                touched = True
                print(
                    f"attempt{attempt}: princess touched (bonus={bonus()})",
                    flush=True,
                )
                break
            prev_pr = pr
            if yeti.is_dead(iface):
                break
            if done or trunc:
                break
        if not touched:
            continue

        # Phase 2: ride through the victory animation into level 3. Ignore
        # done/trunc (emulator keeps running); stop when bonus resets to 1000.
        prev_b = bonus()
        captured = False
        for _ in range(args.ride_steps):
            try:
                obs, _, done, trunc, _ = gym_env.step(act(obs))
            except Exception as e:
                print(f"  step raised after princess ({e}); stopping ride.", flush=True)
                break
            raw = base._last_raw_obs
            if raw is not None:
                frames.append(np.asarray(raw, dtype=np.uint8))
            b = bonus()
            if b == 1000 and prev_b != 1000:
                # Level 3 just loaded; step NOOPs through the input-frozen
                # intro and snapshot candidates, then keep the earliest
                # controllable one (see capture_level2_start for the rationale).
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
                                yeti.read_lives(iface),
                                raw_arr,
                            )
                        )
                print(
                    f"attempt{attempt}: level 3 loaded; validating "
                    f"{len(candidates)} candidates for live control...",
                    flush=True,
                )
                chosen = _pick_controllable(candidates, gym_env, iface, pre)
                # Always save the transition video so level 3 can be eyeballed
                # even if no controllable candidate is found.
                imageio.mimsave(
                    os.path.join(args.out, "transition.mp4"), frames, fps=50
                )
                if chosen is None:
                    print(
                        f"attempt{attempt}: no controllable candidate in "
                        f"{args.settle_window}-frame window; transition.mp4 saved; "
                        f"retrying for a clean seed",
                        flush=True,
                    )
                    break
                cf, cstate, cb, cx, cy, clv, craw = chosen
                with open(os.path.join(args.out, "level3_start.sav"), "wb") as fh:
                    fh.write(cstate)
                if craw is not None:
                    imageio.imwrite(os.path.join(args.out, "level3_start.png"), craw)
                print(
                    f"attempt{attempt}: LEVEL 3 captured. chosen=+{cf} bonus={cb} "
                    f"x={cx} y={cy} lives={clv} ({len(cstate)} bytes) -> {args.out}",
                    flush=True,
                )
                captured = True
                break
            prev_b = b
        if captured:
            return
        # princess reached but no level-3 load detected; save what we saw.
        if frames:
            imageio.mimsave(
                os.path.join(args.out, f"princess_attempt{attempt}.mp4"), frames, fps=50
            )
        print(
            f"attempt{attempt}: princess reached but bonus never reset to 1000 "
            f"(last bonus={bonus()}); video saved; retrying",
            flush=True,
        )

    print("Failed to capture level-3 start within the attempt budget.")


if __name__ == "__main__":
    main()
