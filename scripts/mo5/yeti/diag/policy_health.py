#!/usr/bin/env python3
"""Representation health of a policy across a run's snapshots.

Why this exists. Every L4 run peaks mid-run and degrades: v16c's best snapshot
is 9.3M of 15M, v18's is 900k of 15M. Two very different causes produce that
same curve and the training scalars cannot tell them apart:

  * the policy never commits -- entropy stays near uniform, so the "peak" is a
    lucky draw from a near-random policy and there is nothing to preserve;
  * the network loses the ability to represent improvements -- feature rank
    collapses and units go dormant (plasticity loss / capacity loss), after
    which no amount of further training or reverting helps.

The first is fixed by an entropy/LR schedule. The second needs normalisation
layers or a churn-reduction loss. Picking the wrong one wastes a 7h run, so
measure before choosing.

All three metrics are computed on ONE fixed batch of observations so the
numbers are comparable across snapshots:

  srank99   singular values of the N x D feature matrix needed to reach 99% of
            their total (Kumar et al.'s stable rank). Falling srank = the
            representation is collapsing onto fewer directions.
  erank     exp(entropy of the normalised singular spectrum). Same idea,
            smooth, less threshold-sensitive.
  dormant   fraction of feature units whose mean |activation| over the batch is
            under 1% of the layer mean. Rising dormant = dead capacity.
  H         mean policy entropy on the batch, in nats and as a percentage of
            the action space's maximum. Low = committed, high = near-random.

Level-agnostic: it reads the observation space off the env and the action space
off the model, so it works for any run this repo produces.

Example::

    env PYTHONPATH=python:build/ci-linux RETRO_AI_ROM_DIR=roms python3 \\
      scripts/mo5/yeti/diag/policy_health.py \\
        --run output/mo5/yeti/training/yeti_curriculum_l4_v16c_payonchange_cold_15m \\
        --level 4 --fruits-total 1 \\
        --start-state output/mo5/yeti/level4/level4_start.sav \\
        --profile yeti_fruit_level4 --stall-threshold 40 \\
        --steps 100000,1000000,3000000,6000000,9300000,12000000,15000000
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import List

import numpy as np

REPO = Path(__file__).resolve().parents[4]


def _collect_obs(stack, model, *, n_obs: int, start_state, settle: int, det: bool):
    """Roll out under ``model`` and keep every observation it saw.

    One fixed batch, gathered once, reused for every snapshot -- comparing
    ranks across snapshots each on its own on-policy states would confound the
    representation with the state distribution.
    """
    gym_env = stack.gym
    iface = stack.base._interface
    out: List[np.ndarray] = []
    while len(out) < n_obs:
        obs, _ = gym_env.reset()
        if start_state is not None:
            iface.load_state(start_state)
            stack.preprocessed.notify_state_loaded()
            for _ in range(settle):
                obs, _, _, _, _ = gym_env.step([0, 0, 0])
        for _ in range(400):
            o = np.transpose(obs, (2, 0, 1))
            out.append(o.copy())
            if len(out) >= n_obs:
                break
            action, _ = model.predict(o, deterministic=det)
            obs, _, term, trunc, _ = gym_env.step(list(action))
            if term or trunc:
                break
    return np.asarray(out[:n_obs])


def _spectrum_stats(feats: np.ndarray):
    """srank99 and erank of an ``N x D`` feature matrix."""
    x = feats - feats.mean(axis=0, keepdims=True)
    sv = np.linalg.svd(x, compute_uv=False)
    total = float(sv.sum())
    if total <= 0:
        return 0, 0.0
    csum = np.cumsum(sv) / total
    srank = int(np.searchsorted(csum, 0.99) + 1)
    p = sv / total
    p = p[p > 0]
    erank = float(np.exp(-(p * np.log(p)).sum()))
    return srank, erank


def _max_entropy(model) -> float:
    space = model.action_space
    nvec = getattr(space, "nvec", None)
    if nvec is not None:
        return float(sum(math.log(int(n)) for n in nvec))
    return float(math.log(int(space.n)))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", required=True, help="training output dir")
    p.add_argument(
        "--steps",
        default=None,
        help="comma-separated snapshot steps. Default: 7 evenly spaced.",
    )
    p.add_argument("--obs", type=int, default=400, help="observations in the batch")
    p.add_argument("--profile", default="yeti_fruit")
    p.add_argument("--level", type=int, default=1)
    p.add_argument("--fruits-total", type=int, default=4)
    p.add_argument("--start-state", default=None)
    p.add_argument("--stall-threshold", type=int, default=15)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--obs-model", default=None, help="model used to gather obs")
    p.add_argument(
        "--batch-cache",
        default=None,
        help="npy path. Loaded if present, else written after gathering. Comparing "
        "two RUNS requires the identical batch -- without this each run is scored "
        "on its own state distribution and the numbers are not comparable.",
    )
    p.add_argument("--out", default=None)
    args = p.parse_args()

    import torch
    from retro_ai.training.env_builder import build_training_env
    from retro_ai.training.run_config import EnvConfig
    from stable_baselines3 import PPO

    snap_dir = os.path.join(args.run, "snapshots")
    avail = sorted(
        int(f[len("model_") : -len("_steps.zip")])
        for f in os.listdir(snap_dir)
        if f.startswith("model_") and f.endswith("_steps.zip")
    )
    if args.steps:
        want = [int(s) for s in args.steps.split(",")]
        steps = [min(avail, key=lambda a: abs(a - w)) for w in want]
    else:
        idx = np.linspace(0, len(avail) - 1, 7).astype(int)
        steps = [avail[i] for i in idx]

    start_bytes = None
    if args.start_state:
        with open(args.start_state, "rb") as f:
            start_bytes = f.read()

    env_cfg = EnvConfig(
        profile=args.profile,
        action_mode="joystick",
        max_steps=args.max_steps,
        stall_threshold=args.stall_threshold,
        resize=(84, 84),
    )
    stack = build_training_env(args.profile, env_cfg)

    obs_model_path = args.obs_model or os.path.join(
        snap_dir, f"model_{steps[-1]}_steps.zip"
    )
    obs_model = PPO.load(obs_model_path, device="cpu")
    maxh = _max_entropy(obs_model)
    if args.batch_cache and os.path.exists(args.batch_cache):
        batch = np.load(args.batch_cache)
        print(f"loaded batch {batch.shape} from {args.batch_cache}", flush=True)
    else:
        print(f"gathering {args.obs} observations under {obs_model_path}", flush=True)
        batch = _collect_obs(
            stack,
            obs_model,
            n_obs=args.obs,
            start_state=start_bytes,
            settle=1,
            det=False,
        )
        if args.batch_cache:
            np.save(args.batch_cache, batch)
            print(f"wrote batch to {args.batch_cache}", flush=True)
    del obs_model
    print(f"batch={batch.shape}  max policy entropy={maxh:.4f} nats\n", flush=True)

    rows = []
    print(
        f"{'step':>10} {'srank99':>8} {'erank':>8} {'dormant':>8} "
        f"{'H(nats)':>8} {'H %max':>7}"
    )
    for st in steps:
        path = os.path.join(snap_dir, f"model_{st}_steps.zip")
        model = PPO.load(path, device="cpu")
        pol = model.policy
        with torch.no_grad():
            obs_t, _ = pol.obs_to_tensor(batch)
            feats = pol.extract_features(obs_t)
            if isinstance(feats, tuple):  # separate pi/vf extractors
                feats = feats[0]
            f = feats.cpu().numpy()
            ent = float(pol.get_distribution(obs_t).entropy().mean().item())
        srank, erank = _spectrum_stats(f)
        mean_act = np.abs(f).mean()
        dormant = float((np.abs(f).mean(axis=0) < 0.01 * mean_act).mean())
        rows.append(
            {
                "step": st,
                "srank99": srank,
                "erank": erank,
                "dormant": dormant,
                "entropy": ent,
                "entropy_frac": ent / maxh,
                "dim": int(f.shape[1]),
            }
        )
        print(
            f"{st:>10} {srank:>8} {erank:>8.1f} {dormant:>8.3f} "
            f"{ent:>8.3f} {100 * ent / maxh:>6.1f}%",
            flush=True,
        )
        del model

    out = args.out or os.path.join(args.run, "policy_health.json")
    with open(out, "w") as fh:
        json.dump(
            {
                "run": args.run,
                "obs": int(batch.shape[0]),
                "max_entropy": maxh,
                "feature_dim": rows[0]["dim"] if rows else None,
                "rows": rows,
            },
            fh,
            indent=1,
        )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
