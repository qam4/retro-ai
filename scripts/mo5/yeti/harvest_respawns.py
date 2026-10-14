#!/usr/bin/env python3
"""Harvest respawn states (real game restarts after a death) as a root pool.

WHY. A level start always plays the same snowball sequence (route choices come from a
generator the start fixes; inputs do not change them), and a policy trained only there
memorises it: l1_v17 scores 215/300 from L1's reset and 0/330 from respawns. Each death
leads to a respawn whose sequence depends on how the life ended, so respawns are the
game's own supply of other sequences. See experiments/003-yeti/snowball_rng_notes.md.

PLAY (run several in parallel, one --part each)::

    python scripts/mo5/yeti/harvest_respawns.py play --policy l1_v17 --game-level 1 \\
        --episodes 50 --seed 1000 --part H1 --out-dir output/mo5/yeti/respawns/l1_r1

Per episode, from the level start: the policy plays (stochastic) until a random step K
in [30, 450], then holds one random input for 12 steps, then no input until it dies
(episodes that reach the princess or never die are skipped). No input through the
death; at the first step the death flag is clear again the emulator state is saved.
Then no input for up to 330 steps to record the respawn's snowballs: inputs do not
change them, so this is the sequence's signature. Settings come from the registry.

MERGE (once, after all parts)::

    python scripts/mo5/yeti/harvest_respawns.py merge \\
        --out-dir output/mo5/yeti/respawns/l1_r1 \\
        --manifest experiments/003-yeti/data/respawns/l1_r1.json

Groups the respawns by snowball sequence (retro_ai.games.yeti_hazards) and holds out
every --heldout-every-th sequence, whole, by first appearance. The manifest lists every
state with its sha256 and is tracked; the states are local (a save holds the game).
"""

from __future__ import annotations

import argparse
import datetime
import glob
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]


def _rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(REPO))
    except ValueError:
        return str(p)


def play(args) -> None:
    import torch
    from retro_ai.games import yeti
    from retro_ai.games.yeti_hazards import hazard_row
    from retro_ai.training.env_builder import build_training_env
    from retro_ai.training.yeti_map import agent_floor_from_pixel_y
    from retro_ai.training.yeti_registry import Registry, sha256_of
    from stable_baselines3 import PPO

    reg = Registry.load()
    s = reg.eval_settings(args.policy, args.game_level)
    actual = sha256_of(REPO / s.model)
    if actual != s.model_sha256:
        raise SystemExit(f"{s.model}: sha256 {actual} is not the registered model")
    out = Path(args.out_dir)
    for sub in ("states", "sig", "parts"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    part_file = out / "parts" / f"{args.part}.json"
    if part_file.exists():
        raise SystemExit(f"{part_file} exists; use another --part")
    stack = build_training_env(s.env.profile, s.env)
    iface = stack.base._interface
    model = PPO.load(str(REPO / s.model), device="cpu")
    start = (REPO / s.start_state).read_bytes() if s.start_state else None
    rng = random.Random(args.seed)
    recs = []
    for i in range(args.episodes):
        # Seeded per episode AFTER PPO.load, which re-seeds with the model's own seed.
        random.seed(args.seed + i)
        np.random.seed(args.seed + i)
        torch.manual_seed(args.seed + i)
        k_step = rng.randint(30, 450)
        held = [rng.randrange(3), rng.randrange(3), rng.randrange(2)]
        obs, _ = stack.gym.reset()
        if start is not None:
            iface.load_state(start)
            stack.preprocessed.notify_state_loaded()
            obs, *_ = stack.gym.step([0, 0, 0])
        prev_pr = iface.read_ram_byte(yeti.PRINCESS_FLAG_ADDR)
        t_d = None
        for t in range(s.env.max_steps):
            if t < k_step:
                a, _ = model.predict(np.transpose(obs, (2, 0, 1)), deterministic=False)
            elif t < k_step + 12:
                a = held
            else:
                a = [0, 0, 0]
            obs, *_ = stack.gym.step(a)
            pr = iface.read_ram_byte(yeti.PRINCESS_FLAG_ADDR)
            if pr == 1 and prev_pr == 0:
                break
            prev_pr = pr
            if yeti.is_dead(iface):
                t_d = t
                break
        if t_d is None:
            print(
                f"{args.part} {i}: no death (princess or timeout), skipped", flush=True
            )
            continue
        x, y = yeti.read_pos(iface)
        floor = agent_floor_from_pixel_y(y, s.layout)
        fruits = s.fruits_total - iface.read_ram_byte(yeti.FRUITS_ADDR)
        waited = 0
        while yeti.is_dead(iface) and waited < 700:
            stack.gym.step([0, 0, 0])
            waited += 1
        if yeti.is_dead(iface):
            print(f"{args.part} {i}: no respawn within 700 steps, skipped", flush=True)
            continue
        state = bytes(iface.save_state())
        sp = out / "states" / f"{args.part}_{i}.sav"
        sp.write_bytes(state)
        at = dict(
            lives=iface.read_ram_byte(yeti.LIVES_ADDR),
            bonus=yeti.read_bonus(iface),
            fruits_left=iface.read_ram_byte(yeti.FRUITS_ADDR),
        )
        rows = [hazard_row(iface)]
        bonus = [at["bonus"]]
        while len(rows) < 330:
            stack.gym.step([0, 0, 0])
            rows.append(hazard_row(iface))
            bonus.append(yeti.read_bonus(iface))
            if yeti.is_dead(iface):
                break
        sig = out / "sig" / f"{args.part}_{i}.npz"
        np.savez_compressed(sig, rows=np.stack(rows))
        recs.append(
            dict(
                path=_rel(sp),
                sha256=hashlib.sha256(state).hexdigest(),
                signature=_rel(sig),
                death_step=t_d,
                death_floor=floor,
                death_px=x * 4 + 8,
                fruits_collected=fruits,
                respawn_after=waited,
                at_respawn=at,
                bonus_ticks=len(set(bonus[:21])) > 1,
            )
        )
        print(
            f"{args.part} {i}: died step {t_d} floor {floor}, {fruits} fruit(s); "
            f"respawn after {waited} steps, {at}",
            flush=True,
        )
    part_file.write_text(
        json.dumps(
            dict(
                part=args.part,
                policy=args.policy,
                game_level=args.game_level,
                layout=s.layout,
                seed=args.seed,
                episodes=args.episodes,
                states=recs,
            ),
            indent=1,
        ),
        encoding="utf-8",
    )
    print("DONE")


def merge(args) -> None:
    from retro_ai.games.yeti_hazards import choices, group_sequences

    out = Path(args.out_dir)
    parts = [
        json.loads(Path(f).read_text())
        for f in sorted(glob.glob(str(out / "parts/*.json")))
    ]
    if not parts:
        raise SystemExit(f"no parts under {out / 'parts'}")
    levels = {(p["game_level"], p["layout"], p["policy"]) for p in parts}
    if len(levels) != 1:
        raise SystemExit(f"parts disagree on level / layout / policy: {levels}")
    ((game_level, layout, policy),) = levels
    states = [st for p in parts for st in p["states"]]
    seqs = [choices(np.load(REPO / st["signature"])["rows"], layout) for st in states]
    gid = group_sequences(seqs)
    n_seq = max(gid) + 1
    held = {g for g in range(n_seq) if g % args.heldout_every == args.heldout_every - 1}
    for st, g, sq in zip(states, gid, seqs):
        st["sequence"] = g
        st["heldout"] = g in held
        st["choices_seen"] = sum(len(c) for c in sq)
    manifest = dict(
        what=(
            "Respawn states (game restarts after a death), grouped by snowball "
            "sequence. "
            "Made by scripts/mo5/yeti/harvest_respawns.py. States are local files; "
            "sha256 pins each one."
        ),
        created=datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
        game_level=game_level,
        layout=layout,
        policy=policy,
        parts=[{k: p[k] for k in ("part", "seed", "episodes")} for p in parts],
        heldout_every=args.heldout_every,
        n_sequences=n_seq,
        heldout_sequences=sorted(held),
        states=states,
    )
    mp = Path(args.manifest)
    mp.parent.mkdir(parents=True, exist_ok=True)
    if mp.exists() and not args.overwrite:
        raise SystemExit(f"{mp} exists (pass --overwrite to replace)")
    mp.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    sizes = sorted((gid.count(g) for g in range(n_seq)), reverse=True)
    n_held = sum(st["heldout"] for st in states)
    print(
        f"{len(states)} respawns, {n_seq} snowball sequences (sizes {sizes}); held out "
        f"{len(held)} sequences / {n_held} states; bonus ticks after respawn in "
        f"{sum(st['bonus_ticks'] for st in states)}/{len(states)}; wrote {mp}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("play")
    a.add_argument("--policy", required=True)
    a.add_argument("--game-level", type=int, required=True)
    a.add_argument("--episodes", type=int, default=50)
    a.add_argument("--seed", type=int, required=True)
    a.add_argument("--part", required=True)
    a.add_argument("--out-dir", required=True)
    m = sub.add_parser("merge")
    m.add_argument("--out-dir", required=True)
    m.add_argument("--manifest", required=True)
    m.add_argument("--heldout-every", type=int, default=3)
    m.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    os.chdir(REPO)
    sys.path.insert(0, str(REPO / "python"))
    play(args) if args.cmd == "play" else merge(args)


if __name__ == "__main__":
    main()
