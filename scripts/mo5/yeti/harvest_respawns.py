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

ENTRIES (levels 2-5; run beside PLAY parts into the same --out-dir)::

    python scripts/mo5/yeti/harvest_respawns.py entries --policy l1_v22 \\
        --game-level 2 --episodes 60 --seed 2000 --part E1 --out-dir <dir>

A level after L1 has no single start: which hazards it opens with depends on when the
previous level was finished (measured 2026-10-20: twelve L1 finishes 373-440 steps gave
11 different L2 goat patterns; L2 -> L3 gave 11 of 12, L3 -> L4 and L4 -> L5 3 of 11).
So the level's start pool also holds ENTRIES: --policy (the PREVIOUS level's champion)
plays the previous level from its start to the princess, recording its actions; the
playthrough is replayed to the last step the agent stands on a surface at least --back
steps before the touch, the agent waits a random
0..--max-wait steps without input, then the policy finishes. It keeps acting through
the victory animation until the level loads, then waits without input until the
previous level's cleared flag is clear and the bonus clock ticks -- the moment play
begins, and the moment the tracked L4 and L5 starts were captured at. That state is
saved and recorded like a respawn. Waiting is legal play; no RAM is written.

MERGE (once, after all parts)::

    python scripts/mo5/yeti/harvest_respawns.py merge \\
        --out-dir output/mo5/yeti/respawns/l1_r1 \\
        --manifest experiments/003-yeti/data/respawns/l1_r1.json

Groups the states by hazard sequence (retro_ai.games.yeti_hazards: ``choices`` on L1's
layout, ``route_choices`` elsewhere) and holds out every --heldout-every-th sequence,
whole, by first appearance. The manifest lists every state with its sha256 and is
tracked; the states are local (a save holds the game).
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
        rec, at = _record(stack, out, f"{args.part}_{i}")
        rec.update(
            kind="respawn",
            death_step=t_d,
            death_floor=floor,
            death_px=x * 4 + 8,
            fruits_collected=fruits,
            respawn_after=waited,
            at_respawn=at,
        )
        recs.append(rec)
        print(
            f"{args.part} {i}: died step {t_d} floor {floor}, {fruits} fruit(s); "
            f"respawn after {waited} steps, {at}",
            flush=True,
        )
    _write_part(part_file, args, args.game_level, s.layout, "respawn", recs)


def _record(stack, out: Path, name: str):
    """Save the emulator state now, and its hazard signature: 330 game steps without
    input (inputs do not change hazards), ending early if the standing agent dies.
    Returns the manifest record and the lives / bonus / fruits it starts with."""
    from retro_ai.games import yeti
    from retro_ai.games.yeti_hazards import hazard_row

    iface = stack.base._interface
    state = bytes(iface.save_state())
    sp = out / "states" / f"{name}.sav"
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
    sig = out / "sig" / f"{name}.npz"
    np.savez_compressed(sig, rows=np.stack(rows))
    rec = dict(
        path=_rel(sp),
        sha256=hashlib.sha256(state).hexdigest(),
        signature=_rel(sig),
        bonus_ticks=len(set(bonus[:21])) > 1,
    )
    return rec, at


def _write_part(part_file: Path, args, game_level, layout, kind, recs) -> None:
    part_file.write_text(
        json.dumps(
            dict(
                part=args.part,
                kind=kind,
                policy=args.policy,
                game_level=game_level,
                layout=layout,
                seed=args.seed,
                episodes=args.episodes,
                states=recs,
            ),
            indent=1,
        ),
        encoding="utf-8",
    )
    print("DONE")


def entries(args) -> None:
    import torch
    from retro_ai.games import yeti
    from retro_ai.training.env_builder import build_training_env
    from retro_ai.training.yeti_registry import Registry, sha256_of
    from stable_baselines3 import PPO

    if args.game_level < 2:
        raise SystemExit("entries are for levels 2 and up; L1 starts from a reset")
    reg = Registry.load()
    target = reg.level(args.game_level)
    s = reg.eval_settings(args.policy, args.game_level - 1)  # plays the level before
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
    gym_env = stack.gym
    model = PPO.load(str(REPO / s.model), device="cpu")
    start = (REPO / s.start_state).read_bytes() if s.start_state else None
    rng = random.Random(args.seed)

    def step(a):
        try:
            return gym_env.step([int(v) for v in np.ravel(a)])[0]
        except Exception:  # the env may refuse after its own done; the game runs on
            return None

    def act(obs):
        if obs is None:
            return [0, 0, 0]
        return model.predict(np.transpose(obs, (2, 0, 1)), deterministic=False)[0]

    def begin():
        obs, _ = gym_env.reset()
        if start is not None:
            iface.load_state(start)
            stack.preprocessed.notify_state_loaded()
            obs = step([0, 0, 0])
        return obs

    def touched(prev):
        pr = iface.read_ram_byte(yeti.PRINCESS_FLAG_ADDR)
        return pr == 1 and prev == 0, pr

    recs = []
    for i in range(args.episodes):
        random.seed(args.seed + i)
        np.random.seed(args.seed + i)
        torch.manual_seed(args.seed + i)
        wait = rng.randint(0, args.max_wait)
        # 1. Play the previous level to the princess, recording the actions.
        obs, actions, grounded, touch = begin(), [], [], None
        prev = iface.read_ram_byte(yeti.PRINCESS_FLAG_ADDR)
        for t in range(s.env.max_steps):
            a = [int(v) for v in np.ravel(act(obs))]
            actions.append(a)
            obs = step(a)
            grounded.append(yeti.read_pose(iface) in yeti.SURFACE_POSES)
            hit, prev = touched(prev)
            if hit:
                touch = t
                break
            if yeti.is_dead(iface) or obs is None:
                break
        if touch is None:
            print(f"{args.part} {i}: no princess, skipped", flush=True)
            continue
        # 2. Replay it to the last GROUNDED step at least --back steps before the
        # touch, wait, then let the policy finish. Close to the princess, because
        # waiting on first reaching her surroundings got L2's champion killed by a
        # goat every time; grounded, because an agent mid-jump lands on her whatever
        # the input, so the wait would not move the finish.
        obs = begin()
        last = len(actions) - args.back
        cut = max([i + 1 for i in range(last) if grounded[i]] or [0])
        for a in actions[:cut]:
            obs = step(a)
        prev = iface.read_ram_byte(yeti.PRINCESS_FLAG_ADDR)
        waited, hit, died = 0, False, False
        while waited < wait and not hit:
            obs = step([0, 0, 0])
            waited += 1
            hit, prev = touched(prev)
            died = yeti.is_dead(iface)
            if died:
                break
        t = cut + waited
        while not hit and not died and t < cut + waited + 200:
            obs = step(act(obs))
            t += 1
            hit, prev = touched(prev)
            died = yeti.is_dead(iface) or obs is None
        if not hit:
            print(
                f"{args.part} {i}: {'died' if died else 'no touch'} after waiting "
                f"{waited}, skipped",
                flush=True,
            )
            continue
        touch = t
        score = (iface.read_ram_byte(yeti.SCORE_HI) << 8) | iface.read_ram_byte(
            yeti.SCORE_LO
        )
        prev_b, loaded, ride = yeti.read_bonus(iface), False, 0
        while ride < 2000:
            nobs = step(act(obs))
            obs = nobs if nobs is not None else obs
            ride += 1
            b = yeti.read_bonus(iface)
            if b == 1000 and prev_b != 1000:
                loaded = True
                break
            prev_b = b
        if not loaded:
            print(f"{args.part} {i}: level {args.game_level} never loaded", flush=True)
            continue
        hold = 0
        while hold < 600 and not (
            iface.read_ram_byte(yeti.PRINCESS_FLAG_ADDR) == 0
            and yeti.read_bonus(iface) < 1000
        ):
            step([0, 0, 0])
            hold += 1
        # CONTROL CHECK: an intro state that ignores input is useless as a start
        # (capture_level2_start.py, H-AE). Hold right, then left, for up to 10 steps.
        state = bytes(iface.save_state())
        live = False
        for move in ([0, 1, 0], [0, 2, 0]):
            iface.load_state(state)
            stack.preprocessed.notify_state_loaded()
            x0 = yeti.read_pos(iface)[0]
            for _ in range(10):
                step(move)
                if yeti.read_pos(iface)[0] != x0:
                    live = True
                    break
            if live:
                break
        iface.load_state(state)
        stack.preprocessed.notify_state_loaded()
        if not live:
            print(f"{args.part} {i}: no control at the entry, skipped", flush=True)
            continue
        rec, at = _record(stack, out, f"{args.part}_{i}")
        rec.update(
            kind="entry",
            from_level=args.game_level - 1,
            wait=waited,
            max_wait=args.max_wait,
            back=args.back,
            touch_step=touch,
            score_at_touch=score,
            ride_steps=ride,
            hold_steps=hold,
            at_entry=at,
        )
        recs.append(rec)
        print(
            f"{args.part} {i}: waited {waited}, princess at step {touch}, "
            f"score {score}; L{args.game_level} entry {at}",
            flush=True,
        )
    _write_part(part_file, args, args.game_level, target.layout, "entry", recs)


def merge(args) -> None:
    from retro_ai.games.yeti_hazards import choices, group_sequences, route_choices

    out = Path(args.out_dir)
    parts = [
        json.loads(Path(f).read_text())
        for f in sorted(glob.glob(str(out / "parts/*.json")))
    ]
    if not parts:
        raise SystemExit(f"no parts under {out / 'parts'}")
    # One level per manifest. The policies may differ: entries are made by the PREVIOUS
    # level's champion, respawns by this level's.
    levels = {(p["game_level"], p["layout"]) for p in parts}
    if len(levels) != 1:
        raise SystemExit(f"parts disagree on level / layout: {levels}")
    ((game_level, layout),) = levels
    policies = sorted({p["policy"] for p in parts})
    states = [st for p in parts for st in p["states"]]
    for p in parts:
        for st in p["states"]:
            st.setdefault("kind", p.get("kind", "respawn"))
            st.setdefault("policy", p["policy"])

    def _seq(rows):
        # L1's layout has a floor model (choices); the others do not.
        return choices(rows, layout) if layout == 1 else route_choices(rows)

    seqs = [_seq(np.load(REPO / st["signature"])["rows"]) for st in states]
    gid = group_sequences(seqs)
    n_seq = max(gid) + 1
    held = {g for g in range(n_seq) if g % args.heldout_every == args.heldout_every - 1}
    for st, g, sq in zip(states, gid, seqs):
        st["sequence"] = g
        st["heldout"] = g in held
        st["choices_seen"] = sum(len(c) for c in sq)
    kinds = sorted({st["kind"] for st in states})
    manifest = dict(
        what=(
            "Level-start states -- "
            + " and ".join(
                {
                    "respawn": "respawns (game restarts after a death)",
                    "entry": "entries (the level as it opens after the previous one, "
                    "finished at different moments)",
                }[k]
                for k in kinds
            )
            + " -- grouped by hazard sequence. Made by "
            "scripts/mo5/yeti/harvest_respawns.py. States are local files; sha256 "
            "pins each one."
        ),
        created=datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
        game_level=game_level,
        layout=layout,
        policies=policies,
        grouping="choices" if layout == 1 else "route_choices",
        parts=[
            {
                "part": p["part"],
                "kind": p.get("kind", "respawn"),
                "policy": p["policy"],
                "seed": p["seed"],
                "episodes": p["episodes"],
            }
            for p in parts
        ],
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
    by_kind = {k: sum(st["kind"] == k for st in states) for k in kinds}
    print(
        f"{len(states)} states {by_kind}, {n_seq} hazard sequences (sizes {sizes}); "
        f"held out {len(held)} sequences / {n_held} states; bonus ticks at the start "
        f"in {sum(st['bonus_ticks'] for st in states)}/{len(states)}; wrote {mp}"
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
    e = sub.add_parser("entries")
    e.add_argument("--policy", required=True, help="the PREVIOUS level's champion")
    e.add_argument("--game-level", type=int, required=True, help="the level entered")
    e.add_argument("--episodes", type=int, default=60)
    e.add_argument("--seed", type=int, required=True)
    e.add_argument("--part", required=True)
    e.add_argument("--out-dir", required=True)
    e.add_argument(
        "--max-wait",
        type=int,
        default=60,
        help="steps waited near the princess, drawn uniformly from 0..this (60 is "
        "about two of L1's 33-step snowball periods)",
    )
    e.add_argument(
        "--back",
        type=int,
        default=3,
        help="the wait happens at the last grounded step at least this many steps "
        "before the recorded princess touch",
    )
    m = sub.add_parser("merge")
    m.add_argument("--out-dir", required=True)
    m.add_argument("--manifest", required=True)
    m.add_argument("--heldout-every", type=int, default=3)
    m.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    os.chdir(REPO)
    sys.path.insert(0, str(REPO / "python"))
    {"play": play, "entries": entries, "merge": merge}[args.cmd](args)


if __name__ == "__main__":
    main()
