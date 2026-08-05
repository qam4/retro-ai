#!/usr/bin/env python3
"""Diagnostic: SEE what actually happens at the escalator (no success gate).

Two parts, both printed as raw numbers so we can eyeball the mechanics instead
of guessing:

  A. MOVING-ENTITY SCAN. From a goat seed, hold NOOP for N frames and report
     every RAM byte in the entity region that CHANGES over time -- this exposes
     the escalator platforms' x/y addresses, their positions and their speed
     (the agent is standing still, so anything moving is the escalator).

  B. JUMP TRACES. For a few (delay, walk, jhold) jump-right scripts, dump the
     agent's (x, y, pose, dead) every frame so we can see whether it ever
     touches a platform, whether pose flickers while riding, and where it dies.
"""
from __future__ import annotations

import pickle

from retro_ai.games import yeti
from retro_ai.training.env_builder import build_training_env
from retro_ai.training.run_config import EnvConfig

X, Y, POSE, DEATH = 11090, 11089, 11092, 11004
SURF = set(yeti.SURFACE_POSES)
ENT_LO, ENT_HI = 11008, 11264  # 0x2B00 .. 0x2C00 (entity table region)


def _agent(ifc):
    return (
        ifc.read_ram_byte(X),
        ifc.read_ram_byte(Y),
        ifc.read_ram_byte(POSE),
        int(ifc.read_ram_byte(DEATH) == 65),
    )


def main():
    cfg = EnvConfig(
        profile="yeti_fruit_level3",
        action_mode="joystick",
        max_steps=400,
        stall_threshold=400,
        resize=(84, 84),
    )
    stack = build_training_env("yeti_fruit_level3", cfg)
    base, gym_env, ifc = stack.base, stack.gym, stack.base._interface
    base.reset(seed=0)
    d = pickle.load(
        open("output/mo5/yeti/training/yeti_curriculum_l3_v4_15m/checkpoints.pkl", "rb")
    )
    goat = [s[2] for k in ("Lgoat_a_top", "Lgoat_b_top") for s in d["waypoints"][k][0]]
    print(f"{len(goat)} goat seeds", flush=True)
    RIGHT, RJUMP, NOOP = [0, 1, 0], [0, 1, 1], [0, 0, 0]

    # ---- Part A: moving-entity scan (agent stands still) --------------------
    sb = goat[0]
    ifc.load_state(sb)
    stack.preprocessed.notify_state_loaded()
    for _ in range(5):
        gym_env.step(NOOP)
    print(
        f"\n[A] agent at rest: x{ifc.read_ram_byte(X)} y{ifc.read_ram_byte(Y)} "
        f"pose{ifc.read_ram_byte(POSE)}",
        flush=True,
    )
    N = 40
    series = {a: [] for a in range(ENT_LO, ENT_HI)}
    for _ in range(N):
        gym_env.step(NOOP)
        for a in range(ENT_LO, ENT_HI):
            series[a].append(ifc.read_ram_byte(a))
    print(
        f"[A] scanning RAM {ENT_LO}-{ENT_HI - 1} over {N} NOOP frames; "
        "changing bytes (addr: first..last [distinct count]):",
        flush=True,
    )
    for a in range(ENT_LO, ENT_HI):
        vals = series[a]
        if len(set(vals)) > 1:
            head = ",".join(str(v) for v in vals[:12])
            print(
                f"  {a} (0x{a:04X}): {head}... "
                f"min={min(vals)} max={max(vals)} distinct={len(set(vals))}",
                flush=True,
            )

    # ---- Part B: jump traces WITH platform entity positions -----------------
    # For each frame print the agent (x,y in RAM units) alongside all 6 entity
    # slots (base 0x2B60, stride 8): (+0, +1) = (y, x) of each platform. Agent x
    # is RAM (px/4); entity coords printed raw so we can see the mapping. This
    # shows whether the agent and a descending platform ever COINCIDE at the
    # boarding point -- the thing every "board" attempt has failed to do.
    ENT_BASE, ENT_STRIDE, ENT_N = 11104, 8, 6

    def ents():
        out = []
        for i in range(ENT_N):
            b = ENT_BASE + i * ENT_STRIDE
            out.append((ifc.read_ram_byte(b), ifc.read_ram_byte(b + 1)))
        return out

    combos = [
        (2, 3, 4, 4),
        (2, 3, 4, 6),
        (4, 3, 4, 4),
        (4, 3, 4, 6),
        (6, 3, 4, 5),
        (0, 3, 4, 6),
    ]
    for delay, walk, jhold, drift in combos:
        ifc.load_state(sb)
        stack.preprocessed.notify_state_loaded()
        for _ in range(5):
            gym_env.step(NOOP)
        seq = (
            [NOOP] * delay
            + [RIGHT] * walk
            + [RJUMP] * jhold
            + [RIGHT] * drift
            + [NOOP] * 45
        )
        print(
            f"\n[B] delay={delay} walk={walk} jhold={jhold} drift={drift} "
            "(ent slots: y,x)",
            flush=True,
        )
        for i, act in enumerate(seq):
            gym_env.step(act)
            x, y, p, dead = _agent(ifc)
            g = "G" if p in SURF else "."
            es = " ".join(f"[{ey},{ex}]" for ey, ex in ents())
            print(f"  t{i:02d} ax={x} ay={y} pose={p}{g} d={dead} | {es}", flush=True)
            if dead:
                break


if __name__ == "__main__":
    main()
