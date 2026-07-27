#!/usr/bin/env python3
"""Map level 2 by primitive-driven Go-Explore from the captured seed.

Per-frame random actions can't traverse level 2 (precise jumps; falling
kills). Instead we explore over short ACTION PRIMITIVES (walk/jump/climb,
each held a sampled number of frames) with save-state teleport: walk along
a platform (each position saved as a cell), and try jumps FROM each saved
cell. Whatever takeoff condition crosses a gap, Go-Explore finds it by
trying jumps from every position. Death (lives drop / respawn to spawn)
ends a branch; we teleport to a saved frontier cell instead.

On every score jump (fruit eaten) we diff full RAM to find the byte that
zeroed -> fruit address, and log the (x,y) -> fruit coordinate.

Outputs to <out>/: trajectory.csv, fruit_events.csv, cells.csv, summary.txt
"""
from __future__ import annotations
import argparse, csv, os, random
from collections import Counter
import numpy as np
from go_explore import make_env, read_state, is_dead

X, Y, SCORE_HI, SCORE_LO, LIVES = 11090, 11089, 11093, 11094, 11095

# primitive: (name, [vert,horiz,fire], (min_frames, max_frames))
PRIMS = [
    ("walkR", [0, 1, 0], (6, 40)),
    ("walkL", [0, 2, 0], (6, 40)),
    ("jumpR", [0, 1, 1], (6, 16)),
    ("jumpL", [0, 2, 1], (6, 16)),
    ("jumpU", [1, 0, 1], (6, 16)),
    ("climbU", [1, 0, 0], (6, 30)),
    ("climbD", [2, 0, 0], (6, 30)),
]
SPAWN_XY = (0, 30)
LAND_FRAMES = 18  # let the agent land after a jump before judging


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="output/mo5/yeti/level2/level2_start.sav")
    ap.add_argument("--out", default="output/mo5/yeti/level2/map")
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--bucket", type=int, default=3)
    args = ap.parse_args()

    env = make_env("yeti_fruit")
    iface = env._interface
    os.makedirs(args.out, exist_ok=True)

    def rb(a):
        return iface.read_ram_byte(a)

    def pos():
        return rb(X), rb(Y)

    def score():
        return (rb(SCORE_HI) << 8) | rb(SCORE_LO)

    B = args.bucket

    def cell(x, y):
        return (y // B, x // B)

    env.reset()
    env.load_state(open(args.state, "rb").read())
    for _ in range(5):
        env.step([0, 0, 0])
    root = env.save_state()
    x0, y0 = pos()
    archive = {cell(x0, y0): {"state": root, "visits": 0, "xy": (x0, y0)}}

    traj = open(os.path.join(args.out, "trajectory.csv"), "w", newline="")
    tw = csv.writer(traj)
    tw.writerow(["it", "x", "y", "score"])
    fr = open(os.path.join(args.out, "fruit_events.csv"), "w", newline="")
    fw = csv.writer(fr)
    fw.writerow(["addr", "x", "y", "score_delta"])
    fruit_addrs: Counter = Counter()

    for it in range(args.iters):
        cells = list(archive.items())
        w = [1.0 / (1 + e["visits"]) for _, e in cells]
        key = random.choices(cells, weights=w, k=1)[0][0]
        archive[key]["visits"] += 1
        env.load_state(archive[key]["state"])
        st = read_state(env)
        base_lives = st["lives"]
        prev_bonus = st["bonus"]
        stall = 0
        name, act, (fmin, fmax) = random.choice(PRIMS)
        nf = random.randint(fmin, fmax)
        seq = [act] * nf + [[0, 0, 0]] * LAND_FRAMES  # act then settle/land
        died = False
        ram_start = np.frombuffer(bytes(iface.read_ram()), dtype=np.uint8)
        prev_score = st["score"]
        for a in seq:
            env.step(a)
            st = read_state(env)
            x, y, s1 = st["x_pos"], st["y_pos"], st["score"]
            tw.writerow([it, x, y, s1])
            if s1 > prev_score:
                after = np.frombuffer(bytes(iface.read_ram()), dtype=np.uint8)
                n = min(len(ram_start), len(after))
                for addr in np.where((ram_start[:n] != 0) & (after[:n] == 0))[0]:
                    fw.writerow([int(addr), x, y, s1 - prev_score])
                    fruit_addrs[int(addr)] += 1
                prev_score = s1
            # Level-2 bonus ticks slowly (~1 per 10 frames), so is_dead's
            # stall check false-fires. Primitives are fixed-length, so we
            # only need the reliable signal: a life lost (falling kills).
            if st["lives"] < base_lives:
                died = True
                break
            c = cell(x, y)
            if c not in archive:
                archive[c] = {"state": env.save_state(), "visits": 0, "xy": (x, y)}
        if it % 500 == 0:
            xs = [e["xy"][0] for e in archive.values()]
            ys = [e["xy"][1] for e in archive.values()]
            print(
                f"it {it}/{args.iters} cells {len(archive)} "
                f"x[{min(xs)},{max(xs)}] y[{min(ys)},{max(ys)}] "
                f"fruit-events {sum(fruit_addrs.values())}",
                flush=True,
            )

    traj.close()
    fr.close()
    with open(os.path.join(args.out, "cells.csv"), "w", newline="") as cf:
        cw = csv.writer(cf)
        cw.writerow(["x", "y", "visits"])
        for e in archive.values():
            cw.writerow([e["xy"][0], e["xy"][1], e["visits"]])
    xs = [e["xy"][0] for e in archive.values()]
    ys = [e["xy"][1] for e in archive.values()]
    with open(os.path.join(args.out, "summary.txt"), "w") as s:
        s.write(f"iters={args.iters} cells={len(archive)}\n")
        s.write(f"x range {min(xs)}..{max(xs)}  y range {min(ys)}..{max(ys)}\n")
        s.write("candidate fruit addrs (byte zeroed on score jump):\n")
        for addr, c in fruit_addrs.most_common(20):
            s.write(f"  {addr} (0x{addr:04X}): {c}\n")
    print(
        "DONE cells",
        len(archive),
        "x",
        min(xs),
        max(xs),
        "y",
        min(ys),
        max(ys),
        "fruit-addrs",
        fruit_addrs.most_common(6),
        flush=True,
    )


if __name__ == "__main__":
    main()
