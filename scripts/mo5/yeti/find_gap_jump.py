#!/usr/bin/env python3
"""Find a macro that clears the first gap on level 2.

Random per-frame actions can't produce a precise run-up + jump, so we
sweep a small macro space from the level-2 spawn save-state:
  run-up: K frames of (direction)
  jump:   J frames of (direction + fire)
  settle: let it land
A failed jump falls into the gap -> death -> respawn at spawn (0,30),
lives drops. A successful one lands on the next platform: x advances and
lives is unchanged. We report final (x,y,lives) + max-x per macro.
"""
from __future__ import annotations
import argparse
from go_explore import make_env

X, Y, LIVES = 11090, 11089, 11095


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="output/mo5/yeti/level2/level2_start.sav")
    ap.add_argument("--settle", type=int, default=40)
    args = ap.parse_args()

    env = make_env("yeti_fruit")
    iface = env._interface

    def rb(a):
        return iface.read_ram_byte(a)

    with open(args.state, "rb") as f:
        spawn = f.read()
    env.reset()
    env.load_state(spawn)
    for _ in range(5):
        env.step([0, 0, 0])
    spawn = env.save_state()  # settled spawn
    x0, y0, l0 = rb(X), rb(Y), rb(LIVES)
    print(f"spawn: x={x0} y={y0} lives={l0}\n")

    results = []
    for dname, d in [("R", 1), ("L", 2)]:
        for K in (20, 40, 60, 80, 100):
            for J in (6, 10, 14, 18, 22):
                env.load_state(spawn)
                maxx = x0
                for _ in range(K):
                    env.step([0, d, 0])
                    maxx = max(maxx, rb(X))
                for _ in range(J):
                    env.step([0, d, 1])
                    maxx = max(maxx, rb(X))
                for _ in range(args.settle):
                    env.step([0, 0, 0])
                    maxx = max(maxx, rb(X))
                xf, yf, lf = rb(X), rb(Y), rb(LIVES)
                alive = lf >= l0
                advanced = abs(xf - x0)
                results.append((advanced, alive, dname, K, J, xf, yf, lf, maxx))

    # rank: alive first, then by how far x advanced
    results.sort(key=lambda r: (r[1], r[0]), reverse=True)
    print(
        f"{'dir':>3} {'K':>4} {'J':>3} {'xf':>4} {'yf':>4} {'lives':>5} "
        f"{'maxx':>5} {'alive':>5} {'dx':>4}"
    )
    for adv, alive, dn, K, J, xf, yf, lf, mx in results[:20]:
        print(
            f"{dn:>3} {K:>4} {J:>3} {xf:>4} {yf:>4} {lf:>5} {mx:>5} "
            f"{str(alive):>5} {adv:>4}"
        )
    best = results[0]
    print(
        f"\nbest: dir={best[2]} K={best[3]} J={best[4]} -> "
        f"x {x0}->{best[5]}, y {y0}->{best[6]}, lives {best[7]}, maxx {best[8]}"
    )


if __name__ == "__main__":
    main()
