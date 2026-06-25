#!/usr/bin/env python3
"""Extract fruit + ladder + floor locations for a level by reading RAM.

The screen tilemap is a 40x25 grid of 1-byte tile-ids in user RAM, row-major,
base 0x2C27 (calibrated on level-1's known fruit tiles):

    tile(col, row) = RAM[0x2C27 + row*40 + col]

Tile-id semantics (confirmed by the user against level 1 & 2):
    1,2,3,4        ladder      (a,b,c,d; c/d = ladder-through-floor join)
    5,6,7,8        floor       (body / alt-body / left-end / right-end)
    >=10           sprites     (fruits; each fruit = a 2x2 block of 4 ids)

Coordinate conversions:
    pixel_x = col*8,  pixel_y = row*8         (8x8 tiles)
    agent X (RAM 11090) = pixel_x / 4 = col*2 (player X is in 4px units)
    agent Y (RAM 11089) = pixel_y       = row*8

Outputs floors, ladders (with the floor rows they connect), and fruits.
Reads a RAM snapshot (.npy) or a live save-state.
"""
from __future__ import annotations
import argparse
import json
import numpy as np

BASE = 0x2C27
W, H = 40, 25
LADDER_IDS = {1, 2, 3, 4}
FLOOR_IDS = {5, 6, 7, 8}


def load_ram(args) -> np.ndarray:
    if args.npy:
        return np.load(args.npy)
    from go_explore import make_env

    env = make_env("yeti_fruit")
    env.reset()
    if args.state:
        env.load_state(open(args.state, "rb").read())
    for _ in range(args.settle):
        env.step([0, 0, 0])
    return np.frombuffer(bytes(env._interface.read_ram()), dtype=np.uint8).copy()


def grid_from_ram(ram, base=BASE):
    return np.array(
        [[int(ram[base + r * W + c]) for c in range(W)] for r in range(H)], dtype=int
    )


def find_floors(grid):
    """A floor row = a row with a long run (>=4) of FLOOR_IDS tiles."""
    floors = []
    for r in range(H):
        cols = [c for c in range(W) if grid[r, c] in FLOOR_IDS]
        if len(cols) >= 4:
            floors.append(
                {"row": r, "col_min": min(cols), "col_max": max(cols), "n": len(cols)}
            )
    return floors


def find_ladders(grid):
    """Ladders = vertically-connected components of LADDER_IDS tiles.

    A ladder is 2 cols wide; we group by contiguous (col, row). Report the
    left col, the row span, and the pixel/agent x.
    """
    visited = set()
    ladders = []
    for c in range(W):
        for r in range(H):
            if grid[r, c] in LADDER_IDS and (c, r) not in visited:
                # flood fill 4-connected over ladder tiles
                stack = [(c, r)]
                comp = []
                while stack:
                    cc, rr = stack.pop()
                    if (cc, rr) in visited:
                        continue
                    if not (0 <= cc < W and 0 <= rr < H):
                        continue
                    if grid[rr, cc] not in LADDER_IDS:
                        continue
                    visited.add((cc, rr))
                    comp.append((cc, rr))
                    stack += [(cc + 1, rr), (cc - 1, rr), (cc, rr + 1), (cc, rr - 1)]
                cols = sorted({cc for cc, _ in comp})
                rows = sorted({rr for _, rr in comp})
                ladders.append(
                    {
                        "col_left": cols[0],
                        "col_right": cols[-1],
                        "row_top": rows[0],
                        "row_bot": rows[-1],
                        "n_tiles": len(comp),
                    }
                )
    ladders.sort(key=lambda d: (d["col_left"], d["row_top"]))
    return ladders


def find_fruits(grid):
    """Fruits = 2x2 blocks of sprite tiles (id >= 10). Group adjacent
    sprite tiles into connected components and report each as a fruit."""
    visited = set()
    fruits = []
    for r in range(H):
        for c in range(W):
            v = grid[r, c]
            if v >= 10 and (c, r) not in visited:
                stack = [(c, r)]
                comp = []
                while stack:
                    cc, rr = stack.pop()
                    if (cc, rr) in visited:
                        continue
                    if not (0 <= cc < W and 0 <= rr < H):
                        continue
                    if grid[rr, cc] < 10:
                        continue
                    visited.add((cc, rr))
                    comp.append((cc, rr))
                    stack += [
                        (cc + 1, rr),
                        (cc - 1, rr),
                        (cc, rr + 1),
                        (cc, rr - 1),
                        (cc + 1, rr + 1),
                        (cc - 1, rr - 1),
                        (cc + 1, rr - 1),
                        (cc - 1, rr + 1),
                    ]
                cols = [cc for cc, _ in comp]
                rows = [rr for _, rr in comp]
                ids = sorted({grid[rr, cc] for cc, rr in comp})
                fruits.append(
                    {
                        "col": min(cols),
                        "row": min(rows),
                        "col_max": max(cols),
                        "row_max": max(rows),
                        "tile_ids": ids,
                        "n_tiles": len(comp),
                    }
                )
    fruits.sort(key=lambda d: (d["row"], d["col"]))
    return fruits


def agent_xy(col, row):
    return {"pixel": (col * 8, row * 8), "agent_x": col * 2, "agent_y": row * 8}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy", default=None)
    ap.add_argument("--state", default=None)
    ap.add_argument("--settle", type=int, default=6)
    ap.add_argument("--base", type=lambda s: int(s, 0), default=BASE)
    ap.add_argument("--json", default=None, help="write structured map to this path")
    args = ap.parse_args()

    ram = load_ram(args)
    grid = grid_from_ram(ram, args.base)

    floors = find_floors(grid)
    ladders = find_ladders(grid)
    fruits = find_fruits(grid)

    # Princess: entity bytes (not in the tilemap). Y=0x2B00, X=0x2B01 (4px units).
    p_y, p_x = int(ram[0x2B00]), int(ram[0x2B01])
    princess = {
        "agent_x": p_x,
        "agent_y": p_y,
        "pixel": (p_x * 4, p_y),
        "y_byte": p_y,
        "x_byte": p_x,
    }

    print(f"=== FLOORS ({len(floors)}) ===  (row -> pixel_y, agent_y)")
    for f in floors:
        print(
            f"  row {f['row']:2d} (y_px={f['row']*8:3d}, agent_y={f['row']*8:3d})"
            f"  cols {f['col_min']}..{f['col_max']}  ({f['n']} tiles)"
        )

    print(f"\n=== LADDERS ({len(ladders)}) ===")
    for L in ladders:
        cx_px = L["col_left"] * 8
        print(
            f"  cols {L['col_left']}-{L['col_right']:2d}  rows {L['row_top']:2d}.."
            f"{L['row_bot']:2d}  x_px={cx_px:3d}  agent_x={L['col_left']*2:2d}  "
            f"({L['n_tiles']} tiles)"
        )

    print(f"\n=== FRUITS ({len(fruits)}) ===")
    for fr in fruits:
        a = agent_xy(fr["col"], fr["row"])
        presence = args.base + fr["row"] * W + fr["col"]
        print(
            f"  tile(col={fr['col']},row={fr['row']})  ids={fr['tile_ids']}  "
            f"px={a['pixel']}  agent=(x={a['agent_x']},y={a['agent_y']})  "
            f"presence_addr=0x{presence:04X} ({presence})  ({fr['n_tiles']} tiles)"
        )

    print(f"\n=== PRINCESS ===")
    print(
        f"  px={princess['pixel']}  agent=(x={princess['agent_x']},"
        f"y={princess['agent_y']})  [Y@0x2B00={princess['y_byte']}, "
        f"X@0x2B01={princess['x_byte']}]"
    )

    if args.json:
        out = {
            "base": args.base,
            "grid": [W, H],
            "floors": floors,
            "ladders": ladders,
            "fruits": [
                {
                    **fr,
                    **agent_xy(fr["col"], fr["row"]),
                    "presence_addr": args.base + fr["row"] * W + fr["col"],
                }
                for fr in fruits
            ],
            "princess": princess,
        }
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2, default=int)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
