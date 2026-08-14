#!/usr/bin/env python3
"""Turn a level's RAM tilemap into a LABELLED map for hand-authoring a LevelMap.

Why this exists
---------------
Authoring ``yeti_map.LEVEL{N}`` needs floors, ladders, the fruit and the princess
in game coordinates. The tilemap gives the first three; the princess is a SPRITE
and never appears in it, so a human has to point at her. This script produces the
two artefacts that make that conversation precise:

  <label>_map.png   the real frame, upscaled, with an 8px tile grid, row/col
                    rulers, and every platform and ladder labelled P1.. / L1..
  a text table      the same structures with rows, columns and pixel coords,
                    so they can be pasted into a LevelMap almost directly

Tile ids, calibrated against the known LEVEL3 map:

  1,2      ladder body (ladders are 2 columns wide)
  3,4      ladder TOP rung, embedded in the platform row it lands on
  5,6      platform middle        7 / 8   platform left / right end
  18-21    the L3 fruit, a 2x2 sprite block (L4 uses 26-29)
  40-44    the L3 escalator (body / top / bottom)

Coordinate conventions (same as LevelMap):
  pixel x = col * 8              standing x_px for a ladder = ladder_ram*4 + 8
  pixel y = row * 8              floor_top_y is the standing y on that platform

Usage::

    PYTHONPATH=python:build/ci-linux:scripts/mo5/yeti RETRO_AI_ROM_DIR=roms \\
      python3 scripts/mo5/yeti/map_builder.py \\
        --state output/mo5/yeti/level4/level4_start.sav \\
        --frame output/mo5/yeti/level4/level4_start.png \\
        --label level4 --out output/mo5/yeti/level4
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

W, H, BASE = 40, 25, 0x2C27
LADDER_BODY = {1, 2}
LADDER_TOP = {3, 4}
PLATFORM = {3, 4, 5, 6, 7, 8}
# 2x2 sprite blocks seen in the tilemap: L3's fruit is 18-21, L4's is 26-29.
FRUIT_BLOCKS = {"L3-style": {18, 19, 20, 21}, "L4-style": {26, 27, 28, 29}}
ESCALATOR = {40, 41, 42, 43, 44}


def read_grid(state_path: str, settle: int = 8) -> np.ndarray:
    from go_explore import make_env

    env = make_env("yeti_fruit")
    env.reset()
    with open(state_path, "rb") as fh:
        env.load_state(fh.read())
    for _ in range(settle):
        env.step([0, 0, 0])
    ram = np.frombuffer(bytes(env._interface.read_ram()), dtype=np.uint8).copy()
    return np.array(
        [[ram[BASE + r * W + c] for c in range(W)] for r in range(H)], dtype=np.uint8
    )


def find_platforms(grid) -> list:
    """Contiguous horizontal runs of platform tiles, top row first."""
    out = []
    for r in range(H):
        c = 0
        while c < W:
            if int(grid[r, c]) in PLATFORM:
                c0 = c
                while c < W and int(grid[r, c]) in PLATFORM:
                    c += 1
                out.append({"row": r, "col0": c0, "col1": c - 1})
            else:
                c += 1
    for i, p in enumerate(out, 1):
        p["id"] = f"P{i}"
        p["y_px"] = p["row"] * 8
        p["x0_px"] = p["col0"] * 8
        p["x1_px"] = p["col1"] * 8 + 7
        p["width"] = p["col1"] - p["col0"] + 1
    return out


def find_ladders(grid) -> list:
    """Vertical runs of ladder body, paired into the 2-column ladders they are.

    A ladder's TOP is the platform row holding its 3,4 top rung (one row above
    the body); its BOTTOM is the row below the last body tile.
    """
    seen = set()
    out = []
    for c in range(W):
        for r in range(H):
            if (r, c) in seen or int(grid[r, c]) not in LADDER_BODY:
                continue
            r0 = r
            r1 = r
            while r1 + 1 < H and int(grid[r1 + 1, c]) in LADDER_BODY:
                r1 += 1
            for rr in range(r0, r1 + 1):
                seen.add((rr, c))
            out.append({"col": c, "row0": r0, "row1": r1})
    # Pair adjacent columns with the same vertical extent into one ladder.
    merged = []
    used = set()
    for i, a in enumerate(out):
        if i in used:
            continue
        mate = None
        for j, b in enumerate(out):
            if j <= i or j in used:
                continue
            if (
                b["col"] == a["col"] + 1
                and b["row0"] == a["row0"]
                and b["row1"] == a["row1"]
            ):
                mate = j
                break
        if mate is not None:
            used.add(mate)
            merged.append(
                {
                    "col0": a["col"],
                    "col1": a["col"] + 1,
                    **{k: a[k] for k in ("row0", "row1")},
                }
            )
        else:
            merged.append(
                {
                    "col0": a["col"],
                    "col1": a["col"],
                    "row0": a["row0"],
                    "row1": a["row1"],
                }
            )
        used.add(i)
    merged.sort(key=lambda d: (d["row0"], d["col0"]))
    for i, m in enumerate(merged, 1):
        m["id"] = f"L{i}"
        # top rung row: the platform row the ladder arrives on
        m["top_row"] = m["row0"] - 1
        m["bot_row"] = m["row1"] + 1
        cx = (m["col0"] + m["col1"] + 1) * 4  # centre pixel x of the 2-col pair
        m["x_px"] = cx
        m["x_ram"] = (cx - 8) // 4
    return merged


def find_sprite_blocks(grid) -> dict:
    out = {}
    for name, ids in FRUIT_BLOCKS.items():
        locs = [(r, c) for r in range(H) for c in range(W) if int(grid[r, c]) in ids]
        if locs:
            rs = [r for r, _ in locs]
            cs = [c for _, c in locs]
            out[name] = {
                "rows": (min(rs), max(rs)),
                "cols": (min(cs), max(cs)),
                "centre_px": ((min(cs) + max(cs) + 1) * 4, (min(rs) + max(rs) + 1) * 4),
            }
    esc = [(r, c) for r in range(H) for c in range(W) if int(grid[r, c]) in ESCALATOR]
    if esc:
        out["escalator"] = {
            "rows": (min(r for r, _ in esc), max(r for r, _ in esc)),
            "cols": (min(c for _, c in esc), max(c for _, c in esc)),
        }
    return out


def annotate(frame_path, out_path, platforms, ladders, blocks, scale=4):
    from PIL import Image, ImageDraw

    img = Image.open(frame_path).convert("RGB")
    img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    pad = 28
    canvas = Image.new("RGB", (img.width + pad, img.height + pad), (16, 16, 16))
    canvas.paste(img, (pad, pad))
    d = ImageDraw.Draw(canvas)
    t = 8 * scale
    # grid + rulers
    for c in range(W + 1):
        x = pad + c * t
        d.line([(x, pad), (x, pad + img.height)], fill=(60, 60, 60))
        if c < W and c % 2 == 0:
            d.text((x + 2, 6), str(c), fill=(200, 200, 90))
    for r in range(H + 1):
        y = pad + r * t
        d.line([(pad, y), (pad + img.width, y)], fill=(60, 60, 60))
        if r < H:
            d.text((4, y + 8), str(r), fill=(200, 200, 90))
    # platforms: outline the run, label at its left end
    for p in platforms:
        x0 = pad + p["col0"] * t
        x1 = pad + (p["col1"] + 1) * t
        y0 = pad + p["row"] * t
        d.rectangle([x0, y0, x1, y0 + t], outline=(80, 220, 120), width=2)
        d.text((x0 + 3, y0 + 3), p["id"], fill=(80, 255, 140))
    # ladders: outline the body, label at the top
    for m in ladders:
        x0 = pad + m["col0"] * t
        x1 = pad + (m["col1"] + 1) * t
        y0 = pad + m["row0"] * t
        y1 = pad + (m["row1"] + 1) * t
        d.rectangle([x0, y0, x1, y1], outline=(90, 170, 255), width=2)
        d.text((x0 + 3, y0 + 3), m["id"], fill=(140, 200, 255))
    for name, b in blocks.items():
        (r0, r1), (c0, c1) = b["rows"], b["cols"]
        d.rectangle(
            [pad + c0 * t, pad + r0 * t, pad + (c1 + 1) * t, pad + (r1 + 1) * t],
            outline=(255, 120, 120),
            width=3,
        )
        d.text((pad + c0 * t + 3, pad + r0 * t - 12), name, fill=(255, 150, 150))
    canvas.save(out_path)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--state", required=True)
    ap.add_argument("--frame", help="PNG of the same moment, for the overlay")
    ap.add_argument("--label", default="level")
    ap.add_argument("--out", required=True)
    ap.add_argument("--settle", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    grid = read_grid(args.state, args.settle)
    platforms = find_platforms(grid)
    ladders = find_ladders(grid)
    blocks = find_sprite_blocks(grid)

    print(f"tile ids present: {sorted({int(v) for v in grid.flatten()} - {0})}")
    print(f"\n=== PLATFORMS ({len(platforms)}) — candidate floors ===")
    print(f"{'id':<5}{'row':>4}{'cols':>10}{'width':>7}   y_px  x range px")
    for p in platforms:
        cols = "{}-{}".format(p["col0"], p["col1"])
        xr = "{}-{}".format(p["x0_px"], p["x1_px"])
        print(
            f"{p['id']:<5}{p['row']:>4}{cols:>10}{p['width']:>7}   {p['y_px']:>4}  {xr}"
        )
    print(f"\n=== LADDERS ({len(ladders)}) ===")
    print(
        f"{'id':<5}{'cols':>8}{'body rows':>12}{'top row':>9}"
        f"{'bot row':>9}{'x_px':>6}{'x_ram':>7}"
    )
    for m in ladders:
        cols = "{}-{}".format(m["col0"], m["col1"])
        body = "{}-{}".format(m["row0"], m["row1"])
        print(
            f"{m['id']:<5}{cols:>8}{body:>12}{m['top_row']:>9}"
            f"{m['bot_row']:>9}{m['x_px']:>6}{m['x_ram']:>7}"
        )
    print("\n=== SPRITE BLOCKS (fruit / escalator; princess is NOT in the tilemap) ===")
    for k, v in blocks.items():
        print(
            f"  {k:<12} rows {v['rows']} cols {v['cols']}"
            + (f" centre_px {v['centre_px']}" if "centre_px" in v else "")
        )

    data = {
        "platforms": platforms,
        "ladders": ladders,
        "blocks": blocks,
        "grid": grid.tolist(),
    }
    jpath = os.path.join(args.out, f"{args.label}_map_raw.json")
    with open(jpath, "w") as fh:
        json.dump(data, fh, indent=1)
    print(f"\nwrote {jpath}")
    if args.frame and os.path.exists(args.frame):
        png = annotate(
            args.frame,
            os.path.join(args.out, f"{args.label}_map.png"),
            platforms,
            ladders,
            blocks,
        )
        print(f"wrote {png}")


if __name__ == "__main__":
    main()
