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


def number_surface(grid, platforms) -> list:
    """Number the tiles worth pointing at: FLOOR surface and LADDER TOPS.

    Only these get numbers, because these are the places a waypoint can live —
    the agent stands on a floor tile, and a ladder top is where a climb hands
    over to a floor. Empty cells and ladder bodies are deliberately unnumbered so
    the overlay stays readable.

    Numbering is row-major (top-left first), so numbers run left-to-right along
    each floor and increase downward. ``ladder_top`` tiles (ids 3,4) sit INSIDE a
    floor row, so they are part of the same sequence rather than a separate one.
    """
    by_pos = {}
    for p in platforms:
        for c in range(p["col0"], p["col1"] + 1):
            by_pos[(p["row"], c)] = p["id"]
    out = []
    n = 0
    for r in range(H):
        for c in range(W):
            v = int(grid[r, c])
            if v not in PLATFORM:
                continue
            n += 1
            out.append(
                {
                    "n": n,
                    "row": r,
                    "col": c,
                    "tile_id": v,
                    "kind": "ladder_top" if v in LADDER_TOP else "floor",
                    # LevelMap conventions: y_px is the standing y for this row,
                    # x_ram is what the game stores for the player's x.
                    "x_px": c * 8,
                    "y_px": r * 8,
                    "x_ram": (c * 8 - 8) // 4,
                    "platform": by_pos.get((r, c)),
                }
            )
    return out


def annotate(frame_path, out_path, grid, surface, ladders, blocks, scale=6):
    """The frame upscaled, with FLOOR and LADDER-TOP tiles numbered.

    Floors are tinted green and ladder tops yellow (they are the hand-over
    points); ladder bodies are outlined blue but unnumbered, and sprite blocks
    are outlined red. Row/col rulers stay for cross-referencing the raw grid.
    """
    from PIL import Image, ImageDraw, ImageFont

    img = Image.open(frame_path).convert("RGB")
    img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    t = 8 * scale
    pad_l, pad_t = 44, 30
    canvas = Image.new(
        "RGB", (img.width + pad_l + 4, img.height + pad_t + 4), (12, 12, 12)
    )
    canvas.paste(img, (pad_l, pad_t))
    d = ImageDraw.Draw(canvas, "RGBA")
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/dejavu-sans-mono-fonts/DejaVuSansMono-Bold.ttf", 13
        )
        ruler = ImageFont.truetype(
            "/usr/share/fonts/dejavu-sans-mono-fonts/DejaVuSansMono-Bold.ttf", 15
        )
    except OSError:
        font = ruler = None

    # ladder bodies: context only, no numbers
    for m in ladders:
        d.rectangle(
            [
                pad_l + m["col0"] * t,
                pad_t + m["row0"] * t,
                pad_l + (m["col1"] + 1) * t,
                pad_t + (m["row1"] + 1) * t,
            ],
            fill=(90, 170, 255, 60),
            outline=(90, 170, 255, 200),
            width=2,
        )
    # sprite blocks (fruit / escalator)
    for name, b in blocks.items():
        (r0, r1), (c0, c1) = b["rows"], b["cols"]
        d.rectangle(
            [
                pad_l + c0 * t,
                pad_t + r0 * t,
                pad_l + (c1 + 1) * t,
                pad_t + (r1 + 1) * t,
            ],
            outline=(255, 90, 90),
            width=3,
        )
        d.text(
            (pad_l + c0 * t, pad_t + r0 * t - 16), name, fill=(255, 140, 140), font=font
        )
    # the numbered tiles
    for s in surface:
        x0, y0 = pad_l + s["col"] * t, pad_t + s["row"] * t
        fill = (250, 210, 60, 130) if s["kind"] == "ladder_top" else (60, 230, 120, 100)
        d.rectangle([x0, y0, x0 + t, y0 + t], fill=fill)
        d.text((x0 + 3, y0 + 2), str(s["n"]), fill=(255, 255, 255), font=font)
    # grid on top so numbers stay readable
    for c in range(W + 1):
        x = pad_l + c * t
        d.line([(x, pad_t), (x, pad_t + img.height)], fill=(70, 70, 70))
    for r in range(H + 1):
        y = pad_t + r * t
        d.line([(pad_l, y), (pad_l + img.width, y)], fill=(70, 70, 70))
    # rulers
    for c in range(W):
        d.text((pad_l + c * t + 6, 8), str(c), fill=(210, 210, 90), font=ruler)
    for r in range(H):
        d.text((6, pad_t + r * t + t // 3), str(r), fill=(210, 210, 90), font=ruler)
    canvas.save(out_path)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--state", required=True)
    ap.add_argument("--frame", help="PNG of the same moment, for the overlay")
    ap.add_argument("--label", default="level")
    ap.add_argument("--out", required=True)
    ap.add_argument("--settle", type=int, default=8)
    ap.add_argument("--scale", type=int, default=6, help="overlay upscale factor")
    ap.add_argument(
        "--tile",
        type=int,
        nargs="*",
        help="translate tile numbers back to row/col/pixels and report what is "
        "there, then exit (e.g. --tile 623 784)",
    )
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    grid = read_grid(args.state, args.settle)

    platforms = find_platforms(grid)
    ladders = find_ladders(grid)
    blocks = find_sprite_blocks(grid)
    surface = number_surface(grid, platforms)

    if args.tile:
        by_n = {s["n"]: s for s in surface}
        for n in args.tile:
            s = by_n.get(n)
            if s is None:
                print(f"tile {n}: not a numbered floor/ladder-top tile")
                continue
            print(
                f"tile {s['n']:>4}  {s['kind']:<10} row {s['row']:>2} col {s['col']:>2}"
                f"  px ({s['x_px']},{s['y_px']})  x_ram {s['x_ram']:>3}"
                f"  on {s['platform']}"
            )
        return

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

    print(f"\n=== NUMBERED TILES ({len(surface)}) — floors and ladder tops ===")
    print("numbers run left-to-right along each floor, top row first")
    for p in platforms:
        ns = [s for s in surface if s["platform"] == p["id"]]
        if not ns:
            continue
        span = (
            "{}-{}".format(ns[0]["n"], ns[-1]["n"]) if len(ns) > 1 else str(ns[0]["n"])
        )
        tops = [str(s["n"]) for s in ns if s["kind"] == "ladder_top"]
        print(
            f"  {p['id']:<4} row {p['row']:>2}  y_px {p['y_px']:>3}  "
            f"cols {p['col0']:>2}-{p['col1']:<2}  tiles {span:<9}"
            + (f"  ladder-top tiles: {','.join(tops)}" if tops else "")
        )

    data = {
        "platforms": platforms,
        "ladders": ladders,
        "blocks": blocks,
        "surface": surface,
        "grid": grid.tolist(),
    }
    jpath = os.path.join(args.out, f"{args.label}_map_raw.json")
    with open(jpath, "w") as fh:
        json.dump(data, fh, indent=1)
    print(f"\nwrote {jpath}")
    print(
        "\nOnly floor and ladder-top tiles are numbered. Translate a number with:"
        "\n  map_builder.py --state <sav> --out <dir> --tile <n> [<n> ...]"
    )
    if args.frame and os.path.exists(args.frame):
        png = annotate(
            args.frame,
            os.path.join(args.out, f"{args.label}_tiles.png"),
            grid,
            surface,
            ladders,
            blocks,
            scale=args.scale,
        )
        print(f"wrote {png}")


if __name__ == "__main__":
    main()
