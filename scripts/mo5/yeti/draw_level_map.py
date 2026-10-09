#!/usr/bin/env python3
"""Render a level's AUTHORED LevelMap over its real frame, for verification.

The point is not decoration. ``LEVEL4`` is ~200 lines of hand-transcribed
coordinates, and a wrong number does not crash — it quietly puts a reward target
off the route and shows up weeks later as an unexplained ceiling. Drawing the
authored data (not the tilemap it came from) on top of the actual frame makes a
human able to spot that in seconds.

Shows:
  platforms   walkable extents, at the floor's surface, labelled with floor id
  ladders     vertical spans between the floors they claim to connect
  jump edges  launch -> landing, dashed, between the WAYPOINT positions
  waypoints   MANDATORY reward targets filled red, optional ones hollow yellow
  fruit       magenta, princess cyan
  graph nodes violet diamonds ON the surface line, with the graph's own edges

WHY THE GRAPH NODES ARE DRAWN SEPARATELY, AND WHY THEY WERE THE MISSING HALF.
The reward does not measure distance to a waypoint. It measures distance to a
NAV-GRAPH NODE, and the two are placed by different rules: a node goes at the
declared platform EDGE nearest the other platform (``_jump_graph``), while a
waypoint can carry a MEASURED anchor from ``jump_waypoint_pos``. Where a level
supplies such an anchor the two separate, and nothing on this map showed it.

L4's rope 2 is the case that matters: ``Low2_launch`` was moved to the measured
px 188 in September, but ``J12_13_a`` -- the point the potential actually aims at
-- still sits on the declared edge at px 184, the pixel that kills 12/12 at rest.
Four pixels, invisible unless both are drawn. Same on the landing side:
``J12_13_b`` and the mandatory ``Low2`` both sit at px 128 while floor 13 is
only standable to 124.

So: violet diamonds are what the REWARD sees, coloured circles are what the
CURRICULUM sees, and any place they do not coincide is worth explaining.

Coordinate conventions. The Y byte is the TOP row of the agent's sprite, and an
anchor's y is the Y byte of an agent standing on it. Measured on live frames (L1 and
L4 reset, Y 182): the sprite fills rows y .. y+17 and columns centre-7 .. centre+6,
centre = x_ram*4 + 8. The floor tiles start at y + 18.

So a waypoint is drawn as what the sprite reach test checks: a dot on the anchor
PIXEL (px, y), plus the outline of an agent standing on it. The outline's bottom
edge is the floor surface, where ladders end and graph nodes sit. Drawing the
anchor at any other row (this script used y + 8 until 2026-10-09) puts it beside
the ladder end it belongs to, or outside the sprite that reaches it.

Usage::

    PYTHONPATH=python:build/ci-linux RETRO_AI_ROM_DIR=roms \\
      python3 scripts/mo5/yeti/draw_level_map.py --level 4 \\
        --frame output/mo5/yeti/level4/level4_start.png \\
        --out output/mo5/yeti/level4/level4_authored_map.png
"""
from __future__ import annotations

import argparse

from PIL import Image, ImageDraw, ImageFont
from retro_ai.games import yeti
from retro_ai.training.targets import SPRITE_H, SPRITE_W, build_targets
from retro_ai.training.yeti_map import build_navigation_map, get_level_map

FONT = "/usr/share/fonts/dejavu-sans-mono-fonts/DejaVuSansMono-Bold.ttf"
SURFACE_DY = SPRITE_H  # standing_y -> visible floor surface (first tile row)
# The reach test's sprite span around the centre pixel: centre-7 .. centre+6.
SPRITE_LEFT = SPRITE_W // 2
SPRITE_RIGHT = SPRITE_W // 2 - 1

PLATFORM = (70, 230, 120)
LADDER = (90, 170, 255)
JUMP = (255, 165, 40)
MAND = (255, 40, 40)
OPT = (250, 220, 70)
FRUIT = (255, 80, 220)
PRINCESS = (80, 240, 240)
GNODE = (195, 165, 255)  # nav-graph node: what the REWARD measures distance to
GEDGE = (145, 115, 240)  # the graph's own jump edge, node -> node


def _font(size):
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:
        return None


def _dashed(d, p0, p1, colour, width=2, dash=9, gap=6):
    (x0, y0), (x1, y1) = p0, p1
    dx, dy = x1 - x0, y1 - y0
    n = max(abs(dx), abs(dy))
    if n == 0:
        return
    steps = int(n / (dash + gap)) + 1
    for i in range(steps):
        a = i * (dash + gap) / n
        b = min(a + dash / n, 1.0)
        d.line(
            [(x0 + dx * a, y0 + dy * a), (x0 + dx * b, y0 + dy * b)],
            fill=colour,
            width=width,
        )


def draw(level, frame_path, out_path, scale=5, nodes=True):
    lvl = get_level_map(level)
    wps = yeti.waypoints(level)
    targets = {t.id: t for t in build_targets(level)}
    mand_wp = {t.id for t in targets.values() if t.mandatory and t.kind == "waypoint"}

    img = Image.open(frame_path).convert("RGB")
    img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    pad_t, pad_l = 34, 4
    canvas = Image.new(
        "RGB", (img.width + pad_l * 2, img.height + pad_t + 4), (10, 10, 10)
    )
    canvas.paste(img, (pad_l, pad_t))
    d = ImageDraw.Draw(canvas, "RGBA")
    f_small, f_mid = _font(max(11, 3 * scale)), _font(max(13, 4 * scale))

    def P(x, y):
        return (pad_l + x * scale, pad_t + y * scale)

    # dim the frame so overlays read
    d.rectangle(
        [pad_l, pad_t, pad_l + img.width, pad_t + img.height], fill=(0, 0, 0, 90)
    )

    # platforms (L1 and L2 define none: their maps are ladders only)
    for p in lvl.platforms or ():
        y = p.y + SURFACE_DY
        d.line([P(p.x_min, y), P(p.x_max, y)], fill=PLATFORM, width=max(2, scale // 2))
        d.text(P(p.x_min + 1, y - 9), f"f{p.floor}", fill=PLATFORM, font=f_small)

    # ladders
    for name, top, bot, x in lvl.ladders:
        y_t = lvl.floor_top_y[top] + SURFACE_DY
        y_b = lvl.floor_top_y[bot] + SURFACE_DY
        d.line([P(x, y_t), P(x, y_b)], fill=LADDER, width=max(2, scale // 2))
        d.text(P(x + 2, (y_t + y_b) / 2 - 4), name, fill=LADDER, font=f_small)

    # jump edges, launch -> landing, using the authored waypoint positions
    names = lvl.jump_waypoint_names or {}
    for a, b in lvl.jump_edges or ():
        land = max(a, b)
        nm = names.get(land)
        if nm and nm in wps and f"{nm}_launch" in wps:
            (lx, ly, _), (ax, ay, _) = wps[f"{nm}_launch"], wps[nm]
            p0 = P(lx * 4 + 8 + 0.5, ly + 0.5)
            p1 = P(ax * 4 + 8 + 0.5, ay + 0.5)
        else:
            pf = {p.floor: p for p in lvl.platforms}
            pa, pb = pf[a], pf[b]
            p0 = P((pa.x_min + pa.x_max) / 2, pa.y + 0.5)
            p1 = P((pb.x_min + pb.x_max) / 2, pb.y + 0.5)
        _dashed(d, p0, p1, JUMP, width=max(2, scale // 2))

    # waypoints: a dot on the anchor pixel, and the sprite of an agent standing on it
    r = max(4, int(scale * 1.5))
    rd = max(3, int(scale * 0.9))
    for wid, (x_ram, y, _floor) in sorted(wps.items()):
        px = x_ram * 4 + 8
        is_m = wid in mand_wp
        colour = MAND if is_m else OPT
        d.rectangle(
            [P(px - SPRITE_LEFT, y), P(px + SPRITE_RIGHT + 1, y + SPRITE_H)],
            outline=colour + (150,),
            width=1,
        )
        cx, cy = P(px + 0.5, y + 0.5)  # the centre of the anchor pixel
        box = [cx - rd, cy - rd, cx + rd, cy + rd]
        if is_m:
            d.ellipse(box, fill=colour + (230,), outline=(0, 0, 0), width=1)
        else:
            d.ellipse(box, fill=(0, 0, 0), outline=colour, width=max(2, scale // 3))
        label = wid.replace("_launch", "^")
        d.text(
            (cx + r + 2, cy - r - 1),
            label,
            fill=colour,
            font=f_small,
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )

    # nav-graph nodes, drawn ON the surface line so they never sit on top of a
    # waypoint marker (which is drawn at the sprite centre). A 4 px offset between
    # a node and its waypoint is then 4*scale px apart and actually visible.
    n_nodes = 0
    if nodes:
        nav = build_navigation_map(level)
        by_ident = {n.ident: n for n in nav.nodes}
        # The graph's OWN edges, between node positions. Compare against the orange
        # dashed edge above, which is drawn between the waypoint positions.
        for a, b in lvl.jump_edges or ():
            na, nb = by_ident.get(f"J{a}_{b}_a"), by_ident.get(f"J{a}_{b}_b")
            if na and nb:
                _dashed(
                    d,
                    P(na.x, lvl.floor_top_y[na.floor] + SURFACE_DY),
                    P(nb.x, lvl.floor_top_y[nb.floor] + SURFACE_DY),
                    GEDGE,
                    width=max(1, scale // 3),
                    dash=4,
                    gap=5,
                )
        rn = max(3, int(scale * 0.9))
        for n in sorted(nav.nodes, key=lambda n: (n.floor, n.x)):
            cx, cy = P(n.x, lvl.floor_top_y[n.floor] + SURFACE_DY)
            d.polygon(
                [(cx, cy - rn), (cx + rn, cy), (cx, cy + rn), (cx - rn, cy)],
                fill=GNODE,
                outline=(0, 0, 0),
            )
            n_nodes += 1
            # Label the JUMP nodes only. Ladder/fruit/princess nodes coincide with a
            # marker that is already labelled, so naming them again just adds clutter;
            # the jump nodes are the ones with no other label on the map, and they are
            # the ones whose placement rule differs from the waypoint's.
            if n.kind == "jump":
                d.text(
                    (cx + rn + 1, cy + 1),
                    n.ident,
                    fill=GNODE,
                    font=f_small,
                    stroke_width=2,
                    stroke_fill=(0, 0, 0),
                )

    # fruit + princess: same convention as a waypoint. Their y is the standing Y of
    # their floor (test_fruit_and_princess_stand_on_their_floor), so they get the
    # standing-agent outline and a SQUARE on the point. The game detects these touches
    # itself (presence byte, princess flag); nothing reads this y except this drawing.
    def _stand_marker(x_px, y, colour, label):
        d.rectangle(
            [P(x_px - SPRITE_LEFT, y), P(x_px + SPRITE_RIGHT + 1, y + SPRITE_H)],
            outline=colour + (170,),
            width=1,
        )
        cx, cy = P(x_px + 0.5, y + 0.5)
        d.rectangle([cx - rd, cy - rd, cx + rd, cy + rd], fill=colour)
        d.text(
            (cx + rd + 2, cy - r),
            label,
            fill=colour,
            font=f_mid,
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )

    for fid, (fx, fy) in lvl.fruit_centre_px.items():
        _stand_marker(fx, fy, FRUIT, f"F{fid}")
    px, py = lvl.princess_centre_px
    _stand_marker(px, py, PRINCESS, "PRINCESS")

    legend = [
        ("platform (floor)", PLATFORM),
        ("ladder", LADDER),
        ("jump/rope/spring edge", JUMP),
        ("mandatory target", MAND),
        ("optional waypoint (^ = launch)", OPT),
        ("fruit", FRUIT),
        ("princess", PRINCESS),
        ("nav-graph node (what the REWARD aims at)", GNODE),
    ]
    x = pad_l + 4
    for text, colour in legend:
        d.rectangle([x, 12, x + 12, 24], fill=colour)
        d.text((x + 16, 11), text, fill=(230, 230, 230), font=f_small)
        x += 18 + int(len(text) * (3 * scale) * 0.62)
    canvas.save(out_path)
    return out_path, len(wps), len(mand_wp), n_nodes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--frame", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=int, default=5)
    ap.add_argument(
        "--no-nodes",
        action="store_true",
        help="omit the nav-graph nodes (restores the pre-2026-09-24 drawing)",
    )
    args = ap.parse_args()
    path, n, m, g = draw(
        args.level, args.frame, args.out, args.scale, nodes=not args.no_nodes
    )
    print(f"wrote {path}  ({n} waypoints, {m} mandatory, {g} graph nodes)")


if __name__ == "__main__":
    main()
