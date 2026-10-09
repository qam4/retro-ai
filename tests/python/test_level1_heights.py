"""L1's standing heights must put each floor's surface on that floor's tile row.

L1 was the only level whose `floor_top_y` disagreed with where the agent stands: 2 px
too low on every floor (184 vs a measured 182 on the ground, and so on up). Its markers
then drew inside the floor tiles. The check uses the extracted tilemap
(experiments/003-yeti/data/level_maps/level1_map.json), not a list of numbers, so it
would also catch a future wrong edit in the other direction.

No emulator needed.
"""

from __future__ import annotations

import json
import pathlib

from retro_ai.training.yeti_map import get_level_map

MAP = (
    pathlib.Path(__file__).resolve().parents[2]
    / "experiments/003-yeti/data/level_maps/level1_map.json"
)
SURFACE_DY = 18  # standing y -> the sprite's feet, i.e. the floor surface
SCREEN_H = 200  # the ground floor has no tiles; its surface is the screen bottom


def test_level1_heights_sit_on_the_tilemap():
    lvl = get_level_map(1)
    tile_tops = {
        f["row"] * 8 for f in json.loads(MAP.read_text(encoding="utf-8"))["floors"]
    }
    for floor, y in lvl.floor_top_y.items():
        surface = y + SURFACE_DY
        if floor == 1:
            assert (
                surface == SCREEN_H
            ), f"ground surface {surface}, not the screen bottom"
        else:
            assert surface in tile_tops, (
                f"floor {floor}: standing y {y} puts the surface at {surface}, "
                f"which is not a floor tile row {sorted(tile_tops)}"
            )


def test_level1_floors_are_one_floor_height_apart():
    ys = [y for _, y in sorted(get_level_map(1).floor_top_y.items())]
    assert all(a - b == 32 for a, b in zip(ys, ys[1:])), ys
