"""Tests for the hand-coded Yeti navigation graph."""

from __future__ import annotations

import pytest
from retro_ai.training.yeti_map import (
    agent_floor_from_pixel_y,
    build_navigation_map,
)


@pytest.fixture(scope="module")
def nav():
    return build_navigation_map()


def test_fixed_nodes_present(nav):
    """All expected nodes are in the graph."""
    for fid in (1, 2, 3, 4):
        assert f"F{fid}" in nav.node_by_ident
    for ladder in ("L12a", "L12b", "L23", "L34", "L45"):
        assert f"{ladder}_bot" in nav.node_by_ident
        assert f"{ladder}_top" in nav.node_by_ident
    assert "princess" in nav.node_by_ident


def test_same_floor_distance_is_absolute_x(nav):
    """Two nodes on the same floor: cost = |dx|."""
    f1 = nav.nodes[nav.node_by_ident["F1"]]
    l12a_bot = nav.nodes[nav.node_by_ident["L12a_bot"]]
    assert f1.floor == l12a_bot.floor == 1
    d = nav.dist[nav.node_by_ident["F1"]][nav.node_by_ident["L12a_bot"]]
    assert d == abs(f1.x - l12a_bot.x) == 64


def test_adjacent_floor_via_ladder(nav):
    """Fruit 2 to fruit 1: walk to L12a_top (40 px), climb (32 px),
    walk to F1 (64 px) = 136 px."""
    d = nav.dist[nav.node_by_ident["F2"]][nav.node_by_ident["F1"]]
    assert d == 136


def test_far_apart_fruits(nav):
    """Fruit 1 (floor 1) to fruit 4 (floor 4): must traverse multiple
    ladders. Expected path cost: 392 px."""
    d = nav.dist[nav.node_by_ident["F1"]][nav.node_by_ident["F4"]]
    assert d == 392


def test_princess_from_spawn_fruit(nav):
    """F1 -> princess should be 464 px (all the way up and across)."""
    d = nav.dist[nav.node_by_ident["F1"]][nav.node_by_ident["princess"]]
    assert d == 464


def test_distance_symmetric(nav):
    """Graph edges are bidirectional; distances symmetric."""
    a = nav.node_by_ident["F1"]
    b = nav.node_by_ident["F4"]
    assert nav.dist[a][b] == nav.dist[b][a]


def test_path_distance_from_agent_at_spawn(nav):
    """Agent at (floor=1, x=0) to F1 (x=184) is a 184px walk."""
    d = nav.path_distance_from_agent(1, 0, "F1")
    assert d == 184


def test_path_distance_uses_better_ladder(nav):
    """Agent on floor 1 at x=280 (right by L12b). To F2 (x=80 on floor
    2). Should go up L12b, then walk left: 0 + 32 + 200 = 232."""
    d = nav.path_distance_from_agent(1, 280, "F2")
    assert d == 232


def test_path_distance_agent_at_target(nav):
    """Agent at the target fruit gives distance 0."""
    d = nav.path_distance_from_agent(1, 184, "F1")
    assert d == 0


def test_agent_floor_from_pixel_y_standing():
    """Standing y (within tolerance) resolves to correct floor."""
    assert agent_floor_from_pixel_y(184) == 1
    assert agent_floor_from_pixel_y(152) == 2
    assert agent_floor_from_pixel_y(120) == 3
    assert agent_floor_from_pixel_y(88) == 4
    assert agent_floor_from_pixel_y(56) == 5


def test_agent_floor_from_pixel_y_tolerance():
    """8px tolerance around each floor top."""
    assert agent_floor_from_pixel_y(180) == 1
    assert agent_floor_from_pixel_y(192) == 1


def test_agent_floor_from_pixel_y_mid_air_returns_none():
    """A y clearly between floors resolves to None."""
    # Between floor 1 (y=184) and floor 2 (y=152): midpoint 168 is
    # outside the 8px tolerance of either, so None.
    assert agent_floor_from_pixel_y(168) is None
    # Death animation region.
    assert agent_floor_from_pixel_y(16) is None
