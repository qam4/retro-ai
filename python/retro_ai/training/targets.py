"""ONE model for every route target: fruits, waypoints, the princess.

Why this module exists
---------------------
We shipped four *parallel* implementations of the same idea — fruit checkpoints
(CP), position waypoints (WP), mandatory reward milestones, and seed pools — and
each time a behaviour was added to one and not the others it produced a real
bug, not a cosmetic inconsistency:

* ``defer_fruit_credit`` deferred credit until grounded-alive for FRUITS only.
  Milestones kept banking credit for an arrival the agent died from, so on L3
  "touch SN3 then die" paid +5.04 while waiting for a safe phase paid 0.00 — the
  policy correctly learned to arrive recklessly and never learned to survive
  there (measured: at SN3 it is no better than random).
* "already reached" state survived a seed load for FRUITS (their state lives in
  emulator RAM) but not for milestones (positional, held in the reward object),
  so seeded episodes re-targeted milestones BEHIND them and the potential paid
  them to RETREAT.
* ``reached_next`` means "collected the next fruit", which on a level with ONE
  fruit at the summit is effectively dead — so waypoint admission silently
  degenerated to "survive 30 steps" with no credit for progress.

The lesson is not "add the missing flag again"; it is that a target's TYPE
should determine only how it is DETECTED. Everything else — whether it is
mandatory, whether it can be seeded from, how credit is paid, how progress is
tracked — must be one shared code path.

The model
---------
A :class:`Target` is anything on the route that can be *reached* and pays credit
once. The five roles from experiments/003-yeti/curriculum_cp_wp_model.md become
plain data on it:

===================  ==========================================================
role                 field
===================  ==========================================================
1 graph node         ``node_ident``  (for graph path-distance shaping)
2 reward target      ``mandatory``   (summed in the PBRS potential)
3 seed source        ``seedable``    (capture states here, start episodes from)
4 progress-tracked   ``tracked``     (reach / progress EMAs, route table)
5 trigger            ``trigger`` + ``pos`` (how "reached" is decided)
===================  ==========================================================

``trigger`` is the ONLY genuinely type-specific part:

* ``"event"``  — a game event; fruits (the fruit-count / presence byte drops).
  Event state lives in emulator RAM, so it survives a save-state for free.
* ``"position"`` — grounded within a tolerance box of ``pos``; waypoints.
  Positional state does NOT survive a save-state, so it must be captured with
  the seed and restored on load.
* ``"flag"`` — a RAM flag; the princess touch (terminal).

Ordering is deliberately NOT part of this model: targets are unordered and the
potential sums over all not-yet-reached ones (settled decision #5). Levels
branch, so "the next target" is ill-defined. ``LevelMap.route_order`` exists only
to sort log output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Target:
    """One reachable route target. See the module docstring for the roles."""

    id: str
    kind: str  # "fruit" | "waypoint" | "princess" (informational)
    trigger: str  # "event" | "position" | "flag"
    # Navigation-graph node used for path-distance shaping. May differ from
    # ``id``: the curriculum calls a jump landing "A1" while the graph calls the
    # same point "J10_11_b". None => not shaped (no graph node).
    node_ident: Optional[str] = None
    # (x_ram, y_px) for positional detection; None for event/flag triggers.
    pos: Optional[Tuple[int, int]] = None
    floor: Optional[int] = None
    mandatory: bool = False  # summed in the reward potential
    seedable: bool = False  # capture + start pool
    tracked: bool = True  # reach/progress metrics

    @property
    def positional(self) -> bool:
        """True when 'reached' is decided by position — and therefore does NOT
        survive a save-state, so it must be restored with the seed."""
        return self.trigger == "position"


def build_targets(level: int = 1) -> List[Target]:
    """Derive every target for ``level`` from its :class:`LevelMap`.

    Single source of truth, so the reward, the curriculum pools and the logs can
    never disagree about what exists or what it is called.
    """
    from retro_ai.games import yeti
    from retro_ai.training.yeti_map import get_level_map, jump_waypoints

    lvl = get_level_map(level)
    out: List[Target] = []

    # --- fruits: detected by a game EVENT (count/presence byte drops) --------
    for fid in sorted(lvl.fruit_centre_px):
        x_px, y_px = lvl.fruit_centre_px[fid]
        out.append(
            Target(
                id=f"F{fid}",
                kind="fruit",
                trigger="event",
                node_ident=f"F{fid}",
                pos=((int(x_px) - 8) // 4, int(y_px)),
                floor=lvl.fruit_floor.get(fid),
                mandatory=True,  # fruits are always summed in the potential
                seedable=True,  # the CP pools
            )
        )

    # --- waypoints: detected by POSITION ------------------------------------
    # Mandatory-ness comes from reward_waypoints, which names graph nodes; a jump
    # landing has two names for one point (curriculum "A1" vs graph "J10_11_b"),
    # so resolve by POSITION to avoid the alias trap that silently disabled the
    # milestone restore.
    reward_idents = {i for g in (lvl.reward_waypoints or []) for i in g}
    jump_wps = jump_waypoints(lvl)
    pos_of_ident: Dict[str, Tuple[int, int]] = {}
    for name, (x_ram, y_px, _f) in jump_wps.items():
        pos_of_ident[name] = (x_ram, y_px)
    # graph positions for the reward idents (so we can match aliases by pos)
    from retro_ai.training.yeti_map import build_fixed_nodes

    node_by_ident = {nd.ident: nd for nd in build_fixed_nodes(lvl)}
    mandatory_pos = set()
    for ident in reward_idents:
        nd = node_by_ident.get(ident)
        if nd is not None:
            mandatory_pos.add(((nd.x - 8) // 4, lvl.floor_top_y[nd.floor]))

    for wid, (x_ram, y_px, floor) in yeti.waypoints(level).items():
        # Prefer the graph ident with the same name; else the aliased jump node.
        node_ident = wid if wid in node_by_ident else None
        if node_ident is None:
            for ident, p in (
                (i, ((n.x - 8) // 4, lvl.floor_top_y[n.floor]))
                for i, n in node_by_ident.items()
            ):
                if p == (x_ram, y_px) and ident.startswith("J"):
                    node_ident = ident
                    break
        out.append(
            Target(
                id=wid,
                kind="waypoint",
                trigger="position",
                node_ident=node_ident,
                pos=(x_ram, y_px),
                floor=floor,
                mandatory=(wid in reward_idents) or ((x_ram, y_px) in mandatory_pos),
                seedable=True,
            )
        )

    # --- princess: detected by a RAM FLAG (terminal) -------------------------
    px, py = lvl.princess_centre_px
    out.append(
        Target(
            id="princess",
            kind="princess",
            trigger="flag",
            node_ident="princess",
            pos=((int(px) - 8) // 4, int(py)),
            floor=lvl.princess_floor,
            mandatory=True,
            seedable=False,  # terminal: nothing to practise FROM
        )
    )
    return out


def targets_by_id(level: int = 1) -> Dict[str, Target]:
    """``build_targets`` keyed by id."""
    return {t.id: t for t in build_targets(level)}


__all__ = ["Target", "build_targets", "targets_by_id"]
