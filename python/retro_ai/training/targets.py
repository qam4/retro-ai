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
    # (x_ram, y_px). Set for EVERY kind, because fruits and the princess have a
    # location that feeds path-distance shaping -- but only ``trigger == "position"``
    # targets are DETECTED by it (see :meth:`reached`).
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

    def reached(self, x_ram: int, y_px: int, tol_x: int, tol_y: Optional[int] = None):
        """THE reach test for a positional target. See :func:`within_tol`.

        Callers must supply the tolerance because the two consumers currently use
        DIFFERENT values (see the module docstring); the point of routing both
        through here is that the comparison itself can no longer drift.
        """
        # Guard on the TRIGGER, not on ``pos``: fruits and the princess also carry a
        # pos (it feeds path-distance shaping), but they are detected by a RAM event
        # and a RAM flag. Answering a position question for them would silently
        # invent a second, wrong detector -- exactly what this module exists to stop.
        if not self.positional:
            raise ValueError(
                f"target {self.id!r} is detected by {self.trigger!r}, not position"
            )
        if self.pos is None:
            raise ValueError(f"target {self.id!r} has no position")
        return within_tol(self.pos, x_ram, y_px, tol_x, tol_y)


def within_tol(
    pos: Tuple[int, int],
    x_ram: int,
    y_px: int,
    tol_x: int,
    tol_y: Optional[int] = None,
) -> bool:
    """Is the agent at ``pos``, within tolerance? One implementation, two callers.

    ``pos`` is ``(x_ram, y_px)``. The agent's x is the RAM byte 0x2B52 in 4-PIXEL
    units; y is 0x2B51 in PIXELS. So a single ``tol`` means very different things
    per axis: ``tol=2`` is +-8 px horizontally but only +-2 px vertically. ``tol_y``
    defaults to ``tol_x`` to preserve that historical behaviour; pass it explicitly
    to be per-axis. Measured guidance is on
    ``CurriculumConfig.waypoint_tolerance``: tol_x 1, tol_y 2.

    THE TWO CALLERS, AND THE DIVERGENCE THIS DOES NOT YET FIX
    ---------------------------------------------------------
    Both use the SAME anchor (verified: positions differ for 0 targets on L3 and
    L4) but different tolerances:

    * curriculum reach/capture (``train_checkpoint_curriculum.py``) --
      ``waypoint_tolerance`` = 2 for ladder waypoints, ``max(6, tol)`` = 6 for
      jump waypoints.
    * reward milestone marking (``rewards.py``) -- ``waypoint_reward_tol`` = 2 for
      everything.

    That is why one anchor can report two answers. Measured on L4 ``Fr1``
    (anchor x_ram 60, floor 3's left edge): the curriculum box 54..66 catches the
    agent, the reward box 58..62 does not, because the agent only ever occupies
    64..68 on that platform (0 hits in 10 from-reset episodes, 120 grounded steps).
    The reward therefore never marks that milestone and keeps summing distance to
    it for the whole episode, at ~12x the base route gradient.

    Unifying the VALUES is a reward change and is deliberately NOT done here --
    see experiments/003-yeti/level4_notes.md for the staging and the L3 warning.
    """
    if tol_y is None:
        tol_y = tol_x
    wx, wy = pos
    return abs(int(x_ram) - wx) <= tol_x and abs(int(y_px) - wy) <= tol_y


def build_targets(level: int = 1) -> List[Target]:
    """Derive every target for ``level`` from its :class:`LevelMap`.

    Single source of truth, so the reward, the curriculum pools and the logs can
    never disagree about what exists or what it is called.
    """
    from retro_ai.games import yeti
    from retro_ai.training.yeti_map import get_level_map

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
    from retro_ai.training.yeti_map import build_fixed_nodes

    node_by_ident = {nd.ident: nd for nd in build_fixed_nodes(lvl)}

    # ALIAS MAP, derived from the jump-edge STRUCTURE, not from position equality.
    #
    # A jump landing has two names for one point: the curriculum calls it "A1" or
    # "Fr1", the graph calls it "J10_11_b". This used to be resolved by comparing
    # coordinates -- which silently broke the moment a level supplied a measured anchor
    # override (`LevelMap.jump_waypoint_pos`): the waypoint moved, the graph node did
    # not, the positions stopped matching, and BOTH `node_ident` and `mandatory` were
    # lost. Measured when L4's Fr1/Rope1 anchors were first moved: the mandatory set
    # dropped from 14 targets to 12, silently shortening the progress ladder.
    #
    # `_jump_graph` names an edge (fa, fb) as J{fa}_{fb}_a on fa and J{fa}_{fb}_b on
    # fb, and `jump_waypoints` names the ARRIVAL after the landing floor (always the
    # larger id, asserted in tests) with "_launch" for the departure pad. So the
    # correspondence is structural and needs no coordinates.
    alias_of: Dict[str, str] = {}
    if lvl.jump_edges and lvl.jump_waypoint_names:
        for fa, fb in lvl.jump_edges:
            nm = lvl.jump_waypoint_names.get(max(fa, fb))
            if not nm:
                continue
            arrival = f"J{fa}_{fb}_b" if fb > fa else f"J{fa}_{fb}_a"
            launch = f"J{fa}_{fb}_a" if fb > fa else f"J{fa}_{fb}_b"
            alias_of[nm] = arrival
            alias_of[f"{nm}_launch"] = launch

    for wid, (x_ram, y_px, floor) in yeti.waypoints(level).items():
        # Prefer the graph ident with the same name; else the structural alias.
        node_ident = wid if wid in node_by_ident else alias_of.get(wid)
        if node_ident is not None and node_ident not in node_by_ident:
            node_ident = None
        out.append(
            Target(
                id=wid,
                kind="waypoint",
                trigger="position",
                node_ident=node_ident,
                pos=(x_ram, y_px),
                floor=floor,
                # Mandatory-ness follows the NAME or its structural alias -- not the
                # coordinates, which move when an anchor is corrected.
                mandatory=(wid in reward_idents) or (node_ident in reward_idents),
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


__all__ = ["Target", "build_targets", "targets_by_id", "within_tol"]
