"""Root pools: extra level starts, grouped by snowball sequence.

A level start (L1's game reset, L5's save) always plays the same snowball sequence: the
route choices come from a generator that a level start fixes, and the player's inputs
do not change them. A policy trained only there can memorise one sequence instead of
reacting (l1_v17: 215/300 from reset, 0/330 from 198 real respawns covering 26 other
sequences). The game makes new sequences itself: every death leads to a respawn whose
snowballs depend on how the previous life ended. A root pool is a set of such respawn
states, harvested by scripts/mo5/yeti/harvest_respawns.py.

From L2 on there is no single level start either: the hazards a level opens with depend
on when the previous level was finished. Those levels' pools also hold ENTRIES (the
level as it opens after the previous one, finished at different moments), harvested by
the same script. A level's pool is named in experiments/003-yeti/registry.yaml
(``start_pool``) and the referee scores the level on its held-out sequences.

A MANIFEST (tracked JSON) lists each state with its sha256, its sequence id and whether
that sequence is HELD OUT (never trained on, used to score whether a policy reacts).
The states are local files: a save holds the game program, so it is never committed.

Drawing is uniform over SEQUENCES, then over the states of the drawn sequence, so a
sequence harvested 51 times weighs the same as one harvested once.
"""

from __future__ import annotations

import hashlib
import json
import random as _random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

REPO = Path(__file__).resolve().parents[3]
SPLITS = ("train", "heldout", "all")


@dataclass
class RootPool:
    manifest: str
    manifest_sha256: str
    game_level: int
    layout: int
    split: str
    # sequence id -> [(repo-relative path, state bytes)]
    sequences: Dict[int, List[tuple]]

    @property
    def n_states(self) -> int:
        return sum(len(v) for v in self.sequences.values())

    def draw(self, rng=_random) -> tuple:
        """``(path, bytes)``: a sequence uniformly, then one of its states uniformly."""
        seq = rng.choice(sorted(self.sequences))
        return rng.choice(self.sequences[seq])


def load(manifest: str, split: str = "train", check: bool = True) -> RootPool:
    """Load ``manifest`` and the states of ``split`` (train | heldout | all).

    Every state's sha256 is checked against the manifest: a pool whose files changed is
    not the pool the manifest describes, and the held-out split is only held out while
    the files are the ones it lists.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, not {split!r}")
    mpath = Path(manifest)
    if not mpath.is_absolute():
        mpath = REPO / mpath
    raw = mpath.read_bytes()
    data = json.loads(raw)
    seqs: Dict[int, List[tuple]] = {}
    for st in data["states"]:
        if split != "all" and bool(st["heldout"]) != (split == "heldout"):
            continue
        p = Path(st["path"])
        body = (p if p.is_absolute() else REPO / p).read_bytes()
        if check and hashlib.sha256(body).hexdigest() != st["sha256"]:
            raise ValueError(f"{st['path']}: sha256 differs from {manifest}")
        seqs.setdefault(int(st["sequence"]), []).append((st["path"], body))
    if not seqs:
        raise ValueError(f"{manifest}: no states in split {split!r}")
    return RootPool(
        manifest=str(manifest),
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
        game_level=int(data["game_level"]),
        layout=int(data["layout"]),
        split=split,
        sequences=seqs,
    )


def read_manifest(manifest: str) -> dict:
    """The manifest's JSON alone, without reading or checking any state file.

    For callers that only need what a pool IS (its game level, layout, sequences) --
    the referee's plan and --check, which run where the local state files may be absent
    (CI). Anything that plays from the states goes through :func:`load`.
    """
    mpath = Path(manifest)
    if not mpath.is_absolute():
        mpath = REPO / mpath
    return json.loads(mpath.read_text(encoding="utf-8"))


def describe(pool: Optional[RootPool]) -> str:
    if pool is None:
        return "none"
    return (
        f"{pool.manifest} [{pool.split}]: {pool.n_states} states in "
        f"{len(pool.sequences)} snowball sequences, game level {pool.game_level}"
    )


__all__ = ["RootPool", "load", "read_manifest", "describe", "SPLITS"]
