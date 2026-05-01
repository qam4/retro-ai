"""Aggregation of per-episode rows into TensorBoard-friendly scalar tags.

Both the live ``EpisodeMetricsCallback`` (during training) and the
one-shot ``scripts/episodes_to_tb.py`` (replay after the fact) call
:func:`aggregate` to compute a flat ``{tag: value}`` dict from a list
of episode rows. Keeping the aggregation in one place guarantees the
two paths stay consistent.

Row schema
----------

Rows are plain dicts with the columns defined by
:data:`retro_ai.training.run_manifest.EPISODE_COLUMNS`. Only a few
fields are used by aggregation:

- ``start_level`` and ``reached_level`` (int-like)
- ``length`` (int-like, PPO steps)
- ``end_reason`` (str)

All other fields are ignored here but persist in the CSV.

Tag scheme
----------

All tags are slash-namespaced so TB groups them neatly:

- ``reach/from_{S}/ge_{L}``     — fraction of episodes with
  ``start_level=S`` whose ``reached_level >= L``.
- ``length/from_{S}/reached_{R}/mean`` — mean PPO steps for episodes
  starting at S and ending at R.
- ``end_reason/{reason}/fraction``  — fraction of episodes in the
  window ending for that reason.
- ``n_episodes/from_{S}``           — how many episodes starting at S
  fell in the window. Noise-checking helper for the rates above.
- ``n_episodes/total``              — total episodes in the window.

Missing combinations (e.g. no episodes with ``start_level=2`` in this
window) are simply omitted from the returned dict.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Mapping, Sequence


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def aggregate(
    episodes: Sequence[Mapping[str, Any]],
    max_level: int,
    min_n: int = 5,
) -> Dict[str, float]:
    """Flatten a window of episode rows into ``{tag: scalar}``.

    Parameters
    ----------
    episodes
        Each entry must at minimum contain ``start_level``,
        ``reached_level``, ``length``, and ``end_reason``.
    max_level
        Highest level to consider for ``reach/`` and ``length/`` tags.
        Typically the highest ``reached_level`` ever seen in the run.
    min_n
        Skip a reach-rate or mean-length tag if the number of matching
        episodes in the window is below this. Avoids publishing very
        noisy tags at the start of training.
    """
    out: Dict[str, float] = {}
    total = len(episodes)
    if total == 0:
        return out

    # Group by start_level for the reach + length tags.
    by_start: Dict[int, list] = {}
    for row in episodes:
        s = _as_int(row["start_level"])
        by_start.setdefault(s, []).append(row)

    out["n_episodes/total"] = float(total)

    for s in range(max_level + 1):
        eps = by_start.get(s, [])
        out[f"n_episodes/from_{s}"] = float(len(eps))
        if len(eps) < min_n:
            continue
        # Reach rates: fraction of these episodes with reached_level >= L.
        for L in range(1, max_level + 1):
            hit = sum(1 for r in eps if _as_int(r["reached_level"]) >= L)
            out[f"reach/from_{s}/ge_{L}"] = hit / len(eps)

        # Mean length per (start, reached) pair.
        by_reached: Dict[int, list] = {}
        for r in eps:
            by_reached.setdefault(_as_int(r["reached_level"]), []).append(
                _as_int(r["length"])
            )
        for reached, lengths in by_reached.items():
            if len(lengths) < min_n:
                continue
            out[f"length/from_{s}/reached_{reached}/mean"] = sum(lengths) / len(lengths)

    # End reason distribution over the whole window.
    reasons: Counter = Counter()
    for r in episodes:
        reasons[r.get("end_reason") or "unknown"] += 1
    for reason, count in reasons.items():
        out[f"end_reason/{reason}/fraction"] = count / total

    return out


def infer_max_level(episodes: Sequence[Mapping[str, Any]]) -> int:
    """Highest level seen in ``start_level`` or ``reached_level``.

    Returns 0 if there are no episodes.
    """
    lo = 0
    for r in episodes:
        lo = max(lo, _as_int(r["start_level"]), _as_int(r["reached_level"]))
    return lo


__all__ = ["aggregate", "infer_max_level"]
