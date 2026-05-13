"""Per-step reward / state tracer for training-time debugging.

Used by ``SegmentEnv`` to record full per-step traces of episodes
whose total reward exceeds an analytically-computed bound. The
intent is forensic: a normal episode pays nothing extra, but if a
reward formula develops a farming exploit, the offending episode is
automatically pickled to disk with all the context needed to figure
out what happened.

The tracer is reward-formula-agnostic: callers pass in dicts of
"reward function internals" if they want them logged (e.g. the
per-fruit ``best_d`` dict for ``fruit_bonus_path_progress``).

File layout (output directory):
    reward_traces/
        env000_ep00012345_total926.72_bound2.48.pkl

Each pickle contains a dict with:
    - meta: {env_id, episode_id, start_state_hash, start_x, start_y,
             total_reward, bound, scale, fruit_scale}
    - steps: list of dicts, one per step:
        {step, agent_x, agent_y, agent_floor, fruits_present, action,
         done, truncated, reward, reward_state}
"""

from __future__ import annotations

import dataclasses
import os
import pickle
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class _TraceEntry:
    step: int
    agent_x: int
    agent_y: int
    agent_floor: Optional[int]
    fruits_present: tuple
    action: list
    done: bool
    truncated: bool
    reward: float
    reward_state: dict


@dataclass
class EpisodeTracer:
    """Records per-step data for one episode.

    Caller invokes ``record_step`` after every env step, then
    ``finalize_and_maybe_dump`` once the episode ends. If the total
    reward exceeds ``bound`` by more than ``epsilon``, the trace is
    written to disk.
    """

    env_id: int
    output_dir: str
    epsilon: float = 0.01
    steps: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def reset(self, meta: dict) -> None:
        """Start a new episode trace. Stores ``meta`` to be written
        alongside the per-step records."""
        self.steps = []
        self.meta = dict(meta)

    def record_step(
        self,
        step: int,
        agent_x: int,
        agent_y: int,
        agent_floor: Optional[int],
        fruits_present: tuple,
        action,
        reward: float,
        done: bool,
        truncated: bool,
        reward_state: dict,
    ) -> None:
        action_list = list(action) if hasattr(action, "__iter__") else [action]
        self.steps.append(
            _TraceEntry(
                step=step,
                agent_x=int(agent_x),
                agent_y=int(agent_y),
                agent_floor=agent_floor,
                fruits_present=tuple(bool(b) for b in fruits_present),
                action=[int(a) for a in action_list],
                done=bool(done),
                truncated=bool(truncated),
                reward=float(reward),
                reward_state=dict(reward_state),
            )
        )

    def finalize_and_maybe_dump(
        self, total_reward: float, bound: float
    ) -> Optional[str]:
        """End the episode. If total exceeded bound, dump the trace.

        Returns the output path if dumped, else None.
        """
        self.meta["total_reward"] = float(total_reward)
        self.meta["bound"] = float(bound)
        self.meta["bound_exceeded"] = total_reward > bound + self.epsilon
        if not self.meta["bound_exceeded"]:
            return None
        os.makedirs(self.output_dir, exist_ok=True)
        ep_id = int(self.meta.get("episode_id", 0))
        fname = (
            f"env{self.env_id:03d}_ep{ep_id:08d}"
            f"_total{total_reward:.2f}_bound{bound:.2f}.pkl"
        )
        path = os.path.join(self.output_dir, fname)
        payload = {
            "meta": self.meta,
            "steps": [dataclasses.asdict(s) for s in self.steps],
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        return path


def compute_path_progress_bound(
    nav,
    agent_pix_x: int,
    agent_floor: int,
    fruits_present: tuple,
    progress_scale: float,
    fruit_scale: float,
    initial_bonus: int,
) -> float:
    """Maximum reward achievable in a single episode under
    ``fruit_bonus_path_progress`` semantics.

    The progress term is bounded by the sum of initial path distances
    to each remaining fruit (since best_d ratchets monotonically
    downward for each fruit, all the way to 0 if the fruit is
    collected). The pickup term is bounded by collecting all
    remaining fruits at the initial bonus value. Both are upper
    bounds — actual play will pay less because:
      - bonus drops while the agent is moving (so fruit collected
        late pays less)
      - the agent rarely actually reaches every fruit

    But "above this bound" is unambiguously a bug.
    """
    progress_bound = 0.0
    pickup_bound = 0.0
    for i, present in enumerate(fruits_present, start=1):
        if not present:
            continue
        d = nav.path_distance_from_agent(agent_floor, agent_pix_x, f"F{i}")
        progress_bound += d * progress_scale
        pickup_bound += initial_bonus * fruit_scale
    return progress_bound + pickup_bound


__all__ = [
    "EpisodeTracer",
    "compute_path_progress_bound",
]
