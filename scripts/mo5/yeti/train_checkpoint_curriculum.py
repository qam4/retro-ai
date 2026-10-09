#!/usr/bin/env python3
"""Checkpoint curriculum training for Yeti.

Trains a single PPO policy progressively:

1. Start from game reset, learn to collect fruit 1
2. Save states when fruit 1 is collected
3. Mix game-start + frontier-checkpoint starts for the next segment
4. Continue until all fruits + princess

Configuration is YAML-driven — pass ``--config`` pointing at a file with
``training``, ``env``, ``ppo``, ``reward``, and ``curriculum`` sections.

Example::

    python scripts/mo5/yeti/train_checkpoint_curriculum.py \\
        --config experiments/003-yeti/configs/curriculum_v6.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import threading
import time
from collections import Counter
from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from retro_ai.games import yeti
from retro_ai.training.callbacks import EpisodeMetricsCallback
from retro_ai.training.env_builder import build_training_env
from retro_ai.training.rewards import (
    RewardContext,
    RewardFn,
    reset_reward,
    restore_reached_waypoints,
)
from retro_ai.training.rewards import create as create_reward
from retro_ai.training.run_config import RunConfig
from retro_ai.training.run_manifest import (
    EpisodeLogger,
    RunManifest,
    seed_everything,
)
from retro_ai.training.targets import reaches
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

# Yeti RAM addresses
FRUITS_ADDR = 11055
# Per-fruit presence bytes (non-zero = on map, zero = collected).
FRUIT_PRESENCE_ADDRS = {1: 0x2FAD, 2: 0x2F00, 3: 0x2E68, 4: 0x2DD8}
LIVES_ADDR = 11095
BONUS_HI = 11010
BONUS_LO = 11011
SCORE_HI = 11093
SCORE_LO = 11094
X_POS = 11090
Y_POS = 11089
# Player sprite-pose index (0x2B54). Surface codes {0-5 walk, 8 ladder};
# airborne/falling {9,10 jump, 11 fall, 12 death-anim}. Used by pose-gated
# rewards to freeze shaping while airborne. See experiments/003-yeti-
# training.md "run 3" for the measured table.
POSE_ADDR = 11092
# Poses where the agent is on a surface (grounded floor / ladder). Seeds are
# only snapshotted while grounded.
#
# This comment used to justify that with "a mid-jump state inherits a fall on
# reload". That is FALSE and was measured: save a pose-9 state mid-arc, reload
# it, feed the same inputs, and the trajectory is frame-identical (y 114 -> 110
# -> 108 -> 110 -> 114, pose stays 9). A save-state is a full emulator snapshot,
# so velocity and jump counters come back with it; there is no mechanism for it
# to degrade into a fall.
#
# Grounded is also INSUFFICIENT for the property actually wanted: `Low2_launch`
# held 100 seeds that read grounded on floor 12's brink and fell one step after
# load. The direct test is `admit_requires_survival`, which keeps a capture only
# if the episode survived from it. Grounded remains a reasonable PREFERENCE --
# a mid-jump seed hands the agent a committed trajectory it cannot steer for the
# first few steps -- but it is a preference, not a correctness requirement.
# (Shared definition in retro_ai.games.yeti; re-exported here.)
SURFACE_POSES = yeti.SURFACE_POSES
# Pose 13 is the L3 ESCALATOR RIDE pose (measured): the agent stands on a
# descending platform, pinned at the wall, riding y94->158 alive. It is NOT a
# fall (that is pose 11) — reloading a pose-13 state resumes a survivable
# descent that just needs a right-jump exit. So it is a legitimate state to
# SEED from, even though it is not a static-floor pose. We therefore extend the
# *seeding* allow-list with it (a general rule, not per-waypoint), while
# deliberately leaving the reward's own airborne set unchanged (the ride stays
# passively shaped there). The survival gate still rejects rides that lead
# nowhere, so this only ever seeds on-escalator states the agent truly reaches.
POSE_ESCALATOR_RIDE = 13
SEED_POSES = frozenset(SURFACE_POSES | {POSE_ESCALATOR_RIDE})
# Level-cleared flag. See scripts/mo5/yeti/train_segment.py for the empirical
# justification (probe_princess_flag_long_baseline.py PASSes with zero
# false positives across 26k frames). Detect princess touch via 0->1
# rising edge.
PRINCESS_FLAG_ADDR = 11050


# Shared global-step & episode-id counters across threaded envs.
_global_step = 0
_global_step_lock = threading.Lock()
_episode_counter = 0
_episode_counter_lock = threading.Lock()


def _next_episode_id() -> int:
    global _episode_counter
    with _episode_counter_lock:
        _episode_counter += 1
        return _episode_counter


def _set_global_step(step: int) -> None:
    global _global_step
    with _global_step_lock:
        _global_step = step


def _get_global_step() -> int:
    with _global_step_lock:
        return _global_step


def _normalize_seed(s, default_source_cp: int = 0):
    """Coerce a stored seed to the current shape
    ``(source_cp, bonus, state_bytes, frame_stack, reached_targets)``.

    All legacy-format handling lives HERE, in one place, rather than being spread
    through the loader. Older files are shorter because fields were appended over
    time; a missing field takes its neutral value:

    ==========  ===================================================
    tuple len   missing field -> neutral value
    ==========  ===================================================
    5           (current shape)
    4           reached_targets -> empty (pre milestone-restore)
    3           frame_stack     -> None  (pre H-AB; reseeds on load)
    2           source_cp       -> caller's default
    other       treated as raw state bytes
    ==========  ===================================================
    """
    if not isinstance(s, tuple):
        return (default_source_cp, 0, bytes(s), None, frozenset())
    source_cp, bonus, state, stack, reached = (
        default_source_cp,
        0,
        None,
        None,
        frozenset(),
    )
    if len(s) >= 5:
        reached = frozenset(s[4] or ())
    if len(s) >= 4:
        stack = s[3]
    if len(s) >= 3:
        source_cp, bonus, state = int(s[0]), int(s[1]), s[2]
    elif len(s) == 2:
        bonus, state = int(s[0]), s[1]
    return (source_cp, bonus, bytes(state), stack, reached)


class StartPool:
    """A pool of captured start-states with a self-regulating goal-score.

    The shared base (by composition) for both start sources: fruit
    CHECKPOINTS and position WAYPOINTS. It owns exactly what they have in
    common:
      - ``states``: retained (source_cp, bonus, state_bytes) entries;
      - ``goal_score``: EMA of reached_level/total_goals from this start
        (the H-T self-regulating weight is ``1 - goal_score``);
      - admission/retention/eviction (reset-origin + diversity);
      - sampling.
    Level ORDERING, the reach-gate, per-segment success and frontier
    advancement are CP-specific and deliberately stay in CheckpointManager
    — a waypoint has none of those.
    """

    def __init__(self, capacity: int, reach_alpha: float = 0.02):
        self.capacity = int(capacity)
        self.reach_alpha = float(reach_alpha)
        self.states: list = []  # (source_cp, bonus, state_bytes, stack|None)
        self.goal_score = 0.0

    def __len__(self) -> int:
        return len(self.states)

    def update_goal_score(self, score: float) -> None:
        a = self.reach_alpha
        self.goal_score = (1 - a) * self.goal_score + a * float(score)

    def weight(self) -> float:
        """H-T allocation weight: starts you can't finish from get more."""
        return max(1.0 - self.goal_score, 1e-3)

    def sample(self):
        """A random (source_cp, bonus, state_bytes, stack, reached) entry."""
        return random.choice(self.states)

    def insert(self, source_cp, bonus, state_bytes, stack=None, reached=None):
        """Insert keeping the pool reset-origin AND diverse.

        Retention priority is source_cp (lower = closer to a reset
        trajectory = more on-distribution). When full, admit a newcomer
        only if it's at least as reset-origin as the worst tier we hold,
        evicting a UNIFORMLY RANDOM member of that worst tier (diversity-
        preserving, recency-biased). Returns "appended" (pool grew),
        "replaced" (evict+insert), or None (rejected).

        ``stack`` is the optional saved frame-stack blob (from
        PreprocessedEnv.export_frame_stack) captured at the SAME moment as
        ``state_bytes``; on load it restores the real motion history so the
        seeded start is on-distribution (H-AB). None for stack-less sources
        (offline seeds / file-based starts) which fall back to reseed.

        ``reached`` is the set of route-point ids this seed had ALREADY reached
        when captured (accumulated transitively: the producing episode's own
        inherited set UNION what it reached). It is restored into the reward on
        load so milestones behind the seed stop being summed as pending targets
        — without it the potential pays a seeded episode to RETREAT (measured on
        L3: the milestone-sum bottomed out at SN3, so seeds above it were paid
        to climb back down). Milestone progress is POSITIONAL, so unlike CP
        progress (fruit bytes in RAM) it does NOT survive a load_state.
        """
        entry = (
            int(source_cp),
            int(bonus),
            bytes(state_bytes),
            stack,
            frozenset(reached or ()),
        )
        if len(self.states) < self.capacity:
            self.states.append(entry)
            return "appended"
        worst_cp = max(e[0] for e in self.states)
        if entry[0] > worst_cp:
            return None
        worst_idxs = [i for i, e in enumerate(self.states) if e[0] == worst_cp]
        self.states[random.choice(worst_idxs)] = entry
        return "replaced"


# Sentinel top-level candidate representing the ENTIRE waypoint collection
# as one source in pick_start's draw (H-AK). Distinct object so it can't
# collide with an int CP level or a str waypoint id.
_WP_GROUP = object()
# Tags for the three-bucket partition (split_mandatory): the two non-reset
# buckets, and what kind of pool a member of a bucket is.
_MAND_GROUP = object()
_OTHER_GROUP = object()
_RUNG = object()
_WP = object()


class CheckpointManager:
    """Manages save state buffers for each fruit checkpoint.

    ``reset_fraction`` / ``frontier_fraction`` / ``earlier_fraction``
    control the start distribution. All three must sum to 1.0 (validated
    on construction).
    """

    def __init__(
        self,
        max_states_per_checkpoint: int,
        min_states_to_advance: int,
        reset_fraction: float,
        frontier_fraction: float,
        earlier_fraction: float,
        min_survival_steps: int = 30,
        reach_threshold: float = 0.15,
        segment_floor: float = 0.0,
        n_rungs: int = 4,
        mandatory_groups=None,
        gate_waypoints: bool = False,
        gate_waypoints_by_predecessor: bool = False,
        split_mandatory: bool = False,
        earned_progress_score: bool = False,
        admit_requires_survival: bool = False,
        admit_requires_grounded: bool = False,
    ):
        # PROGRESS LADDER SIZE. Historically this was the fruit count, so a pool
        # meant "N fruits collected" and the ladder had one step per fruit. That
        # starves a level whose fruits are few and deep: L3 has ONE fruit at the
        # summit, so the ladder had a single step (0 -> 1) and `cp=[0, 100]` /
        # `success=[0->1]` carried no information, while "reached the next
        # checkpoint" (used by seed admission) could effectively never fire.
        #
        # A pool is now keyed by how many MANDATORY TARGETS are done — fruits,
        # the princess, and (where a level defines them) waypoint milestones. On
        # L1/L2 that is IDENTICAL to the fruit count, because those levels define
        # no milestones, so their pools and champions are untouched; on L3 it
        # turns 1 step into 12, which is the curriculum granularity that
        # plausibly made waypoints work on L2 in the first place.
        #
        # ``mandatory_groups`` is a list of frozensets: one per route STEP, each
        # holding every name that satisfies it -- alternative routes to the same
        # place, plus the graph alias a jump landing carries (the curriculum's
        # "A1", the graph's "J10_11_b"). A step is done when ANY member is
        # reached, so nothing double-counts. None => fall back to fruit-count
        # keying. See `_progress_ladder` for the L4 cases that forced this.
        self.mandatory_groups = [frozenset(g) for g in (mandatory_groups or [])]
        self.mandatory_ids = {name for g in self.mandatory_groups for name in g}
        self.N_RUNGS = int(n_rungs)
        # ANCHOR PROVENANCE for waypoint pools: wp_id -> (x_ram, y_px) as the level
        # defined it when these states were captured. Persisted, and compared on load
        # so a pool captured around an OLD anchor is discarded rather than inherited.
        #
        # Why this is needed: the resume path already drops pools for waypoints the
        # level no longer DEFINES, but that is a check on the NAME. Moving an anchor
        # keeps the name, so the stale pool sails through -- and a pool is exactly the
        # thing an anchor move is meant to invalidate, because its states were selected
        # by proximity to the old position.
        #
        # Measured on L4: `Low2_launch` was anchored at px 184, floor 12's tile edge,
        # where the agent reads as grounded for one frame and then falls. All 100 of its
        # seeds were unrecoverable, so the pool meant to teach the rope-2 crossing
        # taught falling instead and the agent never attempted the jump. Correcting the
        # anchor to px 188 fixes future captures; without this check a warm start would
        # have carried the 100 dead states forward and the fix would have done nothing.
        self.waypoint_anchors: Dict[str, Tuple[int, int]] = {}
        # Every sprite pose this run has observed, and how often. Surfaced in the status
        # line so an UNCATALOGUED pose cannot pass unnoticed again.
        #
        # An unknown pose is a silent behaviour change, not trivia: every pose-gated
        # decision (waypoint detection, seed capture, reward milestone marking, floor
        # crediting) reads an unrecognised code as "not on a surface", so whatever
        # happened in that frame does not count. Poses 6 and 7 are grounded left-walk
        # frames that were missing from SURFACE_POSES, which suppressed ~54% of grounded
        # frames on any leftward approach for the whole history of this project.
        self.pose_seen: Counter = Counter()
        # frontier_fraction / earlier_fraction are retained for config
        # back-compat but no longer used by pick_start (approach 30).
        # reset_fraction is reinterpreted as the CP0 floor and only
        # needs to be a valid probability.
        if not (0.0 <= reset_fraction <= 1.0):
            raise ValueError(
                f"reset_fraction (CP0 floor) must be in [0, 1], "
                f"got {reset_fraction}"
            )
        self.max_states_per_checkpoint = max_states_per_checkpoint
        self.min_states_to_advance = min_states_to_advance
        self.reset_fraction = reset_fraction
        self.frontier_fraction = frontier_fraction
        self.min_survival_steps = min_survival_steps
        # Approach 30: pick_start now weights non-reset levels by
        # (1 - success_rate) and reserves a fixed CP0 floor. We reuse
        # ``reset_fraction`` as that floor (frontier_fraction /
        # earlier_fraction are no longer used by pick_start, kept only
        # for config back-compat).
        self.cp0_floor = reset_fraction
        # Anti-starvation floor (H-flooring): fraction of the non-reset
        # allocation distributed UNIFORMLY across eligible segments,
        # blended with the (1 - success) weighting. 0.0 = pure
        # success-weighting (old behavior). >0 guarantees each eligible
        # segment a minimum share so a very-hard segment (e.g. CP4, ~2%
        # success -> weight ~1.0) can't starve its prerequisite
        # (CP3->CP4), which destabilized reach-4.
        self.segment_floor = float(segment_floor)

        # Each pool is a plain list of (source_cp, bonus, state_bytes)
        # entries.
        #   source_cp: the CP level the *episode that produced this
        #     snapshot* started from. Lower = closer to a reset
        #     trajectory = more on-distribution for P(princess|CP0).
        #     This is the primary retention key (approach 30, R2).
        #   bonus: in-game bonus countdown at the snapshot moment;
        #     retained for logging only — it no longer drives eviction.
        # When a pool is full we evict from the HIGHEST source_cp tier
        # (most artificial) first, picking a UNIFORMLY RANDOM entry
        # within that tier. (The old lowest-bonus tiebreak froze pools
        # onto a few high-bonus states -> the v6 collapse; random keeps
        # them diverse.) So the pool drifts toward reset-origin states
        # the agent actually reaches from reset.
        self.checkpoints = [
            StartPool(self.max_states_per_checkpoint) for _ in range(self.N_RUNGS + 1)
        ]
        # Waypoint start-pools, keyed by waypoint id (e.g. "L34_top").
        # Lazily created on first capture. Same StartPool as CP pools, so
        # they share the self-regulating goal-score weighting in pick_start
        # — but WPs are NON-GATING: no reach-gate, no seg_success, and they
        # never enter the reported reach/success metrics (success = fruit).
        self.waypoints: dict = {}
        self.wp_start_counts: dict = {}
        # WP diagnostics (display-only; not persisted). wp_closest[id] = the
        # smallest grounded Chebyshev distance (px) the agent has come to that
        # waypoint this run — surfaces "getting close but not capturing" and,
        # critically, whether the agent reaches a waypoint's vicinity AT ALL
        # (a pool stays absent until a within-tol grounded capture, so pools
        # alone can't distinguish "never went there" from "went near, missed").
        # wp_captures[id] = lifetime capture count (uncapped; pool size caps at
        # max_states_per_checkpoint).
        # ORDER-FREE segment health, keyed by start (int CP level or WP id):
        # P(an episode started here reaches at least one NEW route point it did
        # not start from / inherit). This is `seg_success` generalised to every
        # route point. It deliberately does NOT name a "next" point: waypoints
        # are unordered (decision #5) and levels branch, so a pairwise
        # start->next metric is ill-defined. Pairwise links stay a DIAGNOSTIC
        # (scripts/mo5/yeti/route_report.py over episodes.csv), which is where
        # they belong. A start whose progress is ~0 is a stuck hand-off — that is
        # what caught SN3 (seeded there, never reached anything new).
        self.progress_ema: dict = {}
        self.wp_closest: dict = {}
        self.wp_captures: dict = {}
        self.frontier = 0
        self.stats = {
            "saves": [0] * (self.N_RUNGS + 1),
            "starts": [0] * (self.N_RUNGS + 1),
            # How many snapshots we rejected for being too precarious
            # (died too soon under the policy, didn't reach next CP).
            "rejected_precarious": [0] * (self.N_RUNGS + 1),
            # Of those, how many were disqualified by the GROUNDED half of the gate
            # rather than by the step count. Split out because a lumped tally cannot
            # tell a gate that never fires from a gate with nothing to reject -- which
            # is how `admit_requires_grounded` stayed dead through a whole 1M run.
            "rejected_airborne": [0] * (self.N_RUNGS + 1),
            # Admission breakdown (H-R instrumentation): of the snapshots
            # offered to save_scored, how many were admitted because the
            # episode reached the next CP (reached_next) vs admitted only
            # because the agent survived >= min_survival frames. With
            # rejected_precarious (the third outcome) this shows what the
            # filter keeps vs throws away, per CP, over time.
            "admit_reached": [0] * (self.N_RUNGS + 1),
            "admit_survived": [0] * (self.N_RUNGS + 1),
        }
        # WP admission breakdown, keyed by wp_id (waypoints have no fruit
        # level so they can't share the per-CP arrays above). Waypoints now go
        # through the SAME play-based survival gate as fruit checkpoints
        # (see _admit_by_play): a grounded capture is only kept if the episode
        # that produced it survived >= min_survival steps OR reached the next
        # CP. This rejects doomed captures (e.g. a one-frame grounded clip on a
        # departing escalator platform, or a surface pose read mid-fatal-fall).
        self.wp_admit_reached: dict = {}
        self.wp_admit_survived: dict = {}
        self.wp_rejected_precarious: dict = {}
        # Subset of the above: rejected for ending the survival window airborne.
        self.wp_rejected_airborne: dict = {}
        self.segment_attempts = [0] * (self.N_RUNGS + 1)
        self.segment_successes = [0] * (self.N_RUNGS + 1)

        # Approach 31: reach-gated frontier curriculum.
        #
        # Two responsive EMAs replace the all-time cumulative rates as
        # the *decision* signal (the cumulative counters above are kept
        # only for human-readable display):
        #
        #   reset_reach_ema[n]  = P(an episode that STARTED FROM RESET
        #     reaches at least CP_n). This is the only honest evidence
        #     that the agent can get to CP_n on its own. A CP level is
        #     eligible as a start state only once this clears
        #     ``reach_threshold`` — below it, the level's pool is built
        #     from rare lucky reaches (off-distribution), so training
        #     there wastes budget and breaks composition (R2). Index 0
        #     is pinned at 1.0 (reset always "reaches" CP0).
        #
        #   seg_success_ema[n]  = P(advance | start at CP_n), EMA.
        #     Among eligible levels we weight by (1 - this) to
        #     concentrate budget on the deepest unsolved reset-reachable
        #     segment, and to shift the frontier forward as walls crack.
        #
        # EMAs (not cumulative rates) so the gate/weights track the
        # *current* policy: a segment that was hard at 2M steps and is
        # now solved should stop pulling budget, and a newly-reachable
        # deep CP should become eligible promptly.
        self.reach_threshold = reach_threshold
        # Apply the reach gate to WAYPOINT pools too (default False = the
        # historical asymmetry). See pick_start for the evidence both ways.
        self.gate_waypoints = bool(gate_waypoints)
        # Gate a waypoint on its PREDECESSOR's reach rather than its own, so the
        # frontier is reachable-by-one-step instead of already-reached. See
        # _wp_eligible and run_config.CurriculumConfig for the measurement.
        self.gate_waypoints_by_predecessor = bool(gate_waypoints_by_predecessor)
        # Travel order along the route. Set by the trainer after construction. Used
        # for the route table AND, when gating by predecessor, to find a waypoint's
        # predecessor -- so this is no longer display-only.
        self.route_order: list = []
        # Three-bucket start partition (reset | mandatory | other) instead of
        # reset | rungs | one waypoint group. Default False = legacy, so L1/L2/L3
        # are unchanged. See pick_start for why rungs belong with the mandatory
        # waypoints rather than being their own kind of start.
        self.split_mandatory = bool(split_mandatory)
        # Score an episode by what it EARNED over what it had left, instead of the
        # absolute depth it reached. See _episode_score. Default False = legacy.
        self.earned_progress_score = bool(earned_progress_score)
        # Admit a snapshot only if the agent SURVIVED from it, dropping the
        # inherited-credit `reached_next` shortcut. See _admit_by_play for the
        # measured poisoning and for the option we did not take.
        self.admit_requires_survival = bool(admit_requires_survival)
        # Require the capture to END ITS SURVIVAL WINDOW in a seedable pose, not merely
        # to be alive. See _admit_by_play for the measurement; default False.
        self.admit_requires_grounded = bool(admit_requires_grounded)
        self.reach_alpha = 0.02
        # Index 0..N = reach CP0..CP_N from reset; index N+1 = reach the
        # PRINCESS from reset (the actual win condition). Index 0 pinned 1.
        self.reset_reach_ema = [1.0] + [0.0] * (self.N_RUNGS + 1)
        self.seg_success_ema = [0.0] * (self.N_RUNGS + 1)
        # Per-WP reach EMA from RESET-origin episodes (parity with
        # reset_reach_ema for CPs): P(a from-reset episode reaches this WP).
        # This is the CHAINING signal, complementary to WP pool size (the
        # FRONTIER / local-reachability signal, which is inflated by the
        # reverse curriculum seeding each pool from the ones below it). A WP
        # with a full pool but a near-0 reach_ema is a chaining block the pool
        # sizes hide (e.g. v5: Lsc1 pool 100 but from-goat reach ~1%). Keyed by
        # WP id; only updated on reset-origin episodes. Display/diagnostic only,
        # non-gating (curriculum decisions are unchanged; L1/L2 byte-identical).
        # See experiments/003-yeti/curriculum_cp_wp_model.md.
        self.wp_reach_ema: dict = {}
        # H-T: per-start EMA of the AGGREGATE goal score =
        # reached_level / total_goals (4 fruits + princess = 5). Unlike
        # seg_success_ema (boolean "advanced at least one CP"), this only
        # saturates toward 1.0 when the policy actually reaches the
        # PRINCESS from that start — so an early segment that clears its
        # own CP but dies before the princess keeps a real score deficit
        # (and budget), instead of looking "solved" and being starved.
        # pick_start weights every level by (1 - this); the anti-
        # starvation floor and fixed reset reserve fall out naturally.
        # NOTE: the goal-score EMA now lives PER-POOL on each StartPool
        # (self.checkpoints[level].goal_score) so it unifies with waypoint
        # pools; see StartPool.

    def _episode_score(self, reached_level, start_rung, total_goals) -> float:
        """How well did THIS episode do, from where it began?

        Legacy (``earned_progress_score`` off): ``reached_level / total_goals`` --
        absolute depth reached. That cannot equalise and it rewards inheritance:
        seed at Step with 12 of 14 targets banked, die instantly, and you score
        12/14 = 0.86; travel the whole way from reset to Step and die, and you also
        score 0.86. One did nothing, the other did twelve. Since deep seeds hand
        over more, they always score high, so ``1 - goal_score`` gives them the
        SMALLEST sampling weight. Measured on L4 v2: Step 0.643 (the frontier, and
        the lowest weight of any eligible start) versus Lfruit_top 0.549, which is
        already solved at 1.00.

        Earned (flag on): what the episode ADDED over what it had LEFT to do --
        ``(reached - start) / (total - start)``. Step now scores 0/2 = 0.00 and
        Lfruit_top ~0.86, so the frontier gets the largest weight. It also
        equalises: perfect play from any start scores 1.0, which is what the
        self-regulating rule needs to converge.
        """
        if not self.earned_progress_score or start_rung is None:
            return reached_level / float(total_goals)
        left = float(total_goals - start_rung)
        if left <= 0:
            return 1.0
        gained = float(reached_level - start_rung)
        return max(0.0, min(1.0, gained / left))

    def record_episode(
        self,
        start_level,
        reached_level,
        reached_wps=None,
        all_wps=None,
        progressed=None,
        start_rung=None,
    ):
        total_goals = self.N_RUNGS + 1
        # Order-free segment health for EVERY start (CP level or WP id), updated
        # before the WP early-return so waypoints get it too. See progress_ema.
        if progressed is not None:
            a = self.reach_alpha
            prev = self.progress_ema.get(start_level, 0.0)
            self.progress_ema[start_level] = (1 - a) * prev + a * (
                1.0 if progressed else 0.0
            )
        # Waypoint start (id is a str, e.g. "L34_top"): update ONLY that
        # WP pool's goal-score (the self-regulating sampling weight). WPs
        # are non-gating, so they never touch seg_success / reset_reach /
        # the reported success metrics.
        if isinstance(start_level, str):
            pool = self.waypoints.get(start_level)
            if pool is not None:
                pool.update_goal_score(
                    self._episode_score(reached_level, start_rung, total_goals)
                )
            return
        if not (0 <= start_level <= self.N_RUNGS):
            return
        # Cumulative counters (display only).
        self.segment_attempts[start_level] += 1
        advanced = reached_level > start_level
        if advanced:
            self.segment_successes[start_level] += 1
        # Responsive per-segment success EMA (drives frontier weight).
        a = self.reach_alpha
        self.seg_success_ema[start_level] = (1 - a) * self.seg_success_ema[
            start_level
        ] + a * (1.0 if advanced else 0.0)
        # H-T: aggregate goal-score EMA — fraction of the 5 total goals
        # (4 fruits + princess) reached from this start. reached_level is
        # 0..5 (5 = princess via the H-M fix). This is what pick_start
        # weights by (1 - score).
        total_goals = self.N_RUNGS + 1
        self.checkpoints[start_level].update_goal_score(
            self._episode_score(reached_level, start_level, total_goals)
        )
        # Reach-from-reset EMA: only reset (CP0) episodes are evidence
        # for "can the agent get to CP_n unaided". For each n in 1..4
        # the episode reached n iff reached_level >= n.
        if start_level == 0:
            # n up to N+1 = princess (reached_level==N+1 on a touch, via
            # the H-M fix), so reset_reach_ema[-1] is the live princess-
            # from-reset rate — the actual goal.
            for n in range(1, self.N_RUNGS + 2):
                hit = 1.0 if reached_level >= n else 0.0
                self.reset_reach_ema[n] = (1 - a) * self.reset_reach_ema[n] + a * hit
            # Same reset-origin evidence for every configured WP: did this
            # from-reset episode reach it? all_wps is the full WP id universe
            # so never-reached WPs stay at (and decay toward) 0.0 — that is the
            # block signal we want visible in the log.
            if all_wps:
                reached_wps = reached_wps or set()
                for wid in all_wps:
                    prev = self.wp_reach_ema.get(wid, 0.0)
                    hit = 1.0 if wid in reached_wps else 0.0
                    self.wp_reach_ema[wid] = (1 - a) * prev + a * hit

    def _insert(self, level, source_cp, bonus, state_bytes, stack=None, reached=None):
        """Insert into CP ``level``'s pool (delegates retention/eviction to
        StartPool), then update save stats and advance the frontier only
        when the pool actually GREW (append), matching prior behavior.
        """
        status = self.checkpoints[level].insert(
            source_cp, bonus, state_bytes, stack, reached
        )
        if status is None:
            return
        self.stats["saves"][level] += 1
        if status == "appended":
            self._maybe_advance_frontier()

    def save_checkpoint(
        self,
        fruits_collected,
        state_bytes,
        source_cp=0,
        bonus=0,
        stack=None,
        reached=None,
    ):
        # Used for offline seed_archive / preseed (no play-based score, and
        # no frame stack — those seeds fall back to reseed on load).
        if 0 <= fruits_collected <= self.N_RUNGS:
            self._insert(
                fruits_collected, source_cp, bonus, state_bytes, stack, reached
            )

    def save_scored(
        self,
        fruits_collected,
        state_bytes,
        survived_steps,
        reached_next,
        bonus,
        source_cp,
        stack=None,
        reached=None,
        end_pose=None,
    ):
        """Admit a checkpoint snapshot judged by *real play*, not a probe.

        ``survived_steps`` is how many gym steps the agent stayed alive after
        the snapshot, under its own policy, in the episode that
        produced it. ``reached_next`` is whether that same episode went
        on to collect the next fruit (or the princess). ``bonus`` is
        the in-game bonus countdown at the snapshot moment (retention
        tiebreaker). ``source_cp`` is the CP level the producing
        episode started from (primary retention key — lower is more
        reset-origin / on-distribution).

        Admission is lenient (approach 30): keep the snapshot if it
        either led to the next checkpoint OR the agent survived at
        least ``min_survival_steps`` gym steps from it. Leniency protects the
        rare reaches at hard, sparse CPs; retention priority
        (source_cp) does the quality work on full, easy CPs.
        """
        if not (0 <= fruits_collected <= self.N_RUNGS):
            return
        # Admission with a logged reason (H-R instrumentation):
        #   reached_next  -> episode went on to the next CP
        #   survived-only -> didn't reach next, but stayed alive >= min
        #   rejected      -> neither (the "collected fruit then died fast"
        #                    states the H-O fix started filtering out)
        verdict = self._admit_by_play(survived_steps, reached_next, end_pose=end_pose)
        if verdict == "reached":
            self.stats["admit_reached"][fruits_collected] += 1
        elif verdict == "survived":
            self.stats["admit_survived"][fruits_collected] += 1
        else:
            self.stats["rejected_precarious"][fruits_collected] += 1
            if self._airborne_at_window_end(end_pose):
                self.stats["rejected_airborne"][fruits_collected] += 1
            return
        self._insert(fruits_collected, source_cp, bonus, state_bytes, stack, reached)

    def _admit_by_play(self, survived_steps, reached_next, end_pose=None):
        """Shared play-based admission verdict for deferred snapshots.

        A single source of truth for BOTH fruit checkpoints (save_scored)
        and waypoints (save_waypoint) so the survival gate can't silently
        diverge between the two. Returns one of "reached" / "survived" /
        "rejected". Admission is lenient (approach 30): keep the snapshot if
        the producing episode either reached the next target OR stayed alive
        at least ``min_survival_steps`` gym steps from it.

        POISONED POOLS (measured on L4 v3). ``reached_next`` is computed over the
        WHOLE episode (``_max_cp_this_ep > start_level``), so a snapshot taken
        AFTER the episode's deepest point inherits credit for progress that
        happened BEFORE it. On L4 v3 that admitted corpses: of 20 sampled
        ``Lclimb3_top`` states 5 had the death flag (0x2AFC == 65) already SET in
        the saved bytes and 20/20 died within 60 NOOP steps; ``Low1_launch`` was
        3/20 flagged and 20/20 doomed. Every pool up to and including ``Step``
        was clean, i.e. the corruption starts exactly where the episode's
        deepest point starts landing before the capture.

        ``admit_requires_survival`` drops the ``reached_next`` shortcut, so a
        snapshot is kept only if the agent demonstrably SURVIVED from it. Default
        False because it changes admission on L1-L3 too, whose pools may have
        relied on the lenient path at sparse rungs (approach 30's stated reason
        for the leniency) -- so it needs the 3-seed + baseline sweep before it
        becomes the default.

        THE OTHER OPTION, deliberately not taken (recorded so it isn't
        rediscovered). Keep ``reached_next`` but ANCHOR it to the snapshot:
        remember the max rung at save time and admit only when
        ``max_rung_whole_episode > max_rung_at_save_time``, i.e. progress made
        AFTER the capture. That preserves the leniency the sparse rungs were
        given while removing the inherited credit, and it is the more faithful
        reading of "keep cp[n] if the episode went on to cp[n+1]". It was not
        chosen because it needs a new per-snapshot field threaded through
        save_scored/save_waypoint and the deferred-capture buffer, whereas
        survival-only is a one-line change to an existing signal -- and survival
        is the property the pool actually needs. Revisit if survival-only proves
        too strict at sparse rungs.

        ``admit_requires_grounded`` adds the missing half. "Alive after N steps" cannot
        see a state that falls off a platform and is CAUGHT by something. Measured on
        L4:
        81 of 100 `Low2_launch` seeds sit on px 184 -- floor 12's tile edge, where the
        agent reads grounded for one frame and then falls (0/8 survive a NOOP hold,
        while
        px 188 survives 8/8) -- and the trampoline below keeps them alive a MEDIAN OF 83
        STEPS against `min_survival_steps` 30. So 100/100 doomed seeds were admitted,
        and
        the pool meant to teach the rope-2 crossing taught the fall-bounce loop instead.
        Raising the threshold is NOT the fix: it would only have to beat one particular
        bounce cycle, and a different trampoline gives a different number.

        The criterion is therefore "ends the window in SEED_POSES", i.e. on a surface or
        on the L3 escalator. NOTE the escalator: an earlier draft used SURFACE_POSES and
        additionally required y to be unchanged, which rejects 25/25 of L3's `Lesc_top`
        seeds -- legitimate rides, and the y test fails an escalator by construction
        since
        the platform carries the agent down. Blast radius of the corrected criterion,
        measured over existing pools with debug/l4_survival_gate_blast.py: L4 rejects
        23 of
        415 (6%, ALL in `Low2_launch`), L3 rejects 5 of 463 (1%), and `Lesc_top` keeps
        25/25. Concentrated exactly where intended, so it is safe global rather than
        per-level.
        """
        if self._airborne_at_window_end(end_pose):
            return "rejected"
        if reached_next and not self.admit_requires_survival:
            return "reached"
        if survived_steps >= self.min_survival_steps:
            return "survived"
        return "rejected"

    def _airborne_at_window_end(self, end_pose) -> bool:
        """Whether the GROUNDED half of the gate is what disqualifies this capture.

        Callers use it to attribute a rejection, so `_admit_by_play` can keep returning
        the three verdicts its callers already switch on. `end_pose is None` means the
        episode ended before the window closed, which the step-count test already
        handles -- the gate must not reject on missing information.
        """
        return bool(
            self.admit_requires_grounded
            and end_pose is not None
            and int(end_pose) not in SEED_POSES
        )

    def save_waypoint(
        self,
        wp_id,
        state_bytes,
        survived_steps,
        reached_next,
        end_pose=None,
        source_cp=0,
        bonus=0,
        stack=None,
        reached=None,
    ):
        """Admit a WAYPOINT snapshot, judged by *real play* — same gate as
        ``save_scored`` (see _admit_by_play).

        A waypoint is a start-state, but it is only useful if it is
        SURVIVABLE: reloading a doomed grounded clip (e.g. a single frame on
        a departing escalator platform, or a surface-pose read during a fatal
        fall) poisons the pool with unrecoverable seeds. ``survived_steps`` is
        how many gym steps the agent stayed alive after the capture in the
        episode that produced it; ``reached_next`` is whether that episode went
        on to make forward CP progress. The pool is lazily created on first
        admitted capture and shares StartPool's reset-origin retention.
        ``stack`` is the frame-stack blob captured with the state (H-AB).
        """
        verdict = self._admit_by_play(survived_steps, reached_next, end_pose=end_pose)
        if verdict == "reached":
            self.wp_admit_reached[wp_id] = self.wp_admit_reached.get(wp_id, 0) + 1
        elif verdict == "survived":
            self.wp_admit_survived[wp_id] = self.wp_admit_survived.get(wp_id, 0) + 1
        else:
            self.wp_rejected_precarious[wp_id] = (
                self.wp_rejected_precarious.get(wp_id, 0) + 1
            )
            if self._airborne_at_window_end(end_pose):
                self.wp_rejected_airborne[wp_id] = (
                    self.wp_rejected_airborne.get(wp_id, 0) + 1
                )
            return
        pool = self.waypoints.get(wp_id)
        if pool is None:
            pool = StartPool(self.max_states_per_checkpoint, self.reach_alpha)
            self.waypoints[wp_id] = pool
        pool.insert(source_cp, bonus, state_bytes, stack, reached)
        self.wp_captures[wp_id] = self.wp_captures.get(wp_id, 0) + 1

    def note_wp_distance(self, wp_id, dist):
        """Record the closest (grounded) approach to ``wp_id`` seen this run.

        Display-only diagnostic; called every grounded step from the env for
        EVERY configured waypoint (not just captured ones), so we can see the
        agent nearing a waypoint even when it never lands within tol. Dict
        min-update is atomic under the GIL, matching the lock-free pattern the
        threaded envs already use for save_waypoint.
        """
        prev = self.wp_closest.get(wp_id)
        if prev is None or dist < prev:
            self.wp_closest[wp_id] = dist

    def rung_of(self, reached) -> int:
        """How many route STEPS a state has behind it.

        A step is a GROUP, satisfied by any one of its members, so alternative
        routes to the same place and the two names a jump landing carries each
        count once. Counting ids instead double-counted both -- see
        `_progress_ladder` for the measured L4 cases.

        The reached-set is already stored on every seed (added so a seeded
        episode stops re-targeting milestones behind it), so nothing has to be
        recaptured.
        """
        got = set(reached or ())
        return sum(1 for g in self.mandatory_groups if got & g)

    def _maybe_advance_frontier(self):
        while (
            self.frontier < self.N_RUNGS
            and len(self.checkpoints[self.frontier]) >= self.min_states_to_advance
        ):
            self.frontier = max(
                self.frontier,
                max(
                    i
                    for i in range(self.N_RUNGS + 1)
                    if len(self.checkpoints[i]) >= self.min_states_to_advance
                ),
            )
            break

    def _wp_predecessor(self, wid):
        """The route point immediately before ``wid``, or None if it is first/absent."""
        try:
            i = self.route_order.index(wid)
        except (ValueError, AttributeError):
            return None
        return self.route_order[i - 1] if i > 0 else None

    def _wp_eligible(self, wid) -> bool:
        """May ``wid`` be used as a START state?

        Three regimes:

        * gate off                 -- always (the historical asymmetry).
        * own-reach gate           -- ``wp_reach_ema[wid] >= reach_threshold``.
        * PREDECESSOR-reach gate   -- eligible if the agent reaches EITHER this point
          or the one immediately before it on the route.

        The own-reach rule is self-locking at the frontier: the frontier is the point
        the agent does not reach yet, so its reach is ~0, so it is never sampled, so
        the skill is never practised. Measured on L4: `Lclimb3_top`, `Low1` and
        `Low2_launch` each held 100 usable seeds and were sampled 0/7140 times, while
        `Step` -- immediately before `Lclimb3_top` -- was reached 0.81 from reset.

        Gating on the predecessor keeps the protection the gate exists for (a point
        whose predecessor is ALSO unreached stays shut, so the frontier opens one rung
        at a time) while making the one advanceable rung trainable.
        """
        if not self.gate_waypoints:
            return True
        thr = self.reach_threshold
        if self.wp_reach_ema.get(wid, 0.0) >= thr:
            return True
        if not self.gate_waypoints_by_predecessor:
            return False
        prev = self._wp_predecessor(wid)
        if prev is None:
            # First point on the route (or no route order): reset reaches it.
            return wid in self.route_order or not self.route_order
        return self.wp_reach_ema.get(prev, 0.0) >= thr

    def pick_start(self):
        """Pick a starting checkpoint level (H-T: aggregate-goal-score
        weighting — one rule for every level, no fixed reset reserve,
        no anti-starvation floor).

        Every start level CP0..CP4 competes in a single weighting:
        ``weight(level) = 1 - goal_score_ema[level]`` where goal_score is
        the EMA of (reached_level / 5) — how much of the whole journey
        (4 fruits + princess) the policy completes from there.

        Candidates are CP0 (reset, always available) plus any deeper
        level whose pool is non-empty and which passes the reach gate
        (``reset_reach_ema >= reach_threshold``; gate off when threshold
        is 0). Because a level's score only approaches 1.0 once the
        policy reaches the PRINCESS from it, no level is starved while it
        still matters, and reset (CP0) earns a substantial share on its
        own exactly when the from-reset journey is unsolved — shrinking
        as it improves. ``cp0_floor`` (reset_fraction) and
        ``segment_floor`` are retained as optional safety knobs; both
        default to 0 (pure weighting).

        WAYPOINTS compete as ONE GROUP (H-AK), not one-vote-per-waypoint.
        The group is a single top-level candidate whose weight is the
        MEAN of its members' ``1 - goal_score``, so it is invariant to
        the NUMBER of waypoints — adding a waypoint only re-slices the
        group's own budget and never shrinks reset/CP shares. (Summing a
        vote per waypoint let 16 of them crowd reset down to ~4% of
        starts on v8; grouping keeps reset ~26%.) If the group is drawn,
        a second draw within it picks a waypoint by ``1 - goal_score``,
        so a freshly-captured deep WP (goal_score 0 -> weight 1.0) still
        dominates the group and gets "more reps further down"
        automatically. With no waypoints the top-level draw is exactly
        the old CP-only draw (L1 / WP-off byte-identical).
        """
        # Optional hard reset floor (safety net only; 0 = pure weighting).
        if self.cp0_floor > 0.0 and random.random() < self.cp0_floor:
            self.stats["starts"][0] += 1
            return 0, None, None, frozenset()

        # Top-level sources, weighted by StartPool.weight() (= 1 -
        # goal_score, the H-T rule):
        #  - CP0 (reset) always; deeper CP levels once non-empty AND
        #    reset-reachable (reach gate);
        #  - ALL waypoints as ONE group (H-AK), weight = MEAN of members'
        #    weights (count-invariant; see the docstring). Non-gating.
        cp_candidates = [0]
        for n in range(1, self.N_RUNGS + 1):
            if self.checkpoints[n] and self.reset_reach_ema[n] >= self.reach_threshold:
                cp_candidates.append(n)
        # Waypoint pools are UNGATED by default: that asymmetry is deliberate and
        # is how the reverse curriculum bootstrapped L2. ``gate_waypoints`` makes
        # them honour the same reach gate the progress rungs do, using each
        # waypoint's own from-reset reach EMA -- the direct analogue of
        # ``reset_reach_ema`` for a position target.
        #
        # The case for it (L3 evidence): we drilled SN3 and the ascent heavily
        # from ungated pools, seeded skill improved measurably (A1_launch -> A1
        # went 5% -> 24-32%), and it composed to reset at 0.03%. Practising states
        # the agent cannot reach produced skill that did not transfer, and spent
        # ~40% of episodes doing it. The case against: L2's breakthrough came
        # through ungated drilling past the F3 goat, and we cannot tell from those
        # logs whether those waypoints were above 0.15 at the time (wp_reach did
        # not exist yet), so this may block exactly that kind of win.
        wp_candidates = [
            w
            for w, pool in self.waypoints.items()
            if len(pool) > 0 and self._wp_eligible(w)
        ]

        if self.split_mandatory:
            # THREE BUCKETS: reset | MANDATORY starts | OTHER starts.
            #
            # The legacy partition is reset | rungs | one waypoint group, and on
            # L4 that misallocates badly. A rung pool is only ever filled on a
            # FRUIT pickup, so with one fruit exactly one rung pool exists
            # ("just past the fruit") -- and as its own top-level candidate it
            # took 34.6% of all starts, re-practising ground already at 93%,
            # while every waypoint including the frontier shared the remaining
            # third at ~2% each.
            #
            # A rung pool is also not a distinct SITUATION: "3 targets done" on
            # L4 means "standing just past Fr2", which the Fr2 waypoint pool
            # already holds with a known position. So rungs are not a separate
            # kind of start -- they are mandatory starts without a name, and they
            # belong in the same bucket as the mandatory waypoints. On L1/L2,
            # where mandatory targets ARE the fruits and there are no waypoints,
            # that bucket is exactly the old rung set, which is why this unifies
            # instead of adding a concept.
            mand = [
                (_RUNG, n, self.checkpoints[n].weight()) for n in cp_candidates if n
            ]
            other = []
            for w in wp_candidates:
                bucket = mand if w in self.mandatory_ids else other
                bucket.append((_WP, w, self.waypoints[w].weight()))
            candidates = [0]
            weights = [self.checkpoints[0].weight()]
            groups = {}
            for tag, members in ((_MAND_GROUP, mand), (_OTHER_GROUP, other)):
                if not members:
                    continue
                groups[tag] = members
                # Mean, not sum: a bucket is ONE source however many it holds,
                # so adding starts never crowds out reset (the v8 failure).
                candidates.append(tag)
                weights.append(sum(m[2] for m in members) / len(members))
        else:
            candidates = list(cp_candidates)
            weights = [self.checkpoints[n].weight() for n in cp_candidates]
            wp_weights = [self.waypoints[w].weight() for w in wp_candidates]
            groups = {}
            if wp_candidates:
                # Mean, not sum: the group counts as one source regardless of
                # how many waypoints it holds.
                candidates.append(_WP_GROUP)
                weights.append(sum(wp_weights) / len(wp_weights))
        # Optional anti-starvation floor (default 0 -> pure weighting),
        # applied across the top-level sources.
        if self.segment_floor > 0.0 and len(candidates) > 1:
            total = sum(weights)
            k = len(candidates)
            f = self.segment_floor
            weights = [(1.0 - f) * (w / total) + f / k for w in weights]
        key = random.choices(candidates, weights=weights, k=1)[0]
        if key in groups:
            # Second draw inside the chosen bucket, same 1 - goal_score rule. A
            # bucket mixes rung pools and waypoint pools, so dispatch on the tag.
            members = groups[key]
            kind, ident, _w = random.choices(
                members, weights=[m[2] for m in members], k=1
            )[0]
            if kind is _WP:
                self.wp_start_counts[ident] = self.wp_start_counts.get(ident, 0) + 1
                _src, _bonus, state, stack, reached = self.waypoints[ident].sample()
                return ident, state, stack, reached
            self.stats["starts"][ident] += 1
            _src, _bonus, state, stack, reached = self.checkpoints[ident].sample()
            return ident, state, stack, reached
        # Waypoint group: second-level draw within it by 1 - goal_score.
        if key is _WP_GROUP:
            wp_key = random.choices(wp_candidates, weights=wp_weights, k=1)[0]
            self.wp_start_counts[wp_key] = self.wp_start_counts.get(wp_key, 0) + 1
            _src, _bonus, state, stack, reached = self.waypoints[wp_key].sample()
            return wp_key, state, stack, reached
        self.stats["starts"][key] += 1
        if key == 0:
            # A real game reset: nothing reached yet (and CP progress lives in
            # the emulator anyway), so the milestone set is empty.
            return 0, None, None, frozenset()
        # Pool entries are (source_cp, bonus, state_bytes, stack, reached).
        _src, _bonus, state, stack, reached = self.checkpoints[key].sample()
        return key, state, stack, reached

    def summary(self):
        rates = []
        for i in range(self.N_RUNGS + 1):
            if self.segment_attempts[i] > 0:
                pct = 100 * self.segment_successes[i] / self.segment_attempts[i]
                rates.append(f"{i}->{i+1}:{pct:.0f}%")
            else:
                rates.append(f"{i}->{i+1}:N/A")
        reach = "[" + ", ".join(f"{r:.2f}" for r in self.reset_reach_ema) + "]"
        gscore = "[" + ", ".join(f"{p.goal_score:.2f}" for p in self.checkpoints) + "]"
        # SEEDS, not "cp". Every seedable target -- fruits included -- owns a pool keyed
        # by NAME, so the old rung-keyed `cp=[...]` is gone. It printed N_RUNGS+1 slots
        # and could only ever fill the one a fruit pickup landed in: on L4 that was a
        # single slot out of fourteen, while `Lclimb3_top` (mandatory, rung 9) reported
        # zero forever because its states lived in a name-keyed pool. Two numbers that
        # cannot mislead: how many pools hold anything, and the total states held. The
        # per-pool detail is in route_table(), route-ordered.
        _pools = {k: len(v) for k, v in self.waypoints.items() if len(v)}
        base = (
            f"seeds={sum(_pools.values())} in {len(_pools)} pools "
            f"rejected={sum(self.wp_rejected_precarious.values())} "
            f"(air {sum(self.wp_rejected_airborne.values())}) "
            f"success=[{', '.join(rates)}] "
            f"reset_reach={reach} gscore={gscore}"
        )
        # Loudly, on every status line, if the run has seen a pose we cannot name.
        # Silence here is the assertion that the pose catalogue is complete.
        _unknown = yeti.unknown_poses(self.pose_seen)
        if _unknown:
            base += " | UNCATALOGUED POSES " + ", ".join(
                f"{p}x{self.pose_seen[p]}" for p in sorted(_unknown)
            )
        # Grounded frames the surface gate discards. Should be 0 now that the left-walk
        # cycle (poses 6/7) is admitted; a non-zero value means some other grounded pose
        # is still missing from SURFACE_POSES, which silently suppresses detection and
        # capture in those frames.
        _missed = sum(
            n
            for p, n in self.pose_seen.items()
            if p in yeti.KNOWN_POSES
            and "grounded" in yeti.POSE_NAMES[p]
            and p not in yeti.SURFACE_POSES
        )
        if _missed:
            base += f" | grounded frames NOT counted as surface: {_missed}"
        # The per-waypoint detail (pool size, reset-reach, approach distance,
        # capture/reject counts) used to be appended here as THREE parallel
        # walls, each sorted differently. It now lives in route_table(), printed
        # route-ordered at a lower frequency. This line stays fixed-size as
        # route points are added.
        if self.waypoints:
            base += f" | route[{len(self.waypoints)}]: {self.milestone_progress()}"
        return base

    def milestone_progress(self) -> str:
        """Compact scalar: how much of the route is RELIABLY reached from reset.

        Counts route points whose reset-origin reach EMA >= 0.5, so it answers
        "how many milestones are effectively done, and how far does the chain
        get" without listing every point. Detail lives in route_table().
        """
        if not self.wp_reach_ema:
            return "n/a"
        done = sum(1 for v in self.wp_reach_ema.values() if v >= 0.5)
        return f"{done}/{len(self.wp_reach_ema)} reached>=0.5 from reset"

    def route_table(self, route_order=None) -> str:
        """The canonical route view: one row per route point, route-ordered.

        Columns are the four signals that actually drive a decision, and they are
        1-D PROJECTIONS of the start x reached MATRIX (which is ~N^2 and belongs
        in episodes.csv, not a log — render it with route_report.py):
          reach     reset-origin reach EMA        -> does the chain compose?
          prog      order-free progress EMA       -> is this hand-off healthy?
          pool      seed pool size                -> exploration frontier
          near      closest grounded approach px  -> got near but never landed?
          cap/rej   admitted vs precarious-rejected captures, and of those
                    rejections how many were for ending the window AIRBORNE
                    (`air`). Without that split a gate that never fires and a
                    gate with nothing to reject read identically, which is how
                    `admit_requires_grounded` stayed dead through a 1M run.
        ``route_order`` is display-only (LevelMap.route_order); unlisted points
        are appended in a stable order so nothing is ever hidden.
        """
        ids = list(route_order or [])
        # Every FRUIT on the progress ladder gets a row from the first table on. A
        # fruit only enters the reach table once its pool exists, so a fruit nobody
        # had reached yet was simply absent -- which read as "this level has fewer
        # fruits", not as "not reached". It now shows with dashes until it has data.
        # Display only: the reach EMA and the start gate are untouched.
        fruits = sorted(
            {
                n
                for g in getattr(self, "mandatory_groups", ()) or ()
                for n in g
                if n[:1] == "F" and n[1:].isdigit()
            },
            key=lambda s: int(s[1:]),
        )
        rest = sorted(
            (set(self.waypoints) | set(self.wp_reach_ema) | set(fruits)) - set(ids),
            key=lambda s: (0, int(s[1:])) if s in fruits else (1, s),
        )
        ids += [w for w in rest if w not in ids]
        if not ids:
            return ""
        lines = [
            "  route                reach   prog   pool   near   cap/rej(air)",
        ]
        for wid in ids:
            pool = self.waypoints.get(wid)
            reach = self.wp_reach_ema.get(wid)
            prog = self.progress_ema.get(wid)
            near = self.wp_closest.get(wid)
            lines.append(
                "  {:<18s} {:>6s} {:>6s} {:>6s} {:>6s}   {}/{}({})".format(
                    wid,
                    "—" if reach is None else f"{reach:.2f}",
                    "—" if prog is None else f"{prog:.2f}",
                    "—" if pool is None else str(len(pool)),
                    "—" if near is None else str(near),
                    self.wp_captures.get(wid, 0),
                    self.wp_rejected_precarious.get(wid, 0),
                    self.wp_rejected_airborne.get(wid, 0),
                )
            )
        return "\n".join(lines)

    def save_to_disk(self, path):
        import pickle

        data = {
            "checkpoints": [
                list(self.checkpoints[i].states) for i in range(self.N_RUNGS + 1)
            ],
            "stats": self.stats,
            # Waypoint pools: id -> (states, goal_score). Optional; absent in
            # pre-WP checkpoint files (load tolerates that).
            "waypoints": {
                w: (list(p.states), p.goal_score) for w, p in self.waypoints.items()
            },
            # The anchor each pool was captured around, so a later run can tell
            # whether the position has since moved. See CheckpointManager.__init__.
            "waypoint_anchors": dict(self.waypoint_anchors),
        }
        # Atomic write: pickle to a temp file then os.replace, so a crash or
        # interrupt mid-write can never corrupt an existing pool file (matters
        # now that pools are also saved PERIODICALLY, see PoolSaveCallback).
        tmp = f"{path}.tmp"
        with open(tmp, "wb") as f:
            pickle.dump(data, f)
        os.replace(tmp, path)

    def load_from_disk(self, path):
        import pickle

        if not os.path.exists(path):
            return
        with open(path, "rb") as f:
            data = pickle.load(f)
        for i, states in enumerate(data["checkpoints"]):
            # Tolerate files written with a DIFFERENT ladder size: the progress
            # ladder used to be one step per fruit, so an L3 file has 2 entries
            # where the milestone ladder has 14. Extra trailing pools in the file
            # are ignored rather than crashing (they cannot be attributed to a
            # rung without the reached-set, which pre-fix files lack anyway).
            if i >= len(self.checkpoints):
                break
            for s in states:
                # Normalize to the (source_cp, bonus, state, stack) format.
                # Loaded/offline states have unknown origin; mark them
                entry = _normalize_seed(s, default_source_cp=i)
                # KEY BY THE LADDER. Both sources are the ladder key under the
                # writer that produced them: a stored reached-set gives the rung
                # directly, while a file written before the set existed was keyed
                # by fruit count -- which IS the rung on levels without
                # milestones. So prefer the recorded truth, else the file index.
                rung = self.rung_of(entry[4]) if entry[4] else i
                if rung > self.N_RUNGS:
                    continue  # from a level with a longer ladder; not ours
                if len(self.checkpoints[rung]) < self.max_states_per_checkpoint:
                    self.checkpoints[rung].states.append(entry)
        # Waypoint pools (optional; absent in pre-WP files).
        #
        # A pool's states were selected by proximity to the waypoint's anchor, so if
        # that anchor has since MOVED the states no longer describe the position they
        # are filed under and must not be inherited. The name-based stale check in
        # main() cannot see this: an anchor move keeps the name.
        saved_anchors = data.get("waypoint_anchors", {}) or {}
        moved, unknown = [], []
        for wp_id, payload in data.get("waypoints", {}).items():
            states, goal_score = payload
            now = self.waypoint_anchors.get(wp_id)
            was = saved_anchors.get(wp_id)
            if now is not None and was is not None and tuple(was) != tuple(now):
                moved.append((wp_id, tuple(was), tuple(now)))
                continue
            if now is not None and was is None:
                # Pre-provenance file: cannot verify, so the states are inherited as
                # before. Reported rather than silently trusted.
                unknown.append(wp_id)
            pool = StartPool(self.max_states_per_checkpoint, self.reach_alpha)
            for s in states:
                if len(pool) >= self.max_states_per_checkpoint:
                    break
                pool.states.append(_normalize_seed(s))
            pool.goal_score = float(goal_score)
            self.waypoints[wp_id] = pool
        for wp_id, was, now in moved:
            print(
                f"  Dropped inherited pool {wp_id!r}: anchor moved {was} -> {now}, "
                f"so its {len(data['waypoints'][wp_id][0])} state(s) were captured "
                "around a position this level no longer uses",
                flush=True,
            )
        if unknown:
            print(
                f"  {len(unknown)} inherited pool(s) predate anchor provenance and "
                f"could not be checked: {sorted(unknown)}",
                flush=True,
            )
        # Per-rung COUNTERS are deliberately not inherited. They are cumulative
        # display stats indexed by rung, and a file may have been written under a
        # different ladder, which makes its indices meaningless here. The pools
        # are the durable artifact; the counters describe a run. Keeping this
        # rule (rather than padding/truncating) means there is one array length
        # in play and no reader can index off the end.
        print(f"  Loaded checkpoints from {path}: {self.summary()}", flush=True)


# Module-level singleton. Populated in train(); shared across envs.
_manager: Optional[CheckpointManager] = None


class CheckpointCurriculumEnv(gym.Env):
    """Gym env with checkpoint-based curriculum for Yeti."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        cfg: RunConfig,
        reward_fn: RewardFn,
        env_id: int,
        episode_logger: Optional[EpisodeLogger] = None,
    ):
        super().__init__()
        stack = build_training_env(cfg.env.profile, cfg.env)
        self.base = stack.base
        self.gym_env = stack.gym
        self.preprocessed = stack.preprocessed
        self.action_space = stack.gym.action_space
        self.iface = stack.base._interface

        # --- Level awareness (defaults preserve level-1 behavior) ---
        # fruits_total, the per-fruit presence RAM addresses, and an
        # optional CP0 start-state save (level 2 boots from a save, not
        # a game reset). Pulled from the curriculum config.
        cur = cfg.curriculum
        self._fruits_total = getattr(cur, "fruits_total", 4) if cur else 4
        addrs = getattr(cur, "fruit_presence_addrs", None) if cur else None
        # Default to the level-1 dict; coerce YAML int keys/values to int.
        self._fruit_addrs = (
            {int(k): int(v) for k, v in addrs.items()}
            if addrs
            else dict(FRUIT_PRESENCE_ADDRS)
        )
        self._fruit_ids = sorted(self._fruit_addrs)
        self._start_state_bytes: Optional[bytes] = None
        start_path = getattr(cur, "start_state", None) if cur else None
        if start_path:
            with open(start_path, "rb") as _f:
                self._start_state_bytes = _f.read()

        # MultiInputPolicy support: when enabled, the observation becomes
        # a Dict of the image plus a fruit-presence vector (1.0 =
        # fruit still on map, 0.0 = collected). This de-aliases the
        # checkpoint states: a reset state and a "F1 already collected"
        # state look near-identical at 84x84, so a pixels-only policy
        # can't attach different actions to them (the v6 wall). The
        # explicit fruit vector makes the checkpoint observable.
        self._multi_input = bool(
            cfg.curriculum is not None
            and getattr(cfg.curriculum, "multi_input_obs", False)
        )
        image_space = stack.gym.observation_space
        if self._multi_input:
            self.observation_space = gym.spaces.Dict(
                {
                    "image": image_space,
                    "fruits": gym.spaces.Box(
                        low=0.0,
                        high=1.0,
                        shape=(self._fruits_total,),
                        dtype=np.float32,
                    ),
                }
            )
        else:
            self.observation_space = image_space

        self.max_steps = cfg.env.max_steps
        self.stall_threshold = cfg.env.stall_threshold
        self._reward_fn = reward_fn

        # Death detection via the fast 0x2AFC flag, gated to level >= 2 where
        # the lives byte is inert (so the lives-based branch below is dead
        # code there). Left OFF for level 1 so its termination timing is
        # byte-for-byte unchanged (no L1 regression). When on, it provides
        # ctx.died to the reward (the grounded reward suppresses shaping on a
        # fatal step) AND labels end_reason="death" (which the lives branch
        # cannot do on L2). The flag itself is validated cause-agnostic on
        # both levels; we simply don't re-wire L1's working path.
        _level = 1
        try:
            _level = int(cfg.reward.params.get("level", 1))
        except (AttributeError, TypeError, ValueError):
            _level = 1
        self._use_death_flag = _level >= 2

        # --- Waypoint curriculum (H-AI; off unless curriculum.waypoints) ---
        # Optional, non-gating start-seeds at computed ladder top/bottom
        # positions. Capture is grounded-only, once per waypoint per episode,
        # and never re-captures the waypoint an episode was seeded from.
        self._wp_enabled = bool(getattr(cur, "waypoints", False)) if cur else False
        # Geometry of the reach test, for BOTH detection and capture (see the block in
        # step() -- capture used to be pinned to the box and no longer is).
        # Forced into the reward params too, in the runner, so they cannot disagree.
        self._wp_reach_mode = str(
            getattr(cur, "waypoint_reach_mode", "sprite") or "sprite"
        )
        self._wp_tol = int(getattr(cur, "waypoint_tolerance", 2)) if cur else 2
        # Random no-op start (see the block in reset()). 0 = off.
        self._noop_start_max = int(getattr(cur, "noop_start_max", 0)) if cur else 0
        self._noop_start_scope = (
            str(getattr(cur, "noop_start_scope", "reset")) if cur else "reset"
        )
        # JUMP LANDINGS NEED A WIDER BOX THAN LADDERS. A ladder waypoint is a
        # single x_ram -- engagement literally requires the exact value (measured
        # on L4: ladder Lfruit climbs at x_ram 50, not 49 or 51). A jump, rope or
        # spring landing is wherever the arc drops you, and the waypoint sits on
        # the platform EDGE, so a 2-unit box is a knife edge the arrival usually
        # overshoots. Measured on L4 v1: Rope1 read 1.8% while Lclimb2_top -- only
        # reachable THROUGH Rope1's platform -- read 87%, i.e. the crossing was
        # happening ~87% of the time and the detector missed it. Reading that as a
        # wall would have sent us fixing a mechanic that already worked, which is
        # the SN3 mistake. Same rule for the metric and for capture, so the pools
        # and the numbers cannot disagree.
        self._wp_jump_tol = (
            int(getattr(cur, "jump_waypoint_tolerance", max(6, self._wp_tol)))
            if cur
            else 6
        )
        # {wp_id: (x_ram, y_px, floor)} detection targets from the tilemap.
        self._waypoints = yeti.waypoints(_level) if self._wp_enabled else {}
        # (2026-09-18) Waypoints excluded from SEEDING by config. Removing them here --
        # from the detection dict the capture loop and the start sampler both read --
        # means no states are captured there and no episodes start there. Any reward
        # term is unaffected: the reward keeps its own group list from LevelMap.
        #
        # The case this exists for is an EXPOSED spot. Seeding at L4 `Lclimb3_top` hands
        # the agent a hazard phase it did not have to earn (pool seeds survive a median
        # 8 NOOP frames against 3 for its own arrivals), so the timing decision the
        # level actually turns on never gets practised. See CurriculumConfig.
        # IT MUST STAY IN `self._waypoints`. That dict drives DETECTION, the
        # closest-approach display and `wp_reach_ema`, as well as capture. Dropping it
        # here (the first attempt) silently stopped tracking the waypoint: the route
        # table went from 30 rows to 29, so the one number the run exists to read --
        # does removing the pool cost its reach? -- became unobservable, and
        # `gate_waypoints_by_predecessor` lost the predecessor its successors gate on.
        # The exclusion is applied at CAPTURE instead (see step()): no captures means no
        # pool, and starts are sampled from pools.
        self._wp_seed_skip = frozenset(
            tuple(getattr(cur, "seed_waypoint_skip", ()) or ()) if cur else ()
        )
        if self._wp_seed_skip:
            unknown = sorted(w for w in self._wp_seed_skip if w not in self._waypoints)
            if unknown:
                # Fail loudly: a typo would otherwise silently seed the waypoint it was
                # meant to exclude, and the run would look like the control.
                raise SystemExit(
                    f"seed_waypoint_skip names unknown waypoints {unknown}; "
                    f"level {_level} has {sorted(self._waypoints)}"
                )
            print(
                f"[curriculum] seed_waypoint_skip: {sorted(self._wp_seed_skip)} will "
                "be DETECTED and tracked but never captured or seeded from",
                flush=True,
            )
        # Which waypoints are jump-edge ones (landings and launch pads), so they
        # get _wp_jump_tol instead of the ladder tolerance.
        if self._wp_enabled:
            from retro_ai.training.yeti_map import get_level_map, jump_waypoints

            try:
                self._wp_jump_ids = set(jump_waypoints(get_level_map(_level)))
            except (ValueError, KeyError):
                self._wp_jump_ids = set()
        else:
            self._wp_jump_ids = set()
        # FLAT per kind, deliberately. A scheme that narrowed each box to fit inside
        # its platform's standable span was tried and REVERTED: it caused a measured
        # regression and the reasoning is worth keeping.
        #
        # A jump's landing position depends on the POLICY that jumped. Measured on
        # floor 7, same seed pool, two policies: v6's champion lands at px 116, while a
        # later policy lands at px 108 and then jumps to 136..148. A 16 px box sized to
        # the first covered 3/163 grounded frames of the second -- 1.8%, the exact
        # figure this file's tolerance comment already recorded from L4 v1, where
        # `Rope1` read 1.8% while `Lclimb2_top`, reachable only THROUGH Rope1's
        # platform, read 87%.
        #
        # So the wide +-24 px jump box is not slack, it spans the variance a LEARNING
        # agent's landings actually have. Narrowing it took `Low1` and `Low2_launch`
        # from 0.61 to 0.00 in a controlled run (v6's warm start and seed, only the code
        # differing): the agent stopped reaching the rope-2 launch pad at all.
        #
        # The real defect that narrowing was aimed at is different: a box covering a
        # platform's LETHAL EDGE, where the agent reads grounded and falls next step,
        # poisons that waypoint's seed POOL. A box reaching into the void is harmless,
        # because detection is pose-gated. The fix belongs in capture admission --
        # reject a seed inside a lethal margin -- not in detection width. Not done yet;
        # see experiments/003-yeti/level4_notes.md.
        self._wp_tol_of = {
            wid: (self._wp_jump_tol if wid in self._wp_jump_ids else self._wp_tol)
            for wid in self._waypoints
        }
        self._start_wp = None  # the WP this episode was seeded from (skip re-save)
        self._captured_wps: set = set()  # WPs already captured this episode
        # WPs detected but still awaiting a grounded frame to save from. Mirrors
        # `_grounded_snap_due` on the fruit path; see the capture site.
        self._wp_snap_due: set = set()

        self.env_id = env_id
        self.episode_logger = episode_logger

        # Per-episode state
        self._step_count = 0
        self._prev_fruits = self._fruits_total
        self._prev_lives = 5
        self._prev_bonus = 0
        self._prev_score = 0
        self._stall = 0
        self._start_fruits = self._fruits_total
        self._initialized = False

        # Logging state
        self._start_xy = (0, 0)
        self._start_score = 0
        self._start_bonus = 0
        self._start_state_hash = ""
        self._episode_reward = 0.0
        self._fruits_collected_this_ep = 0
        self._episode_id = 0

    # -- helpers -------------------------------------------------------

    def _read_bonus(self) -> int:
        return (self.iface.read_ram_byte(BONUS_HI) << 8) | self.iface.read_ram_byte(
            BONUS_LO
        )

    def _read_score(self) -> int:
        return (self.iface.read_ram_byte(SCORE_HI) << 8) | self.iface.read_ram_byte(
            SCORE_LO
        )

    def _read_pos(self):
        return (
            self.iface.read_ram_byte(X_POS),
            self.iface.read_ram_byte(Y_POS),
        )

    def _fruit_vector(self) -> np.ndarray:
        """Fruit-presence vector (1.0 = on map, 0.0 = collected), one
        entry per fruit in this level (level 1 = 4, level 2 = 2)."""
        return np.array(
            [
                1.0 if self.iface.read_ram_byte(self._fruit_addrs[i]) != 0 else 0.0
                for i in self._fruit_ids
            ],
            dtype=np.float32,
        )

    def _wrap_obs(self, image_obs):
        """Wrap the raw image obs into the policy's observation format."""
        if self._multi_input:
            return {"image": image_obs, "fruits": self._fruit_vector()}
        return image_obs

    # -- gym API -------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        assert _manager is not None, "CheckpointManager not initialized"

        # Reset the (possibly stateful) reward at the episode boundary.
        # Without this, a stateful reward — e.g. the PBRS path-progress
        # reward's prev_phi / last_floor — leaks across episodes. That is
        # especially harmful for save-state starts (every curriculum CP, and
        # all of level 2): the agent is dropped at a new position but the
        # reward still carries the PREVIOUS episode's potential/floor, so the
        # first shaping step is a spurious cross-episode delta and (when the
        # new position's floor doesn't resolve, as at the level-2 spawn) the
        # whole episode is shaped on the stale last_floor.
        reset_reward(self._reward_fn)

        if not self._initialized:
            self.gym_env.reset(seed=seed)
            self._initialized = True

        level, state_bytes, start_stack, inherited_wps = _manager.pick_start()
        # A waypoint start returns a str id; remember it so (a) episode-end
        # record_episode credits the WP pool, and (b) we don't re-capture the
        # waypoint we were seeded at.
        self._start_wp = level if isinstance(level, str) else None
        self._captured_wps = set()
        self._wp_snap_due = set()
        # Milestones this seed had ALREADY banked when captured. Restored into
        # the reward below so they stop being summed as pending targets — else
        # the potential pulls a seeded episode BACKWARD (it is minimised behind
        # the seed). Kept for the transitive union on any capture this episode.
        self._inherited_wps: frozenset = frozenset(inherited_wps or ())
        restore_reached_waypoints(self._reward_fn, self._inherited_wps)
        # WPs the agent came within tol of this episode (grounded/ride pose),
        # for the reset-origin wp_reach_ema. Distinct from _captured_wps (which
        # excludes the seed WP + already-captured); here we want every reach.
        self._reached_wps_this_ep: set = set()

        if state_bytes is None and self._start_state_bytes is not None:
            # CP0 for a level that boots from a save-state rather than a
            # fresh game reset (level 2 starts from level2_start.sav). The
            # file-based start carries no frame stack -> reseed fallback.
            state_bytes = self._start_state_bytes

        if state_bytes is not None:
            self.base._interface.load_state(state_bytes)
            # Preferred (H-AB): restore the frame stack captured WITH this
            # seed, so the first observation is the real motion history the
            # policy saw live — on-distribution, and NO settle steps (which
            # otherwise advance the game ~20 frames past the snapshot and can
            # doom time-sensitive seeds; see the L12a goat investigation).
            restored = (
                start_stack is not None
                and self.preprocessed.restore_frame_stack(start_stack)
            )
            if restored:
                obs = self.preprocessed.current_observation()
            else:
                # Fallback for stack-less seeds (pre-H-AB checkpoints, offline
                # seeds, file-based CP0 starts): reseed the buffers (H-Z) and
                # take ONE settle step. Was 5 — a pure vestige of the
                # pre-notify_state_loaded flush that burned ~20 game frames.
                self.preprocessed.notify_state_loaded()
                for _ in range(1):
                    obs, _, _, _, _ = self.gym_env.step([0, 0, 0])
            self._start_state_hash = hashlib.blake2b(
                state_bytes, digest_size=8
            ).hexdigest()
        else:
            obs, _ = self.gym_env.reset()
            self._start_state_hash = ""

        # RANDOM NO-OP START (the standard Atari trick, applied where it can
        # actually reach us). The profile already has `random_noop_max`, but it
        # fires inside the STARTUP SEQUENCE — and this env calls gym reset once
        # (`if not self._initialized`) then `load_state`s every episode, so that
        # jitter is overwritten by the first load and never affects L2/L3/L4.
        #
        # Why we want it. Measured on v4's pools, the bonus countdown at capture
        # (a monotonic clock, so a proxy for arrival time) is almost constant:
        # Lfruit_top spread 2 over 100 captures, Fr2 18, Step 103 with median 751
        # against max 753. The policy replays ONE open-loop trajectory with
        # near-identical timing, so it always meets the periodic kangaroos at the
        # same phase and never has to READ them. That also makes every pool
        # phase-poor by construction: all 20 sampled Step seeds need a pause of
        # 16-24 steps, only 3 distinct values, so an agent can pass them by
        # memorising "wait 20" without learning to react.
        #
        # `noop_start_max` draws 0..N extra no-op gym steps after the start state
        # is in place, decorrelating arrival phase from the route.
        # `noop_start_scope`: "reset" jitters only reset-origin episodes (keeps
        # seeds representing exactly the situation they were captured for), "all"
        # jitters seeds too (diversifies the pools, at the cost of changing what
        # a seed means). Untested either way — the knob exists so both can be run.
        if self._noop_start_max > 0 and (
            self._noop_start_scope == "all" or self._start_wp is None
        ):
            for _ in range(random.randint(0, self._noop_start_max)):
                obs, _, _done, _trunc, _ = self.gym_env.step([0, 0, 0])
                if _done or _trunc:
                    break

        self._step_count = 0
        self._prev_fruits = self.iface.read_ram_byte(FRUITS_ADDR)
        self._start_fruits = self._prev_fruits
        self._prev_lives = self.iface.read_ram_byte(LIVES_ADDR)
        self._prev_bonus = self._read_bonus()
        self._start_bonus = self._prev_bonus
        self._prev_score = self._read_score()
        self._start_score = self._prev_score
        self._start_xy = self._read_pos()
        self._stall = 0
        self._episode_reward = 0.0
        self._fruits_collected_this_ep = 0
        self._prev_princess_flag = self.iface.read_ram_byte(PRINCESS_FLAG_ADDR)
        # Deferred checkpoint scoring: snapshots taken this episode,
        # each (fruits_collected_level, state_bytes, save_step). They
        # are scored and admitted to the pool at episode end based on
        # how the rest of the episode played out (real survival /
        # reached-next), not a passive probe.
        self._pending_saves = []
        # Deferred WAYPOINT captures, scored at episode end by the same
        # play-based survival gate as fruit checkpoints (H-AI + survival gate):
        # each entry is (wp_id, state_bytes, save_step, bonus, stack). We no
        # longer admit a WP the instant the agent is grounded within tol —
        # that seeds doomed states (the 6 dying-fall Lesc_bot captures). We
        # keep the capture here and judge it by how the rest of the episode
        # actually unfolds.
        self._pending_wp_saves = []
        # Pose observed at each gym step of this episode, indexed by step. Lets the
        # admission gate ask "what was the agent doing when its survival window closed?"
        # -- which is how a state that falls off a platform and is CAUGHT by a
        # trampoline
        # gets rejected despite staying alive. See _admit_by_play.
        self._pose_log = []
        # Sprite-centre px per gym step, same indexing as _pose_log.
        self._x_log = []
        # A checkpoint snapshot deferred to the next grounded frame:
        # (collected_total, save_step) or None. Prefers a state the agent can
        # steer from immediately. NOT because a mid-jump state reloads into a
        # fall — that claim is false, see SURFACE_POSES above.
        self._grounded_snap_due = None
        # Highest checkpoint level reached this episode, in CP-level
        # units (0..fruits_total fruits collected; princess touch counts
        # as fruits_total+1). Start level = fruits_total - fruits_remaining.
        # (H-O fix: was initialized to self._start_fruits — fruits
        # *remaining*, the wrong unit — which pinned reset episodes at the
        # top and over-admitted their seed snapshots via reached_next.)
        # Progress rung this episode STARTS on (a seed carries its reached set,
        # so a mid-route seed starts partway up the ladder). Cached because the
        # start state cannot change mid-episode.
        self._n_rungs = _manager.N_RUNGS
        self._start_rung = self._current_rung()
        self._max_cp_this_ep = self._start_rung
        # Clean per-episode princess-touch flag, credited to
        # record_episode for CP4->princess success.
        self._princess_touched_this_ep = False
        self._episode_id = _next_episode_id()

        return self._wrap_obs(obs), {}

    def step(self, action):
        assert _manager is not None
        obs, _, done, truncated, info = self.gym_env.step(action)
        self._step_count += 1

        fruits = self.iface.read_ram_byte(FRUITS_ADDR)
        lives = self.iface.read_ram_byte(LIVES_ADDR)
        bonus = self._read_bonus()
        score = self._read_score()
        x = self.iface.read_ram_byte(X_POS)
        y = self.iface.read_ram_byte(Y_POS)
        fruits_present = tuple(
            self.iface.read_ram_byte(self._fruit_addrs[i]) != 0 for i in self._fruit_ids
        )
        princess_flag = self.iface.read_ram_byte(PRINCESS_FLAG_ADDR)
        princess_touched = princess_flag == 1 and self._prev_princess_flag == 0

        # Fast, cause-agnostic death (0x2AFC). On L2 this is the ONLY reliable
        # death signal (lives byte inert). Read once; used both for the reward
        # (ctx.died -> grounded reward suppresses shaping on the fatal step)
        # and for termination/labeling below.
        died = self._use_death_flag and yeti.is_dead(self.iface)

        ctx = RewardContext(
            prev_fruits=self._prev_fruits,
            curr_fruits=fruits,
            prev_bonus=self._prev_bonus,
            curr_bonus=bonus,
            prev_score=self._prev_score,
            curr_score=score,
            prev_lives=self._prev_lives,
            curr_lives=lives,
            step_count=self._step_count,
            curr_y=y,
            curr_x=x,
            fruits_present=fruits_present,
            princess_touched=princess_touched,
            pose=self.iface.read_ram_byte(POSE_ADDR),
            died=died,
        )
        reward = float(self._reward_fn(ctx))

        # Census every pose the run observes, so an uncatalogued one shows up in the
        # status line instead of silently failing every pose gate. See
        # CheckpointManager.pose_seen and yeti.unknown_poses.
        _manager.pose_seen[int(ctx.pose)] += 1
        # Indexed by gym step, so the admission gate can look up the pose at
        # save_step + min_survival_steps. See _admit_by_play.
        self._pose_log.append(int(ctx.pose))
        self._x_log.append(int(ctx.curr_x) * 4 + 8)

        # Snapshot on fruit collection. Scoring is deferred to episode
        # end (see _pending_saves): we judge the state by how the rest
        # of the real episode unfolds, not a passive probe. The snapshot
        # itself is deferred to the next GROUNDED frame (pose in
        # SURFACE_POSES), which prefers a seed the agent can steer from on its
        # first step. A mid-jump state does NOT reload into a fall; see the
        # measurement recorded at SURFACE_POSES above.
        if fruits < self._prev_fruits:
            self._fruits_collected_this_ep += self._prev_fruits - fruits
            collected_total = self._current_rung(ctx.fruits_present)
            self._max_cp_this_ep = max(self._max_cp_this_ep, collected_total)
            # SEED THE FRUIT AS A WAYPOINT. A fruit is a route target like any
            # other -- same Target class, same `mandatory` and `seedable` flags,
            # counted by the same `rung_of`. Only its TRIGGER differs: the game's
            # presence byte decides it was collected, where a waypoint is decided by
            # sprite overlap. Nothing about seeding should follow from that, and it
            # used to: fruits captured into a pool keyed by RUNG NUMBER while
            # waypoints captured into pools keyed by NAME.
            #
            # What that cost, visible on every L4 status line: `cp` printed 13 slots
            # and only ever filled ONE. A rung-keyed capture can only happen at a
            # fruit pickup, L4 has one fruit, so one slot. `Lclimb3_top` is mandatory
            # and is rung 9, but its states went to a name-keyed pool, so slot 9 read
            # zero forever. And only 1880 of 25522 episodes in v23 ever started from
            # the rung pool, against 15438 from named pools -- so the structure was
            # near-vestigial as well as misleading.
            #
            # Now the fruit joins the same deferred path: remember it, save at the next
            # grounded frame, admit on survival, into a pool named for the fruit.
            for _i, _present in enumerate(ctx.fruits_present or (), 1):
                if not _present and f"F{_i}" not in self._captured_wps:
                    self._wp_snap_due.add(f"F{_i}")

        # Waypoint capture (H-AI): snapshot a GROUNDED state when the agent
        # is within tolerance of a computed waypoint position, at most once
        # per waypoint per episode, and never the waypoint the episode was
        # seeded from. Real states only (grounded gate); these feed the
        # optional, non-gating WP start-pools.
        # DETECTION and CAPTURE are deliberately SEPARATE decisions here.
        #
        # They used to share one pose gate and one geometry test, which meant widening
        # detection also unpinned the seed pools. Those are different risks: detection
        # feeds the reach metric and (via the reward's own copy of the test) milestone
        # marking, while capture decides what states a pool is built from. Splitting
        # them lets `waypoint_reach_mode` be evaluated without touching the pools.
        #
        #   detection -- geometry from `waypoint_reach_mode`; pose gate is the grounded
        #                allowlist in "box" mode (byte-identical to before) and the
        #                fail-open blocklist in "sprite" mode.
        #   capture   -- grounded + the SAME geometry as detection (2026-09-11). It used
        #                to be pinned to the box test unconditionally, on the reasoning
        #                that unpinning the pools was a separate lever. The consequence
        #                was that the flag's 3-seed A/B could not change pool
        #                composition and did not, so the defect it targeted -- boxes
        #                that overlap where the agent stands, and launch pads whose
        #                tol-6 box fires on the PREDECESSOR's seed 16-24 px away -- was
        #                never actually under test. Capture keeps the GROUNDED gate
        #                either way: a seed has to be a state the agent can act from.
        if self._wp_enabled:
            _grounded = ctx.pose in SEED_POSES
            _pose_ok_detect = (
                _grounded
                if self._wp_reach_mode == "box"
                else ctx.pose not in yeti.NON_TRAVERSAL_POSES
            )
            for wp_id, (wx, wy, _floor) in self._waypoints.items():
                # Record closest GROUNDED approach for EVERY waypoint (even the
                # seed WP / already-captured ones) so the display shows whether
                # the agent reaches each waypoint's vicinity at all. Stays
                # grounded-only whatever the reach mode, so the number keeps
                # meaning the same thing across runs.
                if _grounded:
                    _manager.note_wp_distance(wp_id, max(abs(x - wx), abs(y - wy)))
                _tol = self._wp_tol_of.get(wp_id, self._wp_tol)
                # Shared reach test (retro_ai.training.targets.reaches) -- the SAME
                # comparison AND mode the REWARD uses for milestone marking, so the two
                # cannot drift apart. In "box" mode the tolerance VALUES still differ
                # (this passes 2 for ladders / 6 for jumps, the reward always passes 2),
                # the known divergence documented on within_tol; "sprite" mode ignores
                # tolerance entirely and so removes it.
                detected = _pose_ok_detect and reaches(
                    (wx, wy), x, y, _tol, mode=self._wp_reach_mode
                )
                # Record EVERY reach (incl. the seed WP / already-captured) for
                # the reset-origin wp_reach_ema; capture below is more selective.
                if detected:
                    self._reached_wps_this_ep.add(wp_id)
                # `seed_waypoint_skip` bites HERE and only here: detection and the
                # reach/approach bookkeeping above still run, so the waypoint stays
                # visible in the route table and usable as a predecessor gate; it just
                # never enters a pool, and therefore never starts an episode.
                if (
                    wp_id == self._start_wp
                    or wp_id in self._captured_wps
                    or wp_id in self._wp_seed_skip
                ):
                    continue
                # CAPTURE, deferred to the next GROUNDED frame -- the same rule the
                # fruit path has always used, and the reason it has always worked.
                #
                # THE BUG THIS FIXES. Capture used to require grounded AND in-reach in
                # the SAME frame. A waypoint touched in mid-air was therefore never
                # captured unless the agent also landed within sprite range of the
                # anchor. Measured on L4 `Low2`, the rope-2 landing: its anchor is
                # px 128 and floor 13's extent is [0..128), so the anchor is OFF the
                # platform and NO standable position reaches it -- the nearest is
                # px 120, 8 px away, and the sprite reaches 6 px to the right. So a
                # crossing registers `Low2` mid-flight (a 5-frame window at px
                # 124-132) and then lands a few pixels past it, and nothing is ever
                # saved. v4 banked ONE `Low2` state in the project's history; v13
                # grew that to 11 by re-capturing from it.
                #
                # `Rope1` escapes only by luck: its anchor is px 108 and floor 7
                # starts at px 104, so it sits 4 px inside and exactly one standable
                # spot (px 112) reaches it.
                #
                # Deferring removes the coincidence. The grounded rule itself stays
                # -- a mid-jump seed hands the agent a committed trajectory -- but it
                # now says WHEN to save, not WHETHER, as it does for fruits.
                if detected:
                    self._wp_snap_due.add(wp_id)

        # ONE CAPTURE PATH, for fruits and waypoints alike. Anything detected this
        # episode and still awaiting a surface is saved here, from the SAME state.
        #
        # Deliberately outside the `_wp_enabled` block: a fruit is seeded through this
        # path too, and L1 runs with `curriculum.waypoints` off, so gating this on
        # waypoints being enabled would silently stop capturing fruits there.
        #
        # A detection on a grounded frame reaches here in the same step, so grounded
        # captures behave exactly as before; an airborne one waits for the landing. If
        # the agent dies before grounding, nothing is saved -- which is the rule the
        # fruit path has always had.
        if ctx.pose in SEED_POSES and self._wp_snap_due:
            _state = self.base._interface.save_state()
            # Frame stack at the SAME moment as the save-state, so the seed restores
            # the real motion history on load (H-AB).
            _stack = self.preprocessed.export_frame_stack()
            # Milestones banked AS OF THIS MOMENT (inherited from this episode's own
            # seed + reached since) — transitive, so the set stays complete along a
            # reverse-curriculum chain.
            _reached = self._inherited_wps | self._reached_wps_this_ep
            for _tid in sorted(self._wp_snap_due):
                self._pending_wp_saves.append(
                    (_tid, _state, self._step_count, bonus, _stack, _reached)
                )
                self._captured_wps.add(_tid)
            self._wp_snap_due.clear()

        # Princess touch ends the episode and counts as a success.
        if princess_touched:
            self._fruits_collected_this_ep += 1
            # Princess is the terminal "checkpoint" (level fruits_total+1);
            # any pending fruit snapshot in this episode therefore reached
            # the next checkpoint.
            # Princess = the TERMINAL rung (top of the progress ladder).
            self._max_cp_this_ep = max(self._max_cp_this_ep, self._n_rungs + 1)
            self._princess_touched_this_ep = True

        self._prev_fruits = fruits
        self._prev_score = score
        self._prev_princess_flag = princess_flag
        self._episode_reward += reward

        # Termination
        end_reason = None
        # Death: on L2 via the fast 0x2AFC flag (lives byte is inert there, so
        # the lives check below can never fire — it stays for L1, where 0x2AFC
        # detection is intentionally off to preserve L1's termination timing).
        if died:
            done = True
            end_reason = "death"
        elif lives < self._prev_lives and self._prev_lives > 0:
            done = True
            end_reason = "death"
        self._prev_lives = lives

        if bonus == self._prev_bonus:
            self._stall += 1
        else:
            self._stall = 0
            self._prev_bonus = bonus
        if self._stall >= self.stall_threshold:
            done = True
            if end_reason is None:
                end_reason = "stall"

        if self._step_count >= self.max_steps:
            truncated = True
            if end_reason is None:
                end_reason = "max_steps"

        # Princess touch ends the segment.
        if princess_touched:
            done = True
            if end_reason is None:
                end_reason = "princess_touched"

        if done or truncated:
            # Start/end are PROGRESS RUNGS (how many mandatory targets are
            # done). On L1/L2 a rung is a collected fruit, so this is the old
            # fruit-count value; on L3 the ladder is the milestone chain, which
            # is what makes "reached the next rung" a usable signal there.
            start_level = self._start_rung
            # H-M fix: a princess touch is the terminal rung (N_RUNGS + 1), so
            # the last step -> princess registers as a segment success. Without
            # it reached_level caps at the top rung and the final segment's
            # success is never recorded, pinning its curriculum weight at max.
            reached_level = (
                self._n_rungs + 1
                if self._princess_touched_this_ep
                else max(self._current_rung(), self._start_rung)
            )
            # Credit the goal-score to the actual start: a WP id (str) for
            # a waypoint-seeded episode, else the CP start level (int).
            start_key = self._start_wp if self._start_wp is not None else start_level
            # Pass the reached-WP set + the full WP universe so the manager can
            # update the reset-origin wp_reach_ema (only used when start_key==0).
            # Order-free segment health: did this episode reach any route point
            # it did NOT start from or inherit from its seed? (Inherited points
            # are subtracted so re-touching a milestone the seed already banked
            # does not count as progress.)
            new_points = self._reached_wps_this_ep - self._inherited_wps
            if self._start_wp is not None:
                new_points = new_points - {self._start_wp}
            progressed = bool(new_points) or reached_level > start_level
            # THE REACH UNIVERSE MUST COVER EVERY POOL THE START GATE JUDGES.
            #
            # `wp_reach_ema` is what `_wp_eligible` consults to decide whether a pool
            # may be a start state. It used to be built from `self._waypoints` -- the
            # POSITIONAL DETECTOR universe -- while the gate judges the START-POOL
            # universe. Those are different sets, and any pool in the second but not the
            # first could never get a row, so it read 0.0 reach, failed the gate, fell
            # through to the `route_order` predecessor rule, was absent from THAT too,
            # and was refused forever.
            #
            # Measured on L4, where the two sets differ by exactly `F1` (the fruit pool,
            # introduced when fruit seeds moved out of the rung-keyed checkpoint slots):
            # v24 started 925 episodes from those seeds via the rung gate (3.8% of
            # 24121); v26 started 0 of 28137 from the identical seeds, with the pool
            # holding 100 of them and 10439 captures. Every downstream waypoint lost
            # share to the fruit chain and per-waypoint from-reset reach fell a flat
            # 0.08-0.12 across the route.
            #
            # Fruits are not special here. The rule is that the reach table is keyed by
            # what is gated, so a pool named anything other than a positional waypoint
            # is tracked automatically instead of being silently frozen out.
            #
            # `_reached_targets` already folds collected fruits in as F1/F2/...; passing
            # `_reached_wps_this_ep` instead withheld the evidence, so even a row would
            # have decayed to 0. The union keeps never-reached DETECTORS in the table
            # so they still decay toward 0 -- the block signal the log wants to show.
            # `_reached_targets` only folds fruits in when handed the presence bytes, so
            # read them here exactly as `_current_rung` does a few lines above.
            _fp = tuple(
                self.iface.read_ram_byte(self._fruit_addrs[i]) != 0
                for i in self._fruit_ids
            )
            _pool_ids = set(getattr(_manager, "waypoints", {}) or {})
            _reach_universe = set(self._waypoints.keys()) | _pool_ids
            _manager.record_episode(
                start_key,
                reached_level,
                reached_wps=self._reached_targets(_fp),
                all_wps=_reach_universe if self._wp_enabled else None,
                progressed=progressed,
                start_rung=self._start_rung,
            )
            # Flush deferred checkpoint snapshots, scored by how the
            # rest of this episode actually played out. ``source_cp``
            # is the CP this episode started from — the retention key
            # that biases pools toward reset-origin states (approach 30).
            # A WP-seeded episode's fruit snapshots are NOT reset-origin, so
            # mark them maximally artificial (evict-first).
            save_src = self._n_rungs if self._start_wp is not None else start_level
            for (
                level,
                state_bytes,
                save_step,
                save_bonus,
                save_stack,
                save_reached,
            ) in self._pending_saves:
                survived_steps = self._step_count - save_step
                reached_next = self._max_cp_this_ep > level
                _manager.save_scored(
                    level,
                    state_bytes,
                    survived_steps,
                    reached_next,
                    save_bonus,
                    end_pose=self._pose_at_window_end(save_step),
                    source_cp=save_src,
                    stack=save_stack,
                    reached=save_reached,
                )
            self._pending_saves = []
            # Flush deferred WAYPOINT captures through the SAME play-based
            # survival gate. survived_steps = steps alive after the capture;
            # reached_next = the episode made forward CP progress past its
            # start. A doomed capture (agent died shortly after) is rejected as
            # precarious, so dying-fall states never seed a WP pool.
            for (
                wp_id,
                wp_state,
                wp_step,
                wp_bonus,
                wp_stack,
                wp_reached,
            ) in self._pending_wp_saves:
                wp_survived = self._step_count - wp_step
                wp_reached_next = self._max_cp_this_ep > start_level
                _wp_end_pose = self._pose_at_window_end(wp_step)
                _manager.save_waypoint(
                    wp_id,
                    wp_state,
                    wp_survived,
                    wp_reached_next,
                    end_pose=_wp_end_pose,
                    source_cp=save_src,
                    bonus=wp_bonus,
                    stack=wp_stack,
                    reached=wp_reached,
                )
            self._pending_wp_saves = []
            if end_reason is None:
                end_reason = "env_done" if done else "env_truncated"
            self._log_episode(end_reason, fruits, bonus, score)

        return self._wrap_obs(obs), reward, done, truncated, info

    def _pose_at_window_end(self, save_step):
        """Pose when this capture's survival window closed, or None if unknown.

        The window is ``min_survival_steps`` gym steps after the capture. If the episode
        ended first the caller's ``survived_steps`` test already rejects the snapshot,
        so
        None is returned and the gate falls through to its step-count check.
        """
        want = int(save_step) + int(_manager.min_survival_steps)
        if 0 <= want < len(self._pose_log):
            return self._pose_log[want]
        return None

    def _reached_targets(self, fruits_present=None) -> set:
        """Every target reached so far this episode, INCLUDING what the seed
        already had banked.

        Waypoints come from positional detection; fruits are derived from their
        presence bytes (a collected fruit is gone from the level, so unlike a
        waypoint this survives a save-state and needs no bookkeeping).
        """
        out = set(self._inherited_wps) | set(self._reached_wps_this_ep)
        if fruits_present is not None:
            out |= {
                f"F{i}" for i, present in enumerate(fruits_present, 1) if not present
            }
        return out

    def _current_rung(self, fruits_present=None) -> int:
        """The progress rung: how many MANDATORY targets are done."""
        if fruits_present is None:
            fruits_present = tuple(
                self.iface.read_ram_byte(self._fruit_addrs[i]) != 0
                for i in self._fruit_ids
            )
        return _manager.rung_of(self._reached_targets(fruits_present))

    def _log_episode(
        self, end_reason: str, fruits: int, bonus: int, final_score: int
    ) -> None:
        if self.episode_logger is None:
            return
        final_xy = self._read_pos()
        # Same rung values record_episode uses, so episodes.csv and the
        # curriculum agree (including the terminal princess rung, which the old
        # fruit-count version could never log).
        start_level = self._start_rung
        reached_level = (
            self._n_rungs + 1
            if self._princess_touched_this_ep
            else max(self._current_rung(), self._start_rung)
        )
        self.episode_logger.log(
            global_step=_get_global_step(),
            env_id=self.env_id,
            episode_id=self._episode_id,
            start_level=start_level,
            reached_level=reached_level,
            n_fruits_collected=self._fruits_collected_this_ep,
            length=self._step_count,
            total_reward=round(self._episode_reward, 4),
            end_reason=end_reason,
            start_x=self._start_xy[0],
            start_y=self._start_xy[1],
            start_score=self._start_score,
            start_bonus=self._start_bonus,
            final_x=final_xy[0],
            final_y=final_xy[1],
            final_score=final_score,
            final_bonus=bonus,
            start_state_hash=self._start_state_hash,
            # TRUE start source, so from-reset analysis of episodes.csv is
            # possible: a WP id for waypoint-seeded episodes, else the CP level
            # ("0" = real game reset). start_level cannot distinguish them.
            start_key=(self._start_wp if self._start_wp is not None else start_level),
            # Column index of the start x reached matrix (see EPISODE_COLUMNS):
            # every route point this episode reached.
            reached_points=";".join(sorted(self._reached_wps_this_ep)),
        )


def _level_of(cfg) -> int:
    """The level the reward (and therefore the target set) is configured for."""
    if cfg.reward is not None and cfg.reward.params:
        try:
            return int(cfg.reward.params.get("level", 1))
        except (TypeError, ValueError):
            return 1
    return 1


def _progress_ladder(cfg):
    """``(mandatory_groups, n_rungs)`` — the progress ladder for this level.

    A rung is one STEP of the route, and a step is satisfied by ANY member of a
    group. Most groups hold one target; a group holds several when the level
    offers alternative ways to make the same step, plus the graph alias a jump
    landing carries (the curriculum calls it "A1", the nav graph calls the same
    point "J10_11_b", and a seed may record either).

    WHY GROUPS AND NOT A FLAT ID SET. Counting ids double-counted, two ways, and
    both were live on L4:

        reached Low2 (low route to floor 13)          -> rung 11   correct
        reached Lhi_down_bot (high route, same floor) -> rung 11   correct
        reached BOTH                                  -> rung 12   WRONG
        reached Low2 and its own alias J12_13_b       -> rung 12   WRONG

    `Lhi_down_bot` sits at px 104 and `Low2` landings at px 88-120, so an agent
    that crosses rope 2 and walks a few pixels left registers both and gains two
    rungs for one step. The rung COUNT was already protected against this (it
    was passed in rather than derived from the id set); the function that assigns
    a state TO a rung was not.

    The grouping is the level map's own `reward_waypoints`, so the curriculum and
    the reward cannot disagree about what one step is. Targets the reward does
    not group -- the fruits, which are paid by the fruit term -- each become a
    group of one.

    The implementation now lives in `retro_ai.training.targets.progress_ladder`, so
    the eval rollout counts rungs the same way. It used to be defined only here,
    and the rollout counted ids: when an agent first crossed rope 2, every princess
    episode in eval read 13 of 13 while this ladder has 12 rungs.
    """
    from retro_ai.training.targets import progress_ladder

    return progress_ladder(_level_of(cfg))


def _route_order_for(cfg) -> list:
    """The level's DISPLAY-ONLY route order for the log table (never semantics).

    Empty when the level defines none (L1/L2) — route_table() then falls back to
    a stable arbitrary order, so no point is ever hidden.
    """
    from retro_ai.training.yeti_map import get_level_map

    level = 1
    if cfg.reward is not None and cfg.reward.params:
        try:
            level = int(cfg.reward.params.get("level", 1))
        except (TypeError, ValueError):
            level = 1
    try:
        return list(getattr(get_level_map(level), "route_order", None) or [])
    except ValueError:
        return []


class PoolSaveCallback(BaseCallback):
    """Persist the curriculum seed pools (checkpoints.pkl) PERIODICALLY.

    Previously the pools were only written when training completed normally
    (not on interrupt/kill), so stopping a long run early — even to warm-start
    the next phase — threw away every captured seed (e.g. the L3 A1..A5 ascent
    pools). This saves them every ``save_freq`` env-steps (atomic via
    save_to_disk's temp+replace), so a run can be cut at any time and the next
    phase can reuse the latest pools. Pure I/O; does not affect training.
    """

    def __init__(self, path: str, save_freq: int):
        super().__init__()
        self._path = path
        self._save_freq = max(1, int(save_freq))
        self._last_save = 0

    def _on_step(self) -> bool:
        if _manager is None:
            return True
        if self.num_timesteps - self._last_save >= self._save_freq:
            self._last_save = self.num_timesteps
            _manager.save_to_disk(self._path)
        return True


class NoOpCallback(BaseCallback):
    """Placeholder so the callback list keeps a fixed shape when a feature is off."""

    def _on_step(self) -> bool:
        return True


class RegressionStopCallback(BaseCallback):
    """End the run when the parallel evaluator says the policy has got worse.

    Reads `<output>/best/eval_status.json`, which keep_best_sweep.py --watch rewrites
    after every snapshot eval. Acts only on `consecutive_regressions`, so one dip is
    ignored -- a single eval at n=30 can false-flag, three in a row does not.

    WHY A FILE AND NOT AN IN-PROCESS EVAL. The Crayon emulator keeps in-process global
    state, so evaluation must not share a process with training; keep_best_sweep already
    shells out per snapshot for that reason. A file is also the honest boundary: the
    evaluator runs whether or not training is watching, and training degrades to a no-op
    if the evaluator is not running.

    DELIBERATELY ONLY STOPS. Reverting weights is the next increment and needs this
    as its control arm -- see experiments/003-yeti-training.md step 4b for why
    reverting ALONE is measured not to work (control arm A0: a champion at mean depth
    9.55 put back into training read 1.20 / 2.43 / 7.53 / 1.03 over the next 1M).
    """

    def __init__(self, status_path, patience, check_freq=10_000, verbose=0):
        super().__init__(verbose)
        self.status_path = status_path
        self.patience = int(patience)
        self.check_freq = int(check_freq)
        self._last_check = 0
        self._last_seen_step = None

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_check < self.check_freq:
            return True
        self._last_check = self.num_timesteps
        try:
            with open(self.status_path) as fh:
                st = json.load(fh)
        except (OSError, ValueError):
            return True  # evaluator not running, or mid-write: no-op
        n = int(st.get("consecutive_regressions") or 0)
        step = st.get("step")
        if step != self._last_seen_step:
            self._last_seen_step = step
            if n:
                print(
                    f"  [regression] eval @ {step}: {n} consecutive "
                    f"(patience {self.patience}) "
                    f"frontier={st.get('frontier')}@"
                    f"{(st.get('frontier_rate') or 0):.2f} "
                    f"best={st.get('best_frontier')}@"
                    f"{(st.get('best_frontier_rate') or 0):.2f}",
                    flush=True,
                )
        if n >= self.patience:
            print(
                f"\nSTOPPING: {n} consecutive regressing evals "
                f"(>= patience {self.patience}). "
                f"Best was step {st.get('best_step')} at "
                f"{st.get('best_frontier')}@"
                f"{(st.get('best_frontier_rate') or 0):.2f}, kept at "
                f"{st.get('best_model')}.",
                flush=True,
            )
            return False
        return True


class CurriculumCallback(BaseCallback):
    """Log curriculum progress during training."""

    def __init__(
        self,
        total_timesteps: int,
        log_interval: int = 5000,
        diag_path=None,
        n_rungs: int = 4,
        # The route table is the informative view -- one row per route point with
        # its from-reset reach -- so it is what you read to see how far the agent
        # gets and where it stops. At 500k it appeared ~30 times in a 15M run,
        # which is useless for watching. The step lines carry less information, so
        # print the table often instead.
        table_interval: int = 50_000,
        route_order=None,
    ):
        super().__init__()
        self._diag_path = diag_path
        self._diag_file = None
        self._adm_file = None
        # One counter per progress rung, plus rung 0 (the game start).
        self._n_rungs = int(n_rungs)
        n = self._n_rungs + 1
        self._last_starts = [0] * n
        # H-R instrumentation: last cumulative admission counters, for
        # per-interval deltas.
        self._last_adm_reached = [0] * n
        self._last_adm_survived = [0] * n
        self._last_rejected = [0] * n
        # First diag write captures current counters as the baseline so
        # interval deltas don't include history loaded from a resumed
        # checkpoints.pkl.
        self._baseline_captured = False
        self._total = total_timesteps
        self._log_interval = log_interval
        self._last_log = 0
        self._start = time.monotonic()
        # Route TABLE cadence: the per-step line stays compact/fixed-size; the
        # full route-ordered table (one row per point) prints far less often.
        self._table_interval = table_interval
        self._last_table = 0
        self._route_order = route_order or []

    def _on_step(self) -> bool:
        _set_global_step(self.num_timesteps)

        if self.num_timesteps - self._last_log >= self._log_interval:
            elapsed = time.monotonic() - self._start
            fps = self.num_timesteps / elapsed if elapsed > 0 else 0
            pct = 100 * self.num_timesteps / self._total

            infos = self.locals.get("infos", [])
            rewards = []
            for info in infos:
                ep = info.get("episode")
                if ep:
                    rewards.append(ep["r"])
            reward_str = f"{np.mean(rewards):.1f}" if rewards else "N/A"

            assert _manager is not None
            print(
                f"step {self.num_timesteps}/{self._total} ({pct:.0f}%) "
                f"| reward={reward_str} "
                f"| emu_fps={fps * 4:.0f} "
                f"| {_manager.summary()}",
                flush=True,
            )
            self._write_diag()
            self._last_log = self.num_timesteps

        # Route table: low frequency, route-ordered, one row per point.
        if (
            _manager is not None
            and self.num_timesteps - self._last_table >= self._table_interval
        ):
            table = _manager.route_table(self._route_order)
            if table:
                print(
                    f"\nROUTE @ step {self.num_timesteps} "
                    f"(reach=from-reset, prog=order-free progress)\n{table}\n",
                    flush=True,
                )
            self._last_table = self.num_timesteps
        return True

    def _write_diag(self) -> None:
        """Append a diagnostics row to correlate from-reset reach with the
        curriculum's start distribution and per-segment success (to chase
        the decay/oscillation mechanism). PPO-side metrics (kl, entropy,
        explained_variance) are in tensorboard."""
        if self._diag_path is None or _manager is None:
            return
        if not self._baseline_captured:
            # Resumed runs load lifetime counters from checkpoints.pkl;
            # seed the per-interval baselines from current values so the
            # first emitted deltas reflect only new activity.
            self._last_starts = list(_manager.stats["starts"])
            self._last_adm_reached = list(_manager.stats["admit_reached"])
            self._last_adm_survived = list(_manager.stats["admit_survived"])
            self._last_rejected = list(_manager.stats["rejected_precarious"])
            self._baseline_captured = True
        if self._diag_file is None:
            self._diag_file = open(self._diag_path, "w")
            n = self._n_rungs
            reach_cols = [f"reach{i}" for i in range(1, n + 1)] + ["reach_princess"]
            succ_cols = [f"succ_ema{i}" for i in range(1, n + 1)]
            sfrac_cols = [f"start_frac{i}" for i in range(0, n + 1)]
            gscore_cols = [f"gscore{i}" for i in range(0, n + 1)]
            header = ["step"] + reach_cols + succ_cols + sfrac_cols + gscore_cols
            self._diag_file.write(",".join(header) + "\n")
        n = self._n_rungs
        starts = _manager.stats["starts"]
        delta = [starts[i] - self._last_starts[i] for i in range(n + 1)]
        self._last_starts = list(starts)
        tot = sum(delta) or 1
        fr = [d / tot for d in delta]
        rr = _manager.reset_reach_ema
        se = _manager.seg_success_ema
        gs = [p.goal_score for p in _manager.checkpoints]
        # reach1..reach_N then princess (index N+1); succ_ema1..N;
        # start_frac0..N; gscore0..N.
        vals = [str(self.num_timesteps)]
        vals += [f"{rr[i]:.3f}" for i in range(1, n + 2)]
        vals += [f"{se[i]:.3f}" for i in range(1, n + 1)]
        vals += [f"{fr[i]:.3f}" for i in range(0, n + 1)]
        vals += [f"{gs[i]:.3f}" for i in range(0, n + 1)]
        self._diag_file.write(",".join(vals) + "\n")
        self._diag_file.flush()
        self._write_admission_diag()

    def _write_admission_diag(self) -> None:
        """Per-CP time series of the seed-admission filter (H-R): how many
        snapshots this interval were admitted via reaching the next CP
        (a_reach), admitted only on survival (a_surv), or rejected (rej);
        plus pool diversity (distinct = unique save-states; psize = pool
        size). Shows whether a stricter filter admits fewer CPs and/or
        collapses pool diversity over time."""
        if self._diag_path is None or _manager is None:
            return
        if self._adm_file is None:
            adm_path = os.path.join(
                os.path.dirname(self._diag_path), "admission_diag.csv"
            )
            self._adm_file = open(adm_path, "w")
            cols = ["step"]
            for cp in range(1, _manager.N_RUNGS + 1):
                cols += [
                    f"a_reach{cp}",
                    f"a_surv{cp}",
                    f"rej{cp}",
                    f"distinct{cp}",
                    f"psize{cp}",
                ]
            self._adm_file.write(",".join(cols) + "\n")
        ar = _manager.stats["admit_reached"]
        asv = _manager.stats["admit_survived"]
        rj = _manager.stats["rejected_precarious"]
        row = [str(self.num_timesteps)]
        for cp in range(1, _manager.N_RUNGS + 1):
            d_reach = ar[cp] - self._last_adm_reached[cp]
            d_surv = asv[cp] - self._last_adm_survived[cp]
            d_rej = rj[cp] - self._last_rejected[cp]
            pool = _manager.checkpoints[cp]
            distinct = len({e[2] for e in pool.states})
            row += [
                str(d_reach),
                str(d_surv),
                str(d_rej),
                str(distinct),
                str(len(pool)),
            ]
        self._last_adm_reached = list(ar)
        self._last_adm_survived = list(asv)
        self._last_rejected = list(rj)
        self._adm_file.write(",".join(row) + "\n")
        self._adm_file.flush()


def train(cfg: RunConfig, config_path: Optional[str] = None) -> None:
    global _manager

    if cfg.curriculum is None:
        raise ValueError(
            "train_checkpoint_curriculum.py requires a 'curriculum' "
            "section in the run config"
        )

    seed = seed_everything(cfg.training.seed)

    _ladder_groups, _ladder_rungs = _progress_ladder(cfg)
    _manager = CheckpointManager(
        max_states_per_checkpoint=cfg.curriculum.max_states_per_checkpoint,
        min_states_to_advance=cfg.curriculum.min_states_to_advance,
        reset_fraction=cfg.curriculum.reset_fraction,
        frontier_fraction=cfg.curriculum.frontier_fraction,
        earlier_fraction=cfg.curriculum.earlier_fraction,
        min_survival_steps=cfg.curriculum.min_survival_steps,
        reach_threshold=cfg.curriculum.reach_threshold,
        segment_floor=cfg.curriculum.segment_floor,
        # Progress ladder: pools keyed by how many MANDATORY targets are done.
        mandatory_groups=_ladder_groups,
        n_rungs=_ladder_rungs,
        gate_waypoints=cfg.curriculum.gate_waypoints,
        gate_waypoints_by_predecessor=cfg.curriculum.gate_waypoints_by_predecessor,
        split_mandatory=cfg.curriculum.split_mandatory_starts,
        earned_progress_score=cfg.curriculum.earned_progress_score,
        admit_requires_survival=cfg.curriculum.admit_requires_survival,
        admit_requires_grounded=cfg.curriculum.admit_requires_grounded,
    )
    # Travel order. Feeds the route table AND, under
    # `gate_waypoints_by_predecessor`, the start-eligibility rule -- so this is
    # semantic now, not display-only. Must be set before the first pick_start.
    _manager.route_order = _route_order_for(cfg)
    # Anchor provenance for waypoint pools. Must be set BEFORE any load_from_disk so
    # the resume path can compare a file's anchors against the ones in force now and
    # discard pools captured around a position that has since moved.
    _manager.waypoint_anchors = {
        w: (int(x), int(y)) for w, (x, y, _f) in yeti.waypoints(_level_of(cfg)).items()
    }

    print("Checkpoint Curriculum Training", flush=True)
    print(f"  Profile: {cfg.env.profile}", flush=True)
    print(f"  Timesteps: {cfg.training.timesteps}", flush=True)
    print(f"  Output: {cfg.training.output}", flush=True)
    print(f"  Seed: {seed}", flush=True)

    if cfg.curriculum.seed_archive:
        import pickle

        print(f"  Seeding from {cfg.curriculum.seed_archive}", flush=True)
        with open(cfg.curriculum.seed_archive, "rb") as f:
            archive = pickle.load(f)
        for cell_key, info in archive.items():
            # Support both archive formats:
            #   new: cell_key[2] is a frozenset of collected floor numbers
            #   old: cell_key[2] is an int fruits_remaining (0..4)
            if len(cell_key) < 3:
                continue
            v = cell_key[2]
            if isinstance(v, (frozenset, set, list, tuple)):
                fruits_collected = len(v)
            elif isinstance(v, int) and 0 <= v <= 4:
                fruits_collected = 4 - v
            else:
                continue
            if fruits_collected > 0:
                # Archive seeds are artificial (not reset-origin):
                # mark source_cp = level so fresh reset-origin states
                # evict them first.
                _manager.save_checkpoint(
                    fruits_collected,
                    info["state"],
                    source_cp=fruits_collected,
                )
        print(f"  Seeded: {_manager.summary()}", flush=True)

    os.makedirs(cfg.training.output, exist_ok=True)

    # Validate the reward name is registered (raises early if not).
    create_reward(cfg.reward.name, cfg.reward.params)

    # Persist full, resolved config.
    manifest_extras = cfg.to_dict()
    manifest_extras["resolved_seed"] = seed
    manifest_extras["script"] = "scripts/mo5/yeti/train_checkpoint_curriculum.py"
    manifest = RunManifest.capture(
        {"config_path": config_path},
        cfg.training.output,
        extras=manifest_extras,
    )
    episode_logger = EpisodeLogger(cfg.training.output)

    from retro_ai.wrappers.threaded_vec_env import ThreadedVecEnv

    def make_env(rank: int):
        def _init():
            # Each env gets its own reward_fn instance to avoid the
            # shared-state bug (approach 19/20): SB3 calls reset() on
            # one env while another is mid-episode, and a shared
            # stateful reward (e.g. floor_novelty / climb_novelty /
            # path_progress) would leak resets.
            #
            # Tie any shaping gamma to the agent's PPO gamma by default
            # (PBRS policy-invariance requires them equal). Harmless for
            # rewards that ignore the param.
            reward_params = dict(cfg.reward.params)
            reward_params.setdefault("gamma", cfg.ppo.gamma)
            # NOT setdefault: the curriculum value WINS, so the reward and the
            # curriculum can never disagree about the reach geometry. That exact
            # disagreement is the documented `Fr1` defect.
            reward_params["waypoint_reach_mode"] = cfg.curriculum.waypoint_reach_mode
            env_reward_fn = create_reward(cfg.reward.name, reward_params)
            env = CheckpointCurriculumEnv(
                cfg=cfg,
                reward_fn=env_reward_fn,
                env_id=rank,
                episode_logger=episode_logger,
            )
            return Monitor(env)

        return _init

    num_envs = cfg.training.num_envs
    vec_env = ThreadedVecEnv([make_env(i) for i in range(num_envs)])
    print(f"  Envs: {num_envs} threaded", flush=True)

    n_steps = cfg.ppo.n_steps
    if n_steps is None:
        n_steps = max(1, 128 // num_envs)

    policy = "MultiInputPolicy" if cfg.curriculum.multi_input_obs else "CnnPolicy"
    print(f"  Policy: {policy}", flush=True)

    model = PPO(
        policy,
        vec_env,
        learning_rate=cfg.ppo.learning_rate,
        batch_size=cfg.ppo.batch_size,
        n_steps=n_steps,
        n_epochs=cfg.ppo.n_epochs,
        ent_coef=cfg.ppo.ent_coef,
        clip_range=cfg.ppo.clip_range,
        gamma=cfg.ppo.gamma,
        gae_lambda=cfg.ppo.gae_lambda,
        target_kl=cfg.ppo.target_kl,
        verbose=0,
        tensorboard_log=os.path.join(cfg.training.output, "tb"),
        device="auto",
        seed=seed,
    )

    if cfg.training.resume:
        print(f"  Resuming from {cfg.training.resume}", flush=True)
        if getattr(cfg.training, "warmstart_weights_only", False):
            # Phase-2 anneal: keep the freshly-built model (this run's new
            # PPO hyperparameters — n_steps, target_kl, ...) and copy ONLY
            # the network weights from the checkpoint. A full PPO.load
            # would restore the checkpoint's old hyperparameters and
            # silently ignore the new ones.
            src = PPO.load(cfg.training.resume, device="auto")
            model.policy.load_state_dict(src.policy.state_dict())
            del src
            print(
                "  (weights-only warm-start; new PPO hyperparameters kept)", flush=True
            )
        else:
            model = PPO.load(
                cfg.training.resume,
                env=vec_env,
                tensorboard_log=os.path.join(cfg.training.output, "tb"),
            )
        # Pools default to sitting beside the weights, but a champion lives in
        # <run>/best/ while its pools stay at <run>/checkpoints.pkl -- so warm-starting
        # from a champion inherits nothing unless the path is given explicitly.
        # Missing pools are not an error (a cold pool file is legitimate), so this
        # would fail silently; say something instead.
        ckpt_path = cfg.training.resume_pools or os.path.join(
            os.path.dirname(cfg.training.resume), "checkpoints.pkl"
        )
        if not os.path.exists(ckpt_path):
            print(
                f"  NOTE no pool file at {ckpt_path} — starting with EMPTY pools. "
                "If you meant to inherit them, set training.resume_pools.",
                flush=True,
            )
        _manager.load_from_disk(ckpt_path)
        _manager.load_from_disk(os.path.join(cfg.training.output, "checkpoints.pkl"))
        # DROP pools for waypoints this level no longer defines. A resumed pkl can
        # carry pools for waypoints that have since been removed from the map (L4 v6
        # dropped Spring_launch / Step_launch / Low1_launch as redundant). Their states
        # are real, but the waypoint is no longer detected, so its reach EMA stays 0:
        # under `gate_waypoints` it is silently gated out and merely clutters the route
        # table with an all-dashes row, and with the gate off it would be sampled as an
        # untracked start source. Neither is wanted -- if a waypoint was deleted, its
        # pool should go with it.
        _known_wps = set(yeti.waypoints(_level_of(cfg)))
        _stale = [w for w in _manager.waypoints if w not in _known_wps]
        for w in _stale:
            _manager.waypoints.pop(w, None)
        if _stale:
            print(
                f"  Dropped {len(_stale)} inherited pool(s) for waypoints this level "
                f"no longer defines: {sorted(_stale)}",
                flush=True,
            )

    print("\nTraining...", flush=True)
    status = "COMPLETED"
    exit_code: Optional[int] = 0
    # Periodic model snapshots so a long run that degenerates late can be
    # recovered from its best earlier checkpoint. Pure observability —
    # does not affect training. Saves roughly every 2M timesteps.
    snapshot_freq = max(
        1, (cfg.training.snapshot_freq_steps or 2_000_000) // max(1, num_envs)
    )
    snapshot_cb = CheckpointCallback(
        save_freq=snapshot_freq,
        save_path=os.path.join(cfg.training.output, "snapshots"),
        name_prefix="model",
    )
    try:
        model.learn(
            total_timesteps=cfg.training.timesteps,
            callback=[
                CurriculumCallback(
                    cfg.training.timesteps,
                    diag_path=os.path.join(cfg.training.output, "curriculum_diag.csv"),
                    n_rungs=_ladder_rungs,
                    # Display-only ordering for the route table (no semantics).
                    route_order=_route_order_for(cfg),
                    table_interval=int(
                        getattr(cfg.training, "route_table_freq_steps", 0) or 50_000
                    ),
                ),
                EpisodeMetricsCallback(episode_logger, log_interval=10_000),
                snapshot_cb,
                # End the run when the parallel evaluator reports the policy has got
                # worse. A no-op unless training.on_regression is set AND
                # keep_best_sweep.py --watch is running against this run.
                (
                    RegressionStopCallback(
                        os.path.join(cfg.training.output, "best", "eval_status.json"),
                        patience=getattr(cfg.training, "regression_patience", 3),
                    )
                    if str(getattr(cfg.training, "on_regression", "off")) == "stop"
                    else NoOpCallback()
                ),
                # Persist pools periodically so an early stop keeps captured
                # seeds (pools were previously saved only on normal completion).
                PoolSaveCallback(
                    os.path.join(cfg.training.output, "checkpoints.pkl"),
                    save_freq=1_000_000,
                ),
            ],
        )
        model.save(os.path.join(cfg.training.output, "final_model"))
        _manager.save_to_disk(os.path.join(cfg.training.output, "checkpoints.pkl"))
        print(f"\nSaved model to {cfg.training.output}/final_model.zip", flush=True)
        print(f"Final: {_manager.summary()}", flush=True)
    except Exception:
        status = "FAILED"
        exit_code = 1
        raise
    finally:
        episode_logger.close()
        manifest.finalize(status=status, exit_code=exit_code)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config",
        required=True,
        help="Path to run-config YAML (must include a 'curriculum' section).",
    )
    args = parser.parse_args()
    cfg = RunConfig.from_yaml(args.config)
    train(cfg, config_path=args.config)


if __name__ == "__main__":
    main()
