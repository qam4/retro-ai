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

    python scripts/train_checkpoint_curriculum.py \\
        --config experiments/003-yeti/configs/curriculum_v6.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import threading
import time
from typing import Optional

import gymnasium as gym
import numpy as np
from retro_ai.training.callbacks import EpisodeMetricsCallback
from retro_ai.training.env_builder import build_training_env
from retro_ai.training.rewards import RewardContext, RewardFn
from retro_ai.training.rewards import create as create_reward
from retro_ai.training.run_config import RunConfig
from retro_ai.training.run_manifest import (
    EpisodeLogger,
    RunManifest,
    seed_everything,
)
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
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
# Level-cleared flag. See scripts/train_segment.py for the empirical
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


class CheckpointManager:
    """Manages save state buffers for each fruit checkpoint.

    ``reset_fraction`` / ``frontier_fraction`` / ``earlier_fraction``
    control the start distribution. All three must sum to 1.0 (validated
    on construction).
    """

    FRUITS_TOTAL = 4

    def __init__(
        self,
        max_states_per_checkpoint: int,
        min_states_to_advance: int,
        reset_fraction: float,
        frontier_fraction: float,
        earlier_fraction: float,
        min_survival_frames: int = 30,
        reach_threshold: float = 0.15,
    ):
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
        self.min_survival_frames = min_survival_frames
        # Approach 30: pick_start now weights non-reset levels by
        # (1 - success_rate) and reserves a fixed CP0 floor. We reuse
        # ``reset_fraction`` as that floor (frontier_fraction /
        # earlier_fraction are no longer used by pick_start, kept only
        # for config back-compat).
        self.cp0_floor = reset_fraction

        # Each pool is a plain list of (source_cp, bonus, state_bytes)
        # entries.
        #   source_cp: the CP level the *episode that produced this
        #     snapshot* started from. Lower = closer to a reset
        #     trajectory = more on-distribution for P(princess|CP0).
        #     This is the primary retention key (approach 30, R2).
        #   bonus: in-game bonus countdown at the snapshot moment;
        #     tiebreaker only (higher = faster reach).
        # When a pool is full we evict the entry with the HIGHEST
        # source_cp first (most artificial), tiebreaking on lowest
        # bonus — so the pool drifts toward reset-origin states the
        # agent actually reaches from reset.
        self.checkpoints = [[] for _ in range(self.FRUITS_TOTAL + 1)]
        self.frontier = 0
        self.stats = {
            "saves": [0] * (self.FRUITS_TOTAL + 1),
            "starts": [0] * (self.FRUITS_TOTAL + 1),
            # How many snapshots we rejected for being too precarious
            # (died too soon under the policy, didn't reach next CP).
            "rejected_precarious": [0] * (self.FRUITS_TOTAL + 1),
        }
        self.segment_attempts = [0] * (self.FRUITS_TOTAL + 1)
        self.segment_successes = [0] * (self.FRUITS_TOTAL + 1)

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
        self.reach_alpha = 0.02
        self.reset_reach_ema = [1.0, 0.0, 0.0, 0.0, 0.0]
        self.seg_success_ema = [0.0] * (self.FRUITS_TOTAL + 1)

    def record_episode(self, start_level, reached_level):
        if not (0 <= start_level <= self.FRUITS_TOTAL):
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
        # Reach-from-reset EMA: only reset (CP0) episodes are evidence
        # for "can the agent get to CP_n unaided". For each n in 1..4
        # the episode reached n iff reached_level >= n.
        if start_level == 0:
            for n in range(1, self.FRUITS_TOTAL + 1):
                hit = 1.0 if reached_level >= n else 0.0
                self.reset_reach_ema[n] = (1 - a) * self.reset_reach_ema[n] + a * hit

    def _insert(self, level, source_cp, bonus, state_bytes):
        """Insert a (source_cp, bonus, state) entry, keeping the pool
        reset-origin AND diverse.

        Retention priority is source_cp (lower = closer to a reset
        trajectory = more on-distribution, serves R2). When the pool is
        full we evict a *uniformly random* entry from the worst
        (highest source_cp) tier. The randomness is deliberate: it keeps
        the pool refreshing with recent, on-distribution states instead
        of freezing on a few snapshots.

        This replaces the old bonus-tiebreak eviction, which caused the
        v6 collapse: keeping only the highest-bonus (fastest-reach)
        states froze the CP1 pool to 2 distinct entries, and the
        curriculum then over-trained on those 2 stale states. ``bonus``
        is retained on the entry for logging but no longer drives
        eviction.
        """
        pool = self.checkpoints[level]
        entry = (int(source_cp), int(bonus), bytes(state_bytes))
        if len(pool) < self.max_states_per_checkpoint:
            pool.append(entry)
            self.stats["saves"][level] += 1
            self._maybe_advance_frontier()
            return
        # Pool full. Keep the most reset-origin states: only admit a
        # newcomer that is at least as reset-origin as the worst tier
        # we currently hold, and when we do, evict a random member of
        # that worst tier (diversity-preserving, recency-biased).
        worst_cp = max(e[0] for e in pool)
        if entry[0] > worst_cp:
            # Newcomer is more artificial than everything we have; drop.
            return
        worst_idxs = [i for i, e in enumerate(pool) if e[0] == worst_cp]
        pool[random.choice(worst_idxs)] = entry
        self.stats["saves"][level] += 1

    def save_checkpoint(self, fruits_collected, state_bytes, source_cp=0, bonus=0):
        # Used for offline seed_archive / preseed (no play-based score).
        if 0 <= fruits_collected <= self.FRUITS_TOTAL:
            self._insert(fruits_collected, source_cp, bonus, state_bytes)

    def save_scored(
        self,
        fruits_collected,
        state_bytes,
        survived_frames,
        reached_next,
        bonus,
        source_cp,
    ):
        """Admit a checkpoint snapshot judged by *real play*, not a probe.

        ``survived_frames`` is how long the agent stayed alive after
        the snapshot, under its own policy, in the episode that
        produced it. ``reached_next`` is whether that same episode went
        on to collect the next fruit (or the princess). ``bonus`` is
        the in-game bonus countdown at the snapshot moment (retention
        tiebreaker). ``source_cp`` is the CP level the producing
        episode started from (primary retention key — lower is more
        reset-origin / on-distribution).

        Admission is lenient (approach 30): keep the snapshot if it
        either led to the next checkpoint OR the agent survived at
        least ``min_survival_frames`` from it. Leniency protects the
        rare reaches at hard, sparse CPs; retention priority
        (source_cp) does the quality work on full, easy CPs.
        """
        if not (0 <= fruits_collected <= self.FRUITS_TOTAL):
            return
        admit = reached_next or survived_frames >= self.min_survival_frames
        if not admit:
            self.stats["rejected_precarious"][fruits_collected] += 1
            return
        self._insert(fruits_collected, source_cp, bonus, state_bytes)

    def _maybe_advance_frontier(self):
        while (
            self.frontier < self.FRUITS_TOTAL
            and len(self.checkpoints[self.frontier]) >= self.min_states_to_advance
        ):
            self.frontier = max(
                self.frontier,
                max(
                    i
                    for i in range(self.FRUITS_TOTAL + 1)
                    if len(self.checkpoints[i]) >= self.min_states_to_advance
                ),
            )
            break

    def pick_start(self):
        """Pick a starting checkpoint level (approach 31, reach-gated).

        - Reserve a fixed ``cp0_floor`` share for reset (CP0) starts:
          the engine that feeds fresh, on-distribution deep-CP seeds
          and forces end-to-end composition.
        - Otherwise pick among *eligible* levels only. A level is
          eligible iff its pool is non-empty AND the agent reaches it
          from reset often enough (``reset_reach_ema >= reach_threshold``).
          The reach gate keeps budget off levels whose pools are built
          from rare lucky reaches (off-distribution) — those don't
          transfer to P(princess | CP0).
        - Among eligible levels, weight by (1 - seg_success_ema) so
          practice concentrates on the deepest unsolved reset-reachable
          segment, and the frontier shifts forward as walls crack.
          Solved levels keep a small floor so the skill isn't abandoned.
        """
        # CP0 floor (always-available reset starts).
        if random.random() < self.cp0_floor:
            self.stats["starts"][0] += 1
            return 0, None

        eligible = [
            n
            for n in range(1, self.FRUITS_TOTAL + 1)
            if self.checkpoints[n] and self.reset_reach_ema[n] >= self.reach_threshold
        ]
        if not eligible:
            # Nothing reset-reachable yet: keep building reach via reset.
            self.stats["starts"][0] += 1
            return 0, None

        weights = [max(1.0 - self.seg_success_ema[n], 1e-3) for n in eligible]
        level = random.choices(eligible, weights=weights, k=1)[0]
        self.stats["starts"][level] += 1
        # Pool entries are (source_cp, bonus, state_bytes).
        _src, _bonus, state = random.choice(self.checkpoints[level])
        return level, state

    def summary(self):
        sizes = [len(self.checkpoints[i]) for i in range(self.FRUITS_TOTAL + 1)]
        rates = []
        for i in range(self.FRUITS_TOTAL + 1):
            if self.segment_attempts[i] > 0:
                pct = 100 * self.segment_successes[i] / self.segment_attempts[i]
                rates.append(f"{i}->{i+1}:{pct:.0f}%")
            else:
                rates.append(f"{i}->{i+1}:N/A")
        rej = self.stats.get("rejected_precarious", [0] * (self.FRUITS_TOTAL + 1))
        reach = "[" + ", ".join(f"{r:.2f}" for r in self.reset_reach_ema) + "]"
        return (
            f"cp={sizes} saves={self.stats['saves']} "
            f"rejected={rej} success=[{', '.join(rates)}] "
            f"reset_reach={reach}"
        )

    def save_to_disk(self, path):
        import pickle

        data = {
            "checkpoints": [
                list(self.checkpoints[i]) for i in range(self.FRUITS_TOTAL + 1)
            ],
            "stats": self.stats,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load_from_disk(self, path):
        import pickle

        if not os.path.exists(path):
            return
        with open(path, "rb") as f:
            data = pickle.load(f)
        for i, states in enumerate(data["checkpoints"]):
            for s in states:
                # Normalize to the (source_cp, bonus, state) format.
                # Loaded/offline states have unknown origin; mark them
                # source_cp = i (the level itself = maximally
                # artificial for that level) so fresh reset-origin
                # states evict them first.
                if isinstance(s, tuple) and len(s) == 3:
                    entry = (int(s[0]), int(s[1]), bytes(s[2]))
                elif isinstance(s, tuple) and len(s) == 2:
                    entry = (i, int(s[0]), bytes(s[1]))
                else:
                    entry = (i, 0, bytes(s))
                if len(self.checkpoints[i]) < self.max_states_per_checkpoint:
                    self.checkpoints[i].append(entry)
        loaded_stats = data.get("stats", self.stats)
        # Tolerate older checkpoint files that predate the
        # rejected_precarious counter.
        loaded_stats.setdefault("rejected_precarious", [0] * (self.FRUITS_TOTAL + 1))
        self.stats = loaded_stats
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
        self.action_space = stack.gym.action_space
        self.iface = stack.base._interface

        # MultiInputPolicy support: when enabled, the observation becomes
        # a Dict of the image plus a 4-d fruit-presence vector (1.0 =
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
                        low=0.0, high=1.0, shape=(4,), dtype=np.float32
                    ),
                }
            )
        else:
            self.observation_space = image_space

        self.max_steps = cfg.env.max_steps
        self.stall_threshold = cfg.env.stall_threshold
        self._reward_fn = reward_fn

        self.env_id = env_id
        self.episode_logger = episode_logger

        # Per-episode state
        self._step_count = 0
        self._prev_fruits = 4
        self._prev_lives = 5
        self._prev_bonus = 0
        self._prev_score = 0
        self._stall = 0
        self._start_fruits = 4
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
        """4-d fruit-presence vector (1.0 = on map, 0.0 = collected)."""
        return np.array(
            [
                1.0 if self.iface.read_ram_byte(FRUIT_PRESENCE_ADDRS[i]) != 0 else 0.0
                for i in (1, 2, 3, 4)
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

        if not self._initialized:
            self.gym_env.reset(seed=seed)
            self._initialized = True

        level, state_bytes = _manager.pick_start()

        if state_bytes is not None:
            self.base._interface.load_state(state_bytes)
            for _ in range(5):
                obs, _, _, _, _ = self.gym_env.step([0, 0, 0])
            self._start_state_hash = hashlib.blake2b(
                state_bytes, digest_size=8
            ).hexdigest()
        else:
            obs, _ = self.gym_env.reset()
            self._start_state_hash = ""

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
        # Highest checkpoint level reached this episode (fruits
        # collected; princess touch counts as level 5).
        self._max_cp_this_ep = self._start_fruits
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
            self.iface.read_ram_byte(FRUIT_PRESENCE_ADDRS[i]) != 0 for i in (1, 2, 3, 4)
        )
        princess_flag = self.iface.read_ram_byte(PRINCESS_FLAG_ADDR)
        princess_touched = princess_flag == 1 and self._prev_princess_flag == 0

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
        )
        reward = float(self._reward_fn(ctx))

        # Snapshot on fruit collection. Scoring is deferred to episode
        # end (see _pending_saves): we judge the state by how the rest
        # of the real episode unfolds, not a passive probe.
        if fruits < self._prev_fruits:
            self._fruits_collected_this_ep += self._prev_fruits - fruits
            collected_total = 4 - fruits
            self._max_cp_this_ep = max(self._max_cp_this_ep, collected_total)
            self._pending_saves.append(
                (
                    collected_total,
                    self.base._interface.save_state(),
                    self._step_count,
                    bonus,
                )
            )

        # Princess touch ends the episode and counts as a success.
        if princess_touched:
            self._fruits_collected_this_ep += 1
            # Princess is the terminal "checkpoint" (level 5); any
            # pending fruit snapshot in this episode therefore reached
            # the next checkpoint.
            self._max_cp_this_ep = max(self._max_cp_this_ep, 5)

        self._prev_fruits = fruits
        self._prev_score = score
        self._prev_princess_flag = princess_flag
        self._episode_reward += reward

        # Termination
        end_reason = None
        if lives < self._prev_lives and self._prev_lives > 0:
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
            start_level = 4 - self._start_fruits
            reached_level = 4 - fruits
            _manager.record_episode(start_level, reached_level)
            # Flush deferred checkpoint snapshots, scored by how the
            # rest of this episode actually played out. ``source_cp``
            # is the CP this episode started from — the retention key
            # that biases pools toward reset-origin states (approach 30).
            for level, state_bytes, save_step, save_bonus in self._pending_saves:
                survived = self._step_count - save_step
                reached_next = self._max_cp_this_ep > level
                _manager.save_scored(
                    level,
                    state_bytes,
                    survived,
                    reached_next,
                    save_bonus,
                    source_cp=start_level,
                )
            self._pending_saves = []
            if end_reason is None:
                end_reason = "env_done" if done else "env_truncated"
            self._log_episode(end_reason, fruits, bonus, score)

        return self._wrap_obs(obs), reward, done, truncated, info

    def _log_episode(
        self, end_reason: str, fruits: int, bonus: int, final_score: int
    ) -> None:
        if self.episode_logger is None:
            return
        final_xy = self._read_pos()
        start_level = 4 - self._start_fruits
        reached_level = 4 - fruits
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
        )


class CurriculumCallback(BaseCallback):
    """Log curriculum progress during training."""

    def __init__(self, total_timesteps: int, log_interval: int = 5000):
        super().__init__()
        self._total = total_timesteps
        self._log_interval = log_interval
        self._last_log = 0
        self._start = time.monotonic()

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
            self._last_log = self.num_timesteps
        return True


def train(cfg: RunConfig, config_path: Optional[str] = None) -> None:
    global _manager

    if cfg.curriculum is None:
        raise ValueError(
            "train_checkpoint_curriculum.py requires a 'curriculum' "
            "section in the run config"
        )

    seed = seed_everything(cfg.training.seed)

    _manager = CheckpointManager(
        max_states_per_checkpoint=cfg.curriculum.max_states_per_checkpoint,
        min_states_to_advance=cfg.curriculum.min_states_to_advance,
        reset_fraction=cfg.curriculum.reset_fraction,
        frontier_fraction=cfg.curriculum.frontier_fraction,
        earlier_fraction=cfg.curriculum.earlier_fraction,
        min_survival_frames=cfg.curriculum.min_survival_frames,
        reach_threshold=cfg.curriculum.reach_threshold,
    )

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
    manifest_extras["script"] = "scripts/train_checkpoint_curriculum.py"
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
        verbose=0,
        tensorboard_log=os.path.join(cfg.training.output, "tb"),
        device="auto",
        seed=seed,
    )

    if cfg.training.resume:
        print(f"  Resuming from {cfg.training.resume}", flush=True)
        model = PPO.load(
            cfg.training.resume,
            env=vec_env,
            tensorboard_log=os.path.join(cfg.training.output, "tb"),
        )
        ckpt_path = os.path.join(
            os.path.dirname(cfg.training.resume), "checkpoints.pkl"
        )
        _manager.load_from_disk(ckpt_path)
        _manager.load_from_disk(os.path.join(cfg.training.output, "checkpoints.pkl"))

    print("\nTraining...", flush=True)
    status = "COMPLETED"
    exit_code: Optional[int] = 0
    try:
        model.learn(
            total_timesteps=cfg.training.timesteps,
            callback=[
                CurriculumCallback(cfg.training.timesteps),
                EpisodeMetricsCallback(episode_logger, log_interval=10_000),
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
