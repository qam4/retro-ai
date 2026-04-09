#!/usr/bin/env python3
"""Go-Explore Phase 2: Backward curriculum training.

Starts training from near-goal states and progressively moves the
starting point backward toward the game start. Uses save states from
Phase 1's archive.

Usage:
    python scripts/go_explore_phase2.py \
        --archive output/mo5/yeti/go_explore_v8/archive.pkl \
        --profile yeti \
        --output output/mo5/yeti/go_explore_phase2
"""

import argparse
import os
import pickle
import random

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from retro_ai.envs.base_env import BaseEnv
from retro_ai.core.preprocessing import PreprocessedEnv, PreprocessingPipeline
from retro_ai.wrappers.gymnasium_wrapper import GymnasiumWrapper
from retro_ai.training.game_profile import GameProfileRegistry


class BackwardCurriculumEnv(gym.Env):
    """Env that starts from progressively earlier save states.

    Stage 0: start from floor 4 states (near princess)
    Stage 1: start from floor 3 states
    Stage 2: start from floor 2 states
    Stage 3: start from floor 1 states
    Stage 4: start from game start (floor 0)

    Advances to the next stage when the agent achieves a target reward
    threshold consistently.
    """

    metadata = {"render_modes": []}

    def __init__(self, profile_name, archive, max_steps=500):
        super().__init__()
        registry = GameProfileRegistry()
        profile = registry.load(profile_name)
        config_dict = {}
        if profile.reward_params:
            config_dict["reward_params"] = profile.reward_params

        self.base = BaseEnv(
            emulator_type=profile.emulator_type,
            rom_path=profile.rom_path,
            reward_mode=profile.reward_mode,
            config=config_dict or None,
            action_mode="joystick",
        )
        self.pipeline = PreprocessingPipeline(
            grayscale=profile.grayscale,
            resize=(84, 84),
            frame_stack=profile.frame_stack,
            frame_skip=profile.frame_skip,
        )
        self.preprocessed = PreprocessedEnv(
            self.base,
            self.pipeline,
            frame_maxpool=profile.frame_maxpool,
        )
        self.gym_env = GymnasiumWrapper(self.preprocessed)

        self.observation_space = self.gym_env.observation_space
        self.action_space = self.gym_env.action_space
        self.iface = self.base._interface

        self.archive = archive
        self.max_steps = max_steps

        # Group cells by floor (descending — stage 0 = highest floor)
        self.cells_by_floor = {}
        for cell_key, info in archive.items():
            floor = cell_key[0]
            if floor not in self.cells_by_floor:
                self.cells_by_floor[floor] = []
            self.cells_by_floor[floor].append(info)

        self.floors_desc = sorted(self.cells_by_floor.keys(), reverse=True)

        # Episode state
        self._step_count = 0
        self._start_score = 0
        self._prev_bonus = 0
        self._prev_lives = 5
        self._stall = 0
        self._initialized = False

    # Global curriculum state (shared across all env instances)
    _global_stage = 0
    _global_rewards = []
    _global_episode_count = 0
    _advance_threshold = 20.0  # mean score delta to advance (2 fruits)
    _advance_window = 100  # episodes to average over
    _frontier_ratio = 0.5  # 50% frontier, 50% earlier stages

    @property
    def current_stage(self):
        return BackwardCurriculumEnv._global_stage

    @property
    def num_stages(self):
        return len(self.floors_desc) + 1

    def _get_stage_floor(self):
        """Return the floor for the current stage, or None for game start."""
        if self.current_stage < len(self.floors_desc):
            return self.floors_desc[self.current_stage]
        return None  # game start

    def _pick_starting_floor(self):
        """Pick a floor to start from, mixing frontier and earlier stages."""
        if self.current_stage == 0:
            # Only one option — the highest floor
            return self.floors_desc[0]

        if self.current_stage >= len(self.floors_desc):
            # Past all floors — mix game start with all floors
            if random.random() < self._frontier_ratio:
                return None  # game start
            else:
                return random.choice(self.floors_desc)

        # Mix: 50% current frontier floor, 50% any earlier (higher) floor
        if random.random() < self._frontier_ratio:
            return self.floors_desc[self.current_stage]
        else:
            # Pick from stages 0..current_stage-1 (higher floors)
            earlier_idx = random.randint(0, self.current_stage - 1)
            return self.floors_desc[earlier_idx]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if not self._initialized:
            self.gym_env.reset(seed=seed)
            self._initialized = True

        floor = self._pick_starting_floor()

        if floor is not None and floor in self.cells_by_floor:
            # Start from a random cell on this floor
            cells = self.cells_by_floor[floor]
            info = random.choice(cells)
            self.base._interface.load_state(info["state"])
            obs, _, _, _, _ = self.gym_env.step([0, 0, 0])
        else:
            # Game start
            obs, _ = self.gym_env.reset()

        self._step_count = 0
        self._start_score = (
            self.iface.read_ram_byte(11093) << 8
        ) | self.iface.read_ram_byte(11094)
        self._prev_bonus = (
            self.iface.read_ram_byte(11010) << 8
        ) | self.iface.read_ram_byte(11011)
        self._prev_lives = self.iface.read_ram_byte(11095)
        self._stall = 0

        return obs, {}

    def step(self, action):
        obs, _, done, truncated, info = self.gym_env.step(action)
        self._step_count += 1

        score = (self.iface.read_ram_byte(11093) << 8) | self.iface.read_ram_byte(11094)
        lives = self.iface.read_ram_byte(11095)
        bonus = (self.iface.read_ram_byte(11010) << 8) | self.iface.read_ram_byte(11011)

        # Reward: score increase + survival
        score_delta = max(0, score - self._start_score)
        reward = score_delta * 0.1 + 0.01

        # Death detection
        if lives < self._prev_lives and self._prev_lives > 0:
            done = True
        self._prev_lives = lives

        if bonus == self._prev_bonus:
            self._stall += 1
        else:
            self._stall = 0
            self._prev_bonus = bonus
        if self._stall >= 10:
            done = True

        if self._step_count >= self.max_steps:
            truncated = True

        # Track episode reward for curriculum advancement
        if done or truncated:
            BackwardCurriculumEnv._global_rewards.append(score_delta)
            BackwardCurriculumEnv._global_episode_count += 1
            self._check_advance()

        return obs, reward, done, truncated, info

    def _check_advance(self):
        """Advance to next stage if performance is good enough."""
        cls = BackwardCurriculumEnv
        if len(cls._global_rewards) < cls._advance_window:
            return
        recent = cls._global_rewards[-cls._advance_window :]
        mean_reward = np.mean(recent)
        if (
            mean_reward >= cls._advance_threshold
            and cls._global_stage < self.num_stages - 1
        ):
            old_floor = self._get_stage_floor()
            cls._global_stage += 1
            new_floor = self._get_stage_floor()
            cls._global_rewards = []
            cls._global_episode_count = 0
            print(
                f"  STAGE ADVANCE: {cls._global_stage}/{self.num_stages} "
                f"(floor {old_floor} -> {new_floor}, mean_reward={mean_reward:.1f})",
                flush=True,
            )


def train(args):
    import time
    from stable_baselines3.common.callbacks import BaseCallback
    from retro_ai.wrappers.threaded_vec_env import ThreadedVecEnv

    print(f"Loading archive from {args.archive}", flush=True)
    with open(args.archive, "rb") as f:
        archive = pickle.load(f)
    print(f"  {len(archive)} cells loaded", flush=True)
    floors = {}
    for cell_key in archive:
        f = cell_key[0]
        floors[f] = floors.get(f, 0) + 1
    for f in sorted(floors):
        print(f"  Floor {f}: {floors[f]} cells", flush=True)

    os.makedirs(args.output, exist_ok=True)

    # Reset global curriculum state
    BackwardCurriculumEnv._global_stage = 0
    BackwardCurriculumEnv._global_rewards = []
    BackwardCurriculumEnv._global_episode_count = 0

    # Multi-env setup
    def make_env(rank):
        def _init():
            env = BackwardCurriculumEnv(args.profile, archive, max_steps=args.max_steps)
            return Monitor(env)

        return _init

    num_envs = args.num_envs
    if num_envs > 1:
        vec_env = ThreadedVecEnv([make_env(i) for i in range(num_envs)])
        print(f"  Using {num_envs} threaded envs", flush=True)
    else:
        vec_env = make_env(0)()

    # Progress callback
    class ProgressCallback(BaseCallback):
        def __init__(self, total, log_interval=1000):
            super().__init__()
            self._total = total
            self._log_interval = log_interval
            self._last_log = 0
            self._start = time.monotonic()
            self._frame_skip = 4

        def _on_step(self):
            if self.num_timesteps - self._last_log >= self._log_interval:
                elapsed = time.monotonic() - self._start
                fps = self.num_timesteps / elapsed if elapsed > 0 else 0
                emu_fps = fps * self._frame_skip
                pct = 100 * self.num_timesteps / self._total
                # Get stage from first env
                stage = f"{BackwardCurriculumEnv._global_stage}"
                print(
                    f"step {self.num_timesteps}/{self._total} ({pct:.0f}%) "
                    f"| emu_fps={emu_fps:.0f} | stage={stage}",
                    flush=True,
                )
                self._last_log = self.num_timesteps
            return True

    n_steps = max(1, 128 // num_envs) if num_envs > 1 else 128

    model = PPO(
        "CnnPolicy",
        vec_env,
        learning_rate=3e-4,
        batch_size=64,
        n_steps=n_steps,
        n_epochs=4,
        ent_coef=0.01,
        verbose=0,
        tensorboard_log=os.path.join(args.output, "tb"),
        device="auto",
    )

    print(f"\nBackward curriculum: {args.timesteps} steps, {num_envs} envs", flush=True)
    model.learn(
        total_timesteps=args.timesteps,
        callback=ProgressCallback(args.timesteps),
    )
    model.save(os.path.join(args.output, "final_model"))
    print(f"\nSaved model to {args.output}/final_model.zip", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Go-Explore Phase 2")
    parser.add_argument("--archive", required=True)
    parser.add_argument("--profile", default="yeti")
    parser.add_argument("--timesteps", type=int, default=2000000)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--output", default="output/mo5/yeti/go_explore_phase2")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
