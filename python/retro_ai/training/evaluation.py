"""Deterministic multi-episode evaluation of trained agents."""

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np

from retro_ai.core.logging import StructuredLogger
from retro_ai.core.preprocessing import (
    PreprocessedEnv,
    PreprocessingPipeline,
)
from retro_ai.envs.base_env import BaseEnv
from retro_ai.training.game_profile import (
    GameProfile,
    StartupSequenceWrapper,
)
from retro_ai.training.video import VideoRecorder
from retro_ai.wrappers.gymnasium_wrapper import GymnasiumWrapper


class EvaluationModule:
    """Run deterministic evaluation episodes and save results."""

    def __init__(
        self,
        model_path: str,
        game_profile: GameProfile,
        num_episodes: int = 10,
        base_seed: int = 42,
        output_dir: str = "output",
        video_path: Optional[str] = None,
    ):
        self.model_path = model_path
        self.game_profile = game_profile
        self.num_episodes = num_episodes
        self.base_seed = base_seed
        self.output_dir = output_dir
        self.video_path = video_path
        self._logger = StructuredLogger("evaluation")

    def run(self) -> Dict[str, Any]:
        """Run evaluation and return summary stats."""
        env = self._build_env()
        model = self._load_model(env)
        raw_env = self._find_base_env(env)
        recorder = self._maybe_init_recorder()
        results: List[Dict[str, Any]] = []

        for ep in range(self.num_episodes):
            seed = self.base_seed + ep
            obs, info = env.reset(seed=seed)
            episode_reward = 0.0
            episode_length = 0
            done = False
            step = 0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                done = done or truncated
                episode_reward += reward
                episode_length += 1
                step += 1

                if recorder and raw_env is not None:
                    recorder.add_frame(
                        raw_env._last_raw_obs, reward=episode_reward, step=step
                    )
                elif recorder:
                    recorder.add_frame(obs, reward=episode_reward, step=step)

            results.append(
                {
                    "episode": ep,
                    "seed": seed,
                    "reward": episode_reward,
                    "length": episode_length,
                    "score": info.get("score"),
                }
            )

        if recorder:
            recorder.close()

        summary = self._compute_summary(results)
        self._save_results(results, summary)
        return summary

    def _build_env(self):
        """Build the evaluation environment chain."""
        gp = self.game_profile
        config_dict = {}
        if hasattr(gp, "joystick_index"):
            config_dict["joystick_index"] = gp.joystick_index
        if gp.reward_params:
            config_dict["reward_params"] = gp.reward_params

        base = BaseEnv(
            emulator_type=gp.emulator_type,
            rom_path=gp.rom_path,
            bios_path=gp.bios_path,
            reward_mode=gp.reward_mode,
            config=config_dict or None,
        )
        pipeline = PreprocessingPipeline(
            grayscale=gp.grayscale,
            resize=gp.resize,
            frame_stack=gp.frame_stack,
            frame_skip=gp.frame_skip,
        )
        preprocessed = PreprocessedEnv(base, pipeline)
        env = GymnasiumWrapper(preprocessed)
        if gp.startup_sequence:
            env = StartupSequenceWrapper(env, gp.startup_sequence)
        return env

    def _load_model(self, env):
        """Load the trained model."""
        from stable_baselines3 import DQN, PPO

        for cls in (PPO, DQN):
            try:
                return cls.load(self.model_path, env=env)
            except Exception:
                continue
        raise ValueError(f"Could not load model from {self.model_path}")

    def _maybe_init_recorder(self) -> Optional[VideoRecorder]:
        if self.video_path and VideoRecorder.available():
            frame_skip = self.game_profile.frame_skip or 1
            video_fps = 60.0 / frame_skip
            return VideoRecorder(
                path=self.video_path,
                fps=video_fps,
                overlay=True,
                scale=3,
            )
        return None

    @staticmethod
    def _find_base_env(env):
        """Walk the wrapper chain to find the underlying BaseEnv."""
        current = env
        for _ in range(20):
            if hasattr(current, "_last_raw_obs"):
                return current
            if hasattr(current, "env"):
                current = current.env
            elif hasattr(current, "_env"):
                current = current._env
            else:
                break
        return None

    @staticmethod
    def _compute_summary(
        results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        rewards = [r["reward"] for r in results]
        lengths = [r["length"] for r in results]
        return {
            "num_episodes": len(results),
            "reward_mean": float(np.mean(rewards)),
            "reward_std": float(np.std(rewards)),
            "reward_min": float(np.min(rewards)),
            "reward_max": float(np.max(rewards)),
            "length_mean": float(np.mean(lengths)),
            "length_std": float(np.std(lengths)),
            "length_min": int(np.min(lengths)),
            "length_max": int(np.max(lengths)),
        }

    def _save_results(
        self,
        results: List[Dict[str, Any]],
        summary: Dict[str, Any],
    ) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        output = {
            "model_path": self.model_path,
            "game_profile": self.game_profile.name,
            "num_episodes": self.num_episodes,
            "base_seed": self.base_seed,
            "episodes": results,
            "summary": summary,
        }
        path = os.path.join(self.output_dir, "eval_results.json")
        with open(path, "w") as f:
            json.dump(output, f, indent=2)
