"""Real-time inference runner for trained agents."""

import time
from typing import Optional

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


class InferenceRunner:
    """Run a trained agent at target FPS with optional recording."""

    def __init__(
        self,
        model_path: str,
        game_profile: GameProfile,
        target_fps: float = 60.0,
        video_path: Optional[str] = None,
        overlay: bool = True,
    ):
        self.model_path = model_path
        self.game_profile = game_profile
        self._target_fps = target_fps
        self.video_path = video_path
        self._overlay = overlay
        self._logger = StructuredLogger("inference")

    def run(self, max_episodes: Optional[int] = None) -> None:
        """Run inference loop at target FPS."""
        env = self._build_env()
        model = self._load_model(env)
        raw_env = self._find_base_env(env)
        recorder = self._maybe_init_recorder()
        episodes_run = 0

        while True:
            obs, info = env.reset()
            done = False
            skipped_frames = 0
            step = 0
            episode_reward = 0.0

            while not done:
                frame_start = time.perf_counter()
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                done = done or truncated
                step += 1
                episode_reward += reward

                if recorder and raw_env is not None:
                    # Record raw emulator frame (full resolution RGB)
                    recorder.add_frame(raw_env._last_raw_obs, reward=episode_reward, step=step)
                elif recorder:
                    recorder.add_frame(obs, reward=episode_reward, step=step)

                # Progress logging every 200 steps
                if step % 200 == 0:
                    self._logger.info(
                        "play_progress",
                        {"episode": episodes_run + 1, "step": step, "reward": episode_reward},
                    )

                # Frame pacing
                elapsed = time.perf_counter() - frame_start
                budget = 1.0 / self._target_fps
                if elapsed < budget:
                    time.sleep(budget - elapsed)
                else:
                    skipped_frames += 1

            self._logger.info(
                "episode_complete",
                {
                    "episode": episodes_run + 1,
                    "reward": episode_reward,
                    "length": step,
                    "skipped_frames": skipped_frames,
                },
            )

            episodes_run += 1
            if max_episodes is not None and episodes_run >= max_episodes:
                break

        if recorder:
            recorder.close()

    def _build_env(self):
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
        from stable_baselines3 import DQN, PPO

        for cls in (PPO, DQN):
            try:
                return cls.load(self.model_path, env=env)
            except Exception:
                continue
        raise ValueError(f"Could not load model from {self.model_path}")

    def _maybe_init_recorder(self) -> Optional[VideoRecorder]:
        if self.video_path and VideoRecorder.available():
            # Video FPS = real-time playback rate, independent of emulator speed.
            # The emulator runs at 60 Hz; with frame_skip=N the agent sees
            # 60/N unique frames per real-time second.
            frame_skip = self.game_profile.frame_skip or 1
            video_fps = 60.0 / frame_skip
            return VideoRecorder(
                path=self.video_path,
                fps=video_fps,
                overlay=self._overlay,
                scale=3,
            )
        if self.video_path and not VideoRecorder.available():
            self._logger.warning(
                "video_unavailable",
                {"message": "opencv-python not installed, video recording disabled"},
            )
        return None

    @staticmethod
    def _find_base_env(env):
        """Walk the wrapper chain to find the underlying BaseEnv."""
        current = env
        for _ in range(20):  # safety limit
            if hasattr(current, "_last_raw_obs"):
                return current
            # Wrappers use .env or ._env for the inner environment
            if hasattr(current, "env"):
                current = current.env
            elif hasattr(current, "_env"):
                current = current._env
            else:
                break
        return None
