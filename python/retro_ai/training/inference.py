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
    ):
        self.model_path = model_path
        self.game_profile = game_profile
        self._target_fps = target_fps
        self.video_path = video_path
        self._logger = StructuredLogger("inference")

    def run(self, max_episodes: Optional[int] = None) -> None:
        """Run inference loop at target FPS."""
        env = self._build_env()
        model = self._load_model(env)
        recorder = self._maybe_init_recorder()
        episodes_run = 0

        while True:
            obs, info = env.reset()
            done = False
            skipped_frames = 0
            step = 0

            while not done:
                frame_start = time.perf_counter()
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                done = done or truncated
                step += 1

                if recorder:
                    recorder.add_frame(obs, reward=reward, step=step)

                # Frame pacing
                elapsed = time.perf_counter() - frame_start
                budget = 1.0 / self._target_fps
                if elapsed < budget:
                    time.sleep(budget - elapsed)
                else:
                    skipped_frames += 1

            if skipped_frames > 0:
                self._logger.info(
                    "frames_skipped",
                    count=skipped_frames,
                    episode=episodes_run,
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
            return VideoRecorder(path=self.video_path)
        return None
