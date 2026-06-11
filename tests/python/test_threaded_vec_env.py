"""Tests for ThreadedVecEnv, focused on Dict-observation support
(required for MultiInputPolicy: image + fruit-presence vector)."""

from __future__ import annotations

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")
pytest.importorskip("stable_baselines3")

from gymnasium import spaces  # noqa: E402
from retro_ai.wrappers.threaded_vec_env import ThreadedVecEnv  # noqa: E402


class _DictEnv(gym.Env):
    """Minimal env with a Dict observation space and a step counter."""

    def __init__(self, env_id: int):
        super().__init__()
        self.env_id = env_id
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(0, 255, (4, 4, 1), dtype=np.uint8),
                "fruits": spaces.Box(0.0, 1.0, (4,), dtype=np.float32),
            }
        )
        self.action_space = spaces.Discrete(2)
        self._t = 0

    def _obs(self):
        return {
            "image": np.full((4, 4, 1), self.env_id, dtype=np.uint8),
            "fruits": np.array([self._t % 2] * 4, dtype=np.float32),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        return self._obs(), {}

    def step(self, action):
        self._t += 1
        done = self._t >= 3
        return self._obs(), 1.0, done, False, {}


def _make(i):
    def _init():
        return _DictEnv(i)

    return _init


def test_dict_obs_space_preserved():
    vec = ThreadedVecEnv([_make(i) for i in range(3)])
    try:
        assert isinstance(vec.observation_space, spaces.Dict)
        obs = vec.reset()
        assert set(obs.keys()) == {"image", "fruits"}
        # Batched along axis 0 across the 3 envs.
        assert obs["image"].shape == (3, 4, 4, 1)
        assert obs["fruits"].shape == (3, 4)
        # Each env stamped its id into the image channel.
        assert [int(obs["image"][i].flat[0]) for i in range(3)] == [0, 1, 2]
    finally:
        vec.close()


def test_dict_obs_step_and_autoreset():
    vec = ThreadedVecEnv([_make(i) for i in range(2)])
    try:
        vec.reset()
        term_infos = []
        # Step past the episode horizon (3) to exercise auto-reset.
        for _ in range(4):
            obs, rewards, dones, infos = vec.step(np.array([0, 0]))
            assert obs["image"].shape == (2, 4, 4, 1)
            assert obs["fruits"].shape == (2, 4)
            term_infos.extend(
                info["terminal_observation"]
                for info in infos
                if "terminal_observation" in info
            )
        # Auto-reset must have fired and preserved a Dict terminal obs.
        assert term_infos
        assert set(term_infos[0].keys()) == {"image", "fruits"}
    finally:
        vec.close()
