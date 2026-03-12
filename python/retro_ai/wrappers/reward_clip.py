"""Wrapper that clips rewards to a fixed range."""

import gymnasium as gym


class RewardClipWrapper(gym.Wrapper):
    """Clip rewards to [-max_reward, +max_reward].

    Normalizes reward scale across games and stabilizes training.
    Standard in Atari benchmarks (clip to [-1, +1]).

    Parameters
    ----------
    env : gym.Env
        The environment to wrap.
    max_reward : float
        Maximum absolute reward value (default 1.0).
    """

    def __init__(self, env: gym.Env, max_reward: float = 1.0):
        super().__init__(env)
        self._max = max_reward

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        reward = max(-self._max, min(self._max, reward))
        return obs, reward, done, truncated, info
