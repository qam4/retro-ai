"""Wrapper that adds a per-step survival bonus to the base reward."""

import gymnasium as gym


class SurvivalBonusWrapper(gym.Wrapper):
    """Add a fixed bonus to every non-terminal step's reward.

    Useful for games where the base reward (e.g. score delta) is sparse.
    The survival bonus gives the agent a dense signal to learn from:
    staying alive longer = more reward, plus any score bonuses on top.

    Parameters
    ----------
    env : gym.Env
        The environment to wrap.
    bonus : float
        Reward added per step (default 1.0).
    """

    def __init__(self, env: gym.Env, bonus: float = 1.0):
        super().__init__(env)
        self._bonus = bonus

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        if not done and not truncated:
            reward += self._bonus
        return obs, reward, done, truncated, info
