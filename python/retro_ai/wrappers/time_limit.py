"""Wrapper that truncates episodes after a fixed number of steps."""

import gymnasium as gym


class TimeLimitWrapper(gym.Wrapper):
    """Truncate episodes after *max_steps* agent steps.

    Unlike Gymnasium's built-in ``TimeLimit``, this wrapper is designed
    for retro-ai's wrapper chain and sets ``truncated=True`` (not
    ``terminated``) so the agent knows the episode didn't end naturally.

    Parameters
    ----------
    env : gym.Env
        The environment to wrap.
    max_steps : int
        Maximum steps per episode before truncation.
    """

    def __init__(self, env: gym.Env, max_steps: int):
        super().__init__(env)
        self._max_steps = max_steps
        self._elapsed = 0

    def reset(self, **kwargs):
        self._elapsed = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self._elapsed += 1
        if not done and not truncated and self._elapsed >= self._max_steps:
            truncated = True
        return obs, reward, done, truncated, info
