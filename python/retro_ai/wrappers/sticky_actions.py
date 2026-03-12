"""Wrapper that implements sticky actions for stochastic environments."""

import random

import gymnasium as gym


class StickyActionsWrapper(gym.Wrapper):
    """With probability p, repeat the previous action instead of the new one.

    This adds stochasticity to deterministic games, preventing the agent
    from memorizing frame-perfect action sequences. Standard in ALE v5
    (p=0.25).

    Parameters
    ----------
    env : gym.Env
        The environment to wrap.
    sticky_prob : float
        Probability of repeating the previous action (default 0.25).
    """

    def __init__(self, env: gym.Env, sticky_prob: float = 0.25):
        super().__init__(env)
        self._sticky_prob = sticky_prob
        self._last_action = None

    def reset(self, **kwargs):
        self._last_action = None
        return self.env.reset(**kwargs)

    def step(self, action):
        if self._last_action is not None and random.random() < self._sticky_prob:
            action = self._last_action
        self._last_action = action
        return self.env.step(action)
