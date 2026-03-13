"""Threaded vectorized environment using shared memory.

Takes advantage of the GIL being released during C++ emulator step/reset
calls. Unlike SubprocVecEnv, this avoids IPC serialization overhead by
keeping all environments in the same process and using threads.

This only works when the underlying environment releases the GIL during
its computationally intensive operations (which retro_ai_native does).
"""

from __future__ import annotations

import concurrent.futures
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnv,
    VecEnvObs,
    VecEnvStepReturn,
)


class ThreadedVecEnv(VecEnv):
    """Vectorized environment using a thread pool.

    Parameters
    ----------
    env_fns : list of callables
        Each callable returns a Gymnasium-compatible environment.
    """

    def __init__(self, env_fns: List[Callable[[], Any]]) -> None:
        self._envs = [fn() for fn in env_fns]
        env = self._envs[0]
        super().__init__(
            num_envs=len(env_fns),
            observation_space=env.observation_space,
            action_space=env.action_space,
        )
        self._pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=len(env_fns)
        )
        # Buffers
        self._obs = np.zeros(
            (self.num_envs,) + self.observation_space.shape,
            dtype=self.observation_space.dtype,
        )
        self._rewards = np.zeros(self.num_envs, dtype=np.float64)
        self._dones = np.zeros(self.num_envs, dtype=bool)
        self._infos: List[Dict[str, Any]] = [{} for _ in range(self.num_envs)]

    def reset(self) -> VecEnvObs:
        """Reset all environments in parallel."""
        def _reset(i):
            obs, info = self._envs[i].reset()
            return i, obs, info

        futures = [self._pool.submit(_reset, i) for i in range(self.num_envs)]
        for f in concurrent.futures.as_completed(futures):
            i, obs, info = f.result()
            self._obs[i] = obs
            self._infos[i] = info

        return self._obs.copy()

    def step_async(self, actions: np.ndarray) -> None:
        """Submit step calls to the thread pool."""
        self._pending = []
        for i in range(self.num_envs):
            action = actions[i]
            if isinstance(action, np.ndarray):
                action = action.item() if action.ndim == 0 else action.tolist()
            elif isinstance(action, (np.integer,)):
                action = int(action)
            self._pending.append(
                self._pool.submit(self._step_one, i, action)
            )

    def step_wait(self) -> VecEnvStepReturn:
        """Collect results from pending step calls."""
        for f in concurrent.futures.as_completed(self._pending):
            i, obs, reward, done, truncated, info = f.result()
            if done or truncated:
                # Auto-reset (SB3 convention)
                info["terminal_observation"] = obs
                obs, reset_info = self._envs[i].reset()
                info.update(reset_info)
            self._obs[i] = obs
            self._rewards[i] = reward
            self._dones[i] = done or truncated
            self._infos[i] = info

        self._pending = []
        return self._obs.copy(), self._rewards.copy(), self._dones.copy(), self._infos.copy()

    def _step_one(self, i: int, action) -> Tuple:
        obs, reward, done, truncated, info = self._envs[i].step(action)
        return i, obs, reward, done, truncated, info

    def close(self) -> None:
        """Shut down the thread pool."""
        self._pool.shutdown(wait=False)
        for env in self._envs:
            env.close()

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):
        if indices is None:
            indices = range(self.num_envs)
        return [getattr(self._envs[i], method_name)(*method_args, **method_kwargs) for i in indices]

    def env_is_wrapped(self, wrapper_class, indices=None):
        if indices is None:
            indices = range(self.num_envs)
        return [isinstance(self._envs[i], wrapper_class) for i in indices]

    def get_attr(self, attr_name: str, indices=None):
        if indices is None:
            indices = range(self.num_envs)
        return [getattr(self._envs[i], attr_name) for i in indices]

    def set_attr(self, attr_name: str, value, indices=None):
        if indices is None:
            indices = range(self.num_envs)
        for i in indices:
            setattr(self._envs[i], attr_name, value)

    def seed(self, seed: Optional[int] = None) -> List:
        return [None] * self.num_envs

    def get_images(self) -> Sequence[np.ndarray]:
        return [env.render() for env in self._envs]
