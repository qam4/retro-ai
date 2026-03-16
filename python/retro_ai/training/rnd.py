"""Random Network Distillation (RND) intrinsic reward wrapper.

Provides curiosity-driven exploration by computing an intrinsic reward bonus
based on the prediction error of a randomly initialized target network.
High prediction error indicates novel states, encouraging the agent to explore.

Architecture:
    BaseEnv → PreprocessedEnv → GymnasiumWrapper
    → RNDRewardWrapper → StartupSequenceWrapper
"""

from typing import Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn


class RNDNetwork(nn.Module):
    """Small CNN for RND target/predictor (framebuffer observations)."""

    def __init__(self, input_shape: Tuple[int, ...], output_dim: int = 64):
        super().__init__()
        c, h, w = input_shape
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Compute flattened size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            flat_size = self.conv(dummy).shape[1]
        self.fc = nn.Linear(flat_size, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x))


class RNDMLPNetwork(nn.Module):
    """Small MLP for RND target/predictor (RAM observations)."""

    def __init__(self, input_dim: int, output_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RunningMeanStd:
    """Welford's online algorithm for running mean and standard deviation.

    Used to normalize intrinsic rewards so the bonus scale stays stable
    over the course of training.
    """

    def __init__(self) -> None:
        self.mean: float = 0.0
        self.var: float = 1.0
        self.count: float = 1e-4  # small epsilon to avoid division by zero

    def update(self, x: float) -> None:
        """Incorporate a new sample into the running statistics."""
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.var += (delta * delta2 - self.var) / self.count

    def normalize(self, x: float) -> float:
        """Normalize a value using the running mean and standard deviation."""
        return (x - self.mean) / (max(self.var, 1e-8) ** 0.5)


class RNDRewardWrapper(gym.Wrapper):
    """Gymnasium wrapper that adds an RND intrinsic reward bonus.

    The wrapper maintains a fixed random target network and a trainable
    predictor network. The intrinsic reward is the MSE between their
    outputs — high error means the observation is novel.

    Combined reward = external_reward + coefficient * normalized_intrinsic_reward

    Works with both framebuffer (CNN) and RAM (MLP) observation modes,
    auto-detected from the observation space dimensionality.
    """

    def __init__(
        self,
        env: gym.Env,
        coefficient: float = 1.0,
        device: str = "cpu",
        learning_rate: float = 1e-3,
        update_freq: int = 1,
    ):
        super().__init__(env)
        self.coefficient = coefficient
        self.device = torch.device(device)
        self.update_freq = update_freq
        self._step_count = 0

        obs_space = env.observation_space
        obs_shape = obs_space.shape

        # Choose CNN or MLP based on observation dimensionality
        if len(obs_shape) == 3:
            # Image observations: (H, W, C) → PyTorch (C, H, W)
            c, h, w = obs_shape[2], obs_shape[0], obs_shape[1]
            self._target = RNDNetwork((c, h, w)).to(self.device)
            self._predictor = RNDNetwork((c, h, w)).to(self.device)
            self._is_image = True
        else:
            # Flat observations (RAM mode)
            input_dim = int(np.prod(obs_shape))
            self._target = RNDMLPNetwork(input_dim).to(self.device)
            self._predictor = RNDMLPNetwork(input_dim).to(self.device)
            self._is_image = False

        # Freeze target network — it stays random forever
        for p in self._target.parameters():
            p.requires_grad = False

        self._optimizer = torch.optim.Adam(
            self._predictor.parameters(), lr=learning_rate
        )
        self._reward_stats = RunningMeanStd()

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        # Compute intrinsic reward as MSE between target and predictor
        with torch.no_grad():
            obs_t = self._obs_to_tensor(obs)
            target_out = self._target(obs_t)
            pred_out = self._predictor(obs_t)
            intrinsic = ((target_out - pred_out) ** 2).mean().item()

        # Normalize using running statistics
        self._reward_stats.update(intrinsic)
        normalized = self._reward_stats.normalize(intrinsic)

        # Add intrinsic reward info to the info dict
        info["intrinsic_reward"] = intrinsic
        info["intrinsic_reward_normalized"] = normalized

        # Combined reward
        combined = reward + self.coefficient * normalized

        # Periodically update predictor to match target
        self._step_count += 1
        if self._step_count % self.update_freq == 0:
            self._update_predictor(obs)

        return obs, combined, done, truncated, info

    def _obs_to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        """Convert a numpy observation to a batched float tensor."""
        t = torch.from_numpy(obs.astype(np.float32) / 255.0).to(self.device)
        if self._is_image:
            t = t.permute(2, 0, 1)  # (H, W, C) → (C, H, W)
        else:
            t = t.flatten()
        return t.unsqueeze(0)  # add batch dimension

    def _update_predictor(self, obs: np.ndarray) -> None:
        """Train the predictor network to match the target output."""
        obs_t = self._obs_to_tensor(obs)
        with torch.no_grad():
            target_out = self._target(obs_t)
        pred_out = self._predictor(obs_t)
        loss = ((target_out - pred_out) ** 2).mean()
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
