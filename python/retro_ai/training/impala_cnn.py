"""IMPALA-style ResNet CNN feature extractor for higher resolution inputs.

Based on the architecture from:
  Espeholt et al., "IMPALA: Scalable Distributed Deep-RL" (2018)
  Used by OpenAI for Procgen benchmark with PPO.

Compared to Nature CNN (8x8 stride 4 first layer), this preserves
spatial detail through residual blocks with 3x3 kernels and max pooling.
Suitable for inputs larger than 84x84 (e.g. 160x100, 320x200).
"""

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ResidualBlock(nn.Module):
    """Simple residual block: conv -> relu -> conv + skip."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        out = nn.functional.relu(self.conv1(x))
        out = self.conv2(out)
        return nn.functional.relu(out + x)


class ImpalaCNN(BaseFeaturesExtractor):
    """IMPALA-style CNN with residual blocks.

    Architecture per block:
      Conv2d(in, out, 3x3) -> MaxPool2d(3, stride=2) -> ResidualBlock -> ResidualBlock

    Each block halves the spatial dimensions. Number of blocks adapts
    to input size to produce a manageable feature map.

    Parameters
    ----------
    observation_space : gym.spaces.Box
        Must be image-like (H, W, C) or (C, H, W).
    channels : tuple of int
        Number of output channels per block. Default (16, 32, 32) for
        small inputs, (16, 32, 32, 64) for larger inputs.
    features_dim : int
        Size of the output feature vector (after flatten + linear).
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        channels: tuple = None,
        features_dim: int = 256,
    ):
        # Compute input shape
        shape = observation_space.shape
        if len(shape) == 3:
            if shape[0] < shape[2]:
                # CHW format
                in_channels, h, w = shape
            else:
                # HWC format
                h, w, in_channels = shape
        else:
            raise ValueError(f"Expected 3D observation, got shape {shape}")

        # Auto-select channel depths based on input size
        if channels is None:
            if h >= 160 or w >= 160:
                channels = (16, 32, 32, 64)
            else:
                channels = (16, 32, 32)

        # Build blocks
        blocks = []
        ch_in = in_channels
        cur_h, cur_w = h, w
        for ch_out in channels:
            blocks.append(nn.Conv2d(ch_in, ch_out, 3, padding=1))
            blocks.append(nn.MaxPool2d(3, stride=2, padding=1))
            blocks.append(ResidualBlock(ch_out))
            blocks.append(ResidualBlock(ch_out))
            ch_in = ch_out
            cur_h = (cur_h + 1) // 2
            cur_w = (cur_w + 1) // 2

        blocks.append(nn.Flatten())

        # Compute flattened size
        flat_size = ch_in * cur_h * cur_w

        super().__init__(observation_space, features_dim)

        self.cnn = nn.Sequential(*blocks)
        self.linear = nn.Sequential(
            nn.Linear(flat_size, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # SB3 passes observations as (batch, C, H, W) after VecTransposeImage
        return self.linear(self.cnn(observations))
