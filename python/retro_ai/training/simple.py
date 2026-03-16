"""SimPLe (Simulated Policy Learning) data structures and training components."""

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

from retro_ai.core.logging import StructuredLogger
from retro_ai.training.config import TrainingConfig
from retro_ai.training.metrics import MetricsTracker

logger = logging.getLogger(__name__)


@dataclass
class Transition:
    """A single environment transition."""

    observation: np.ndarray  # (H, W, C) or (C, H, W)
    action: int
    reward: float
    next_observation: np.ndarray
    done: bool


class TransitionBuffer:
    """Fixed-capacity circular buffer for environment transitions.

    Pre-allocates numpy arrays for efficient storage and sampling.
    """

    def __init__(self, capacity: int, obs_shape: Tuple[int, ...]):
        self.capacity = capacity
        self.obs_shape = obs_shape
        self._pos = 0
        self._size = 0

        # Pre-allocate storage
        self.observations = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_observations = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.dones = np.zeros(capacity, dtype=np.bool_)

    @property
    def size(self) -> int:
        """Number of transitions currently stored."""
        return self._size

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Add a transition, wrapping around when full."""
        self.observations[self._pos] = obs
        self.actions[self._pos] = action
        self.rewards[self._pos] = reward
        self.next_observations[self._pos] = next_obs
        self.dones[self._pos] = done

        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return a random batch of transitions as numpy arrays."""
        indices = np.random.randint(0, self._size, size=batch_size)
        return (
            self.observations[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_observations[indices],
            self.dones[indices],
        )

    def sample_starts(self, n: int) -> np.ndarray:
        """Return n random observations for use as rollout starting states."""
        indices = np.random.randint(0, self._size, size=n)
        return self.observations[indices]


class WorldModel(nn.Module):
    """CNN that predicts next observation and reward from (obs, action)."""

    def __init__(self, obs_shape: Tuple[int, int, int], num_actions: int):
        super().__init__()
        c, h, w = obs_shape

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(c, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute flattened size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            flat_size = self.encoder(dummy).shape[1]
        self._flat_size = flat_size
        self._decoder_shape: Tuple[int, ...] = ()  # set below

        self.encoder_fc = nn.Linear(flat_size, 512)

        # Action embedding
        self.action_embed = nn.Embedding(num_actions, 64)

        # Combined
        self.combined_fc = nn.Sequential(
            nn.Linear(512 + 64, 512),
            nn.ReLU(),
        )

        # Observation decoder
        # Trace encoder spatial dims to compute output_padding for each deconv
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            e1 = nn.Conv2d(c, 64, 4, stride=2)(dummy)
            e2 = nn.Conv2d(64, 128, 4, stride=2)(e1)
            e3 = nn.Conv2d(128, 256, 4, stride=2)(e2)
            self._decoder_shape = e3.shape[1:]  # (256, h', w')
            # Compute output_padding so deconv inverts conv exactly
            # ConvTranspose2d out = (in-1)*stride + kernel + output_padding - 2*padding
            sizes = [e3.shape[2], e2.shape[2], e1.shape[2], dummy.shape[2]]

        def _opad(in_s: int, target: int, k: int = 4, s: int = 2) -> int:
            return target - ((in_s - 1) * s + k)

        op0 = _opad(sizes[0], sizes[1])
        op1 = _opad(sizes[1], sizes[2])
        op2 = _opad(sizes[2], sizes[3])

        decoder_flat = int(np.prod(self._decoder_shape))
        self.decoder_fc = nn.Linear(512, decoder_flat)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, output_padding=op0),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, output_padding=op1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, c, 4, stride=2, output_padding=op2),
            nn.Sigmoid(),
        )

        # Reward head
        self.reward_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            obs: (batch, C, H, W) uint8 or float tensor.
            action: (batch,) long tensor of action indices.

        Returns:
            (predicted_next_obs, predicted_reward) where
            predicted_next_obs is (batch, C, H, W) in [0, 1] and
            predicted_reward is (batch, 1).
        """
        # Normalize to [0, 1]
        x = obs.float() / 255.0 if obs.dtype == torch.uint8 else obs.float()

        # Encode
        h = self.encoder(x)
        h = self.encoder_fc(h)

        # Action embedding
        a = self.action_embed(action)

        # Combine
        combined = self.combined_fc(torch.cat([h, a], dim=-1))

        # Decode observation
        dec = self.decoder_fc(combined)
        dec = dec.view(-1, *self._decoder_shape)
        pred_obs = self.decoder(dec)

        # Predict reward
        pred_reward = self.reward_head(combined)

        return pred_obs, pred_reward


def train_world_model(
    model: WorldModel,
    buffer: "TransitionBuffer",
    epochs: int,
    batch_size: int,
    lr: float,
    device: str = "cpu",
) -> List[float]:
    """Train the world model on buffer data.

    Returns list of per-epoch average losses.
    """
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_history: List[float] = []

    steps_per_epoch = max(1, buffer.size // batch_size)

    for _epoch in range(epochs):
        epoch_loss = 0.0
        for _ in range(steps_per_epoch):
            obs, actions, rewards, next_obs, _dones = buffer.sample(batch_size)

            obs_t = torch.from_numpy(obs).float().to(device)
            act_t = torch.from_numpy(actions).long().to(device)
            rew_t = torch.from_numpy(rewards).float().to(device).unsqueeze(-1)
            next_obs_t = torch.from_numpy(next_obs).float().to(device) / 255.0

            pred_obs, pred_reward = model(obs_t, act_t)

            obs_loss = nn.functional.mse_loss(pred_obs, next_obs_t)
            reward_loss = nn.functional.mse_loss(pred_reward, rew_t)
            loss = obs_loss + reward_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        loss_history.append(epoch_loss / steps_per_epoch)

    return loss_history


def validate_world_model(
    model: WorldModel,
    buffer: "TransitionBuffer",
    batch_size: int,
    device: str = "cpu",
    quality_threshold: float = 0.1,
) -> Tuple[float, float]:
    """Compute validation MSE on a random batch.

    Returns (obs_mse, reward_mse). Logs a warning if either exceeds
    quality_threshold.
    """
    model.to(device)
    model.eval()

    obs, actions, rewards, next_obs, _dones = buffer.sample(batch_size)

    obs_t = torch.from_numpy(obs).float().to(device)
    act_t = torch.from_numpy(actions).long().to(device)
    rew_t = torch.from_numpy(rewards).float().to(device).unsqueeze(-1)
    next_obs_t = torch.from_numpy(next_obs).float().to(device) / 255.0

    with torch.no_grad():
        pred_obs, pred_reward = model(obs_t, act_t)
        obs_mse = nn.functional.mse_loss(pred_obs, next_obs_t).item()
        reward_mse = nn.functional.mse_loss(pred_reward, rew_t).item()

    if obs_mse > quality_threshold:
        logger.warning(
            "World model observation MSE %.4f exceeds threshold %.4f",
            obs_mse,
            quality_threshold,
        )
    if reward_mse > quality_threshold:
        logger.warning(
            "World model reward MSE %.4f exceeds threshold %.4f",
            reward_mse,
            quality_threshold,
        )

    return obs_mse, reward_mse


class SyntheticGenerator:
    """Generates synthetic rollouts using the world model."""

    def __init__(
        self,
        world_model: WorldModel,
        horizon: int = 50,
        device: str = "cpu",
        done_threshold: float = -10.0,
    ):
        self.world_model = world_model
        self.horizon = horizon
        self.device = device
        self.done_threshold = done_threshold

    def generate(
        self, start_obs: np.ndarray, policy, num_rollouts: int
    ) -> List[Transition]:
        """Unroll world model from starting observations using policy.

        Each rollout terminates early if the world model predicts a reward
        below ``done_threshold`` (indicating episode end) or when the
        ``horizon`` is reached.

        Args:
            start_obs: Starting observations, (num_rollouts, C, H, W) uint8 numpy array.
            policy: Any object with a ``.predict(obs)``
                method (SB3 API).
            num_rollouts: Number of rollouts to generate.

        Returns:
            Flat list of synthetic Transitions.
        """
        self.world_model.to(self.device)
        self.world_model.eval()

        transitions: List[Transition] = []

        with torch.no_grad():
            for i in range(num_rollouts):
                obs = start_obs[i]  # (C, H, W) uint8 numpy
                for step in range(self.horizon):
                    action, _ = policy.predict(obs, deterministic=True)
                    action_int = int(action)

                    # Prepare tensors for world model (expects (1, C, H, W) float)
                    obs_t = torch.from_numpy(obs).unsqueeze(0).float().to(self.device)
                    act_t = torch.tensor(
                        [action_int], dtype=torch.long, device=self.device
                    )

                    pred_obs, pred_reward = self.world_model(obs_t, act_t)

                    # Convert predicted obs back to uint8 [0, 255] numpy
                    next_obs = (
                        (pred_obs.squeeze(0).cpu().numpy() * 255.0)
                        .clip(0, 255)
                        .astype(np.uint8)
                    )
                    reward = pred_reward.item()

                    # Predict done: reward below threshold or horizon reached
                    done = reward < self.done_threshold or step == self.horizon - 1

                    transitions.append(
                        Transition(
                            observation=obs,
                            action=action_int,
                            reward=reward,
                            next_observation=next_obs,
                            done=done,
                        )
                    )

                    # Early termination when world model predicts done
                    if done:
                        break

                    obs = next_obs

        return transitions


class DreamEnv(gym.Env):
    """Gymnasium environment that uses a WorldModel to simulate steps.

    Resets by sampling a random starting state from the TransitionBuffer.
    Steps by running the world model forward. Terminates after
    ``rollout_horizon`` steps.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        world_model: WorldModel,
        buffer: TransitionBuffer,
        num_actions: int,
        rollout_horizon: int = 50,
        device: str = "cpu",
        real_action_space=None,
        real_obs_space=None,
    ):
        super().__init__()
        self._world_model = world_model
        self._buffer = buffer
        self._num_actions = num_actions
        self._horizon = rollout_horizon
        self._device = device
        self._step_count = 0
        self._current_obs: Optional[np.ndarray] = None
        self._real_action_space = real_action_space

        # Use real env's observation space if provided, else infer from buffer
        if real_obs_space is not None:
            self.observation_space = real_obs_space
        else:
            obs_shape = buffer.obs_shape
            self.observation_space = spaces.Box(
                low=0, high=255, shape=obs_shape, dtype=np.uint8
            )

        # Use real env's action space if provided
        if real_action_space is not None:
            self.action_space = real_action_space
        else:
            self.action_space = spaces.Discrete(num_actions)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Sample a random starting observation from the real buffer (stored as C,H,W)
        starts = self._buffer.sample_starts(1)
        obs_chw = starts[0]
        # Convert to (H,W,C) for Gymnasium
        self._current_obs = np.transpose(obs_chw, (1, 2, 0)) if obs_chw.ndim == 3 else obs_chw
        self._step_count = 0
        return self._current_obs.copy(), {}

    def step(self, action):
        self._world_model.eval()

        # Flatten MultiDiscrete action to single int for world model
        if hasattr(action, '__len__') and len(action) > 1:
            nvec = self._real_action_space.nvec if self._real_action_space is not None and hasattr(self._real_action_space, 'nvec') else None
            if nvec is not None:
                action_int = 0
                multiplier = 1
                for i in reversed(range(len(action))):
                    action_int += int(action[i]) * multiplier
                    multiplier *= int(nvec[i])
            else:
                action_int = int(action[0])
        else:
            action_int = int(action)

        # World model works in (C,H,W) space — transpose from (H,W,C)
        obs_chw = self._current_obs
        if obs_chw.ndim == 3 and obs_chw.shape[2] < obs_chw.shape[0]:
            # (H,W,C) → (C,H,W)
            obs_chw = np.transpose(obs_chw, (2, 0, 1))

        obs_t = torch.from_numpy(obs_chw).unsqueeze(0).float().to(self._device)
        act_t = torch.tensor([action_int], dtype=torch.long, device=self._device)

        with torch.no_grad():
            pred_obs, pred_reward = self._world_model(obs_t, act_t)

        # World model outputs (C,H,W), convert back to (H,W,C) for Gymnasium
        next_obs_chw = (pred_obs.squeeze(0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        next_obs = np.transpose(next_obs_chw, (1, 2, 0)) if next_obs_chw.ndim == 3 else next_obs_chw
        reward = pred_reward.item()

        self._step_count += 1
        terminated = self._step_count >= self._horizon
        truncated = False

        self._current_obs = next_obs
        return next_obs.copy(), reward, terminated, truncated, {}


class SimplePipeline:
    """Orchestrates the SimPLe (Simulated Policy Learning) training loop.

    Each round: collect real data → train world model → train PPO on
    DreamEnv (world-model simulated environment).
    """

    def __init__(
        self,
        config: TrainingConfig,
        logger: Optional[StructuredLogger] = None,
    ):
        self.config = config
        self._logger = logger or StructuredLogger("simple_pipeline")
        self._metrics: Optional[MetricsTracker] = None

    def run(self) -> Path:
        """Execute the iterative SimPLe training loop.

        Returns path to the saved final model.
        """
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv

        from retro_ai.training.pipeline import TrainingPipeline

        simple_cfg = self.config.simple
        os.makedirs(self.config.output_dir, exist_ok=True)

        self._metrics = MetricsTracker(
            self.config.output_dir,
            rolling_window=self.config.rolling_window,
        )

        self._logger.info(
            "simple_start",
            {
                "num_rounds": simple_cfg.num_rounds,
                "total_timesteps": self.config.total_timesteps,
                "synthetic_ratio": simple_cfg.synthetic_ratio,
            },
        )

        # Build real environment using TrainingPipeline's env construction
        helper = TrainingPipeline(self.config, self._logger)
        helper._resolve_profile()
        real_env = helper._build_env()

        # Determine observation shape and action count
        obs_space = real_env.observation_space
        obs_shape = obs_space.shape  # (H, W, C) from Gymnasium
        # WorldModel expects (C, H, W) — transpose for PyTorch convention
        if len(obs_shape) == 3:
            obs_shape_chw = (obs_shape[2], obs_shape[0], obs_shape[1])
        else:
            obs_shape_chw = obs_shape
        if hasattr(real_env.action_space, "n"):
            num_actions = real_env.action_space.n
        else:
            # MultiDiscrete — flatten to total combos for world model
            num_actions = int(np.prod(real_env.action_space.nvec))

        # Resolve device
        device = self.config.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize components — buffer stores (C, H, W) observations
        buffer = TransitionBuffer(
            capacity=self.config.total_timesteps, obs_shape=obs_shape_chw
        )
        world_model = WorldModel(obs_shape_chw, num_actions)
        world_model.to(device)

        # Initialize PPO policy on real env
        policy = PPO(
            "CnnPolicy" if len(obs_shape) == 3 else "MlpPolicy",
            real_env,
            learning_rate=self.config.algorithm.learning_rate,
            batch_size=self.config.algorithm.batch_size,
            device=device,
            verbose=0,
        )

        steps_per_round = max(1, self.config.total_timesteps // simple_cfg.num_rounds)
        start_time = time.monotonic()

        for round_idx in range(simple_cfg.num_rounds):
            round_start = time.monotonic()

            # --- (a) Collect real data with current policy ---
            real_steps = self._collect_real_data(
                real_env, policy, buffer, steps_per_round
            )

            # --- (b) Train world model on all real data ---
            if buffer.size >= simple_cfg.world_model_batch_size:
                losses = train_world_model(
                    world_model,
                    buffer,
                    epochs=simple_cfg.world_model_epochs,
                    batch_size=simple_cfg.world_model_batch_size,
                    lr=simple_cfg.world_model_lr,
                    device=device,
                )
                obs_mse, reward_mse = validate_world_model(
                    world_model,
                    buffer,
                    batch_size=min(simple_cfg.world_model_batch_size, buffer.size),
                    device=device,
                    quality_threshold=simple_cfg.quality_threshold,
                )
            else:
                losses = []
                obs_mse, reward_mse = 0.0, 0.0
                self._logger.warning(
                    "simple_skip_wm_train",
                    {"reason": "buffer too small", "buffer_size": buffer.size},
                )

            # --- (c+d) Train PPO on DreamEnv (synthetic data) ---
            synthetic_steps = simple_cfg.synthetic_ratio * real_steps
            if synthetic_steps > 0 and buffer.size > 0:
                dream_env = DreamEnv(
                    world_model=world_model,
                    buffer=buffer,
                    num_actions=num_actions,
                    rollout_horizon=simple_cfg.rollout_horizon,
                    device=device,
                    real_action_space=real_env.action_space,
                    real_obs_space=real_env.observation_space,
                )
                dream_vec = DummyVecEnv([lambda: Monitor(dream_env)])
                policy.set_env(dream_vec)
                policy.learn(total_timesteps=synthetic_steps, reset_num_timesteps=False)
                # Switch back to real env for next round's data collection
                policy.set_env(
                    DummyVecEnv([lambda: real_env])
                    if not hasattr(real_env, "num_envs")
                    else real_env
                )

            # --- (e) Log round metrics ---
            round_elapsed = time.monotonic() - round_start
            self._logger.info(
                "simple_round",
                {
                    "round": round_idx + 1,
                    "real_steps": real_steps,
                    "synthetic_steps": synthetic_steps,
                    "buffer_size": buffer.size,
                    "wm_loss": losses[-1] if losses else None,
                    "obs_mse": obs_mse,
                    "reward_mse": reward_mse,
                    "round_seconds": round(round_elapsed, 1),
                },
            )

        # Save final model + summary
        model_path = os.path.join(self.config.output_dir, "final_model")
        policy.save(model_path)

        self._metrics.flush_csv()
        self._metrics.write_summary()

        wall_clock = time.monotonic() - start_time
        self._logger.info(
            "simple_complete",
            {
                "wall_clock_seconds": round(wall_clock, 1),
                "model_path": model_path + ".zip",
            },
        )

        real_env.close()
        return Path(model_path + ".zip")

    def _collect_real_data(
        self, env, policy, buffer: TransitionBuffer, num_steps: int
    ) -> int:
        """Collect transitions from the real environment using the policy.

        Returns the number of steps actually collected.
        """
        obs, info = env.reset()
        collected = 0

        for _ in range(num_steps):
            action, _ = policy.predict(obs, deterministic=False)
            action_val = (
                int(action) if np.isscalar(action) or action.ndim == 0 else action
            )
            next_obs, reward, terminated, truncated, info = env.step(action_val)
            done = terminated or truncated

            # Flatten MultiDiscrete action to single int for world model
            if np.isscalar(action_val) or (hasattr(action_val, 'ndim') and action_val.ndim == 0):
                action_int = int(action_val)
            else:
                # MultiDiscrete: encode as flat index
                nvec = env.action_space.nvec if hasattr(env.action_space, 'nvec') else None
                if nvec is not None:
                    action_int = 0
                    multiplier = 1
                    for i in reversed(range(len(action_val))):
                        action_int += int(action_val[i]) * multiplier
                        multiplier *= int(nvec[i])
                else:
                    action_int = int(action_val[0])

            # Transpose (H,W,C) → (C,H,W) for world model buffer
            obs_chw = np.transpose(obs, (2, 0, 1)) if obs.ndim == 3 else obs
            next_obs_chw = np.transpose(next_obs, (2, 0, 1)) if next_obs.ndim == 3 else next_obs
            buffer.add(obs_chw, action_int, float(reward), next_obs_chw, done)
            collected += 1

            if self._metrics is not None:
                if done:
                    ep_reward = info.get("episode", {}).get("r", reward)
                    ep_length = info.get("episode", {}).get("l", 1)
                    self._metrics.record_episode(float(ep_reward), int(ep_length), info)

            if done:
                obs, info = env.reset()
            else:
                obs = next_obs

        return collected
