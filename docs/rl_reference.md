# RL for Retro Games — Reference & Best Practices

Lessons learned from Atari RL research, applied to Videopac/retro game training.

## Key Papers

- **Nature DQN** (Mnih et al. 2015) — "Human-level control through deep reinforcement learning"
  https://www.nature.com/articles/nature14236
  Origin of: NatureCNN architecture, frame skip/stack, 84x84 grayscale, reward clipping, random noop starts.

- **ALE Revisited** (Machado et al. 2018) — "Revisiting the Arcade Learning Environment"
  https://arxiv.org/abs/1709.06009
  Added: sticky actions, determinism analysis, episode life tracking. Documents all the gotchas.

- **PPO** (Schulman et al. 2017) — "Proximal Policy Optimization Algorithms"
  https://arxiv.org/abs/1707.06347
  The algorithm we use via Stable-Baselines3.

- **Implementation Matters** (Engstrom et al. 2020) — "Implementation Matters in Deep RL"
  https://arxiv.org/abs/2005.12729
  Documents which implementation details actually matter for PPO performance.

- **37 PPO Details** (Costa Huang 2022) — Blog post
  https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
  Practical walkthrough of every PPO trick. Highly recommended.

- **OpenAI Retro Contest** (Nichol et al. 2018) — "Gotta Learn Fast"
  https://arxiv.org/abs/1804.03720
  Training on Sonic with sparse rewards. Relevant to our reward shaping challenges.

## Standard Environment Tricks

### Already Implemented
- **Frame skipping** (frame_skip=4): Execute same action for N frames, return accumulated reward.
- **Frame stacking** (frame_stack=4): Stack last N observations as channels for temporal info.
- **Grayscale + resize** (84x84): Standard preprocessing for CNN policies.
- **Random noop starts** (random_noop_max=30): Random 0-30 noop frames after reset to break determinism.
- **Episode termination on score drop**: End episode when game score resets (death detection).

### Not Yet Implemented
- **Sticky actions**: With probability p (typically 0.25), repeat previous action instead of new one. Prevents frame-perfect memorization. Standard in ALE v5.
- **Reward clipping**: Clip rewards to [-1, +1]. Normalizes reward scale across games.
- **Frame max-pooling**: Pixel-wise max of 2 consecutive frames. Handles sprite flickering.
- **Fire on reset**: Some games need fire pressed to start after life loss.
- **Episodic life**: Treat each life as a separate episode for faster learning signal.

## Reward Strategies

### What Works
- **Score delta** (memory reward): Read score from RAM, return change per frame. Best when score changes are frequent (e.g. Course Automobile: score increases every frame while moving up).
- **Survival bonus**: +1 per frame alive. Dense signal, good for learning basic survival. Can plateau if agent finds a safe corner.
- **Combined**: Score delta + survival bonus. Dense baseline signal plus sparse scoring bonus. Best for games with infrequent scoring events.

### What Doesn't Work
- **Score delta alone with score-reset-on-death**: Episode reward always nets to ~0 because the death frame wipes out accumulated score. Fix: zero out the death frame's negative delta.
- **Survival alone for action games**: Agent learns to hide, not play. Needs scoring incentive.

### Reward Shaping Tips
- Scale rewards so the survival bonus and score bonus are in similar ranges. If score gives +10 per hit but survival gives +1 per frame, the agent will prioritize survival.
- For games with very sparse scoring, consider intrinsic motivation (curiosity-driven reward) as an exploration bonus.
- Reward clipping to [-1, +1] helps PPO's value function estimation but loses magnitude information.

## Action Space Design

### Discrete (18 actions)
Flat enumeration of all joystick+button combos. Simple but can't combine direction+fire independently.

### Multi-Discrete [2,2,2,2,2]
5 independent binary dims (up/down/left/right/fire). Allows simultaneous inputs but includes 14 invalid combos (up+down, left+right).

### Joystick [3,3,2] (recommended)
Axis-based: vertical(3) × horizontal(3) × fire(2) = 18 valid combos. Physically correct joystick model. No wasted exploration on impossible states.

## Training Configuration

### PPO Hyperparameters (SB3 defaults, generally good)
- learning_rate: 3e-4 (lower for fine-tuning: 1e-4)
- n_steps: 2048 (rollout buffer size per env)
- batch_size: 64
- n_epochs: 10
- gamma: 0.99 (discount factor)
- gae_lambda: 0.95
- clip_range: 0.2
- ent_coef: 0.01 (entropy bonus for exploration)

### Environment Configuration
- num_envs: 4-8 (parallel environments, scales throughput linearly)
- frame_skip: 4 (standard, gives ~15 decisions/sec at 60fps)
- frame_stack: 4 (temporal context)
- resize: (84, 84) (standard for NatureCNN)
- grayscale: true (reduces input size 3x)

### Training Duration
- 100k steps: Quick experiment, enough to see if reward signal works
- 500k steps: Decent training for simple games
- 1M+ steps: Full training for complex games
- 10M+ steps: Atari benchmark standard (with GPU)

## Common Pitfalls

1. **Deterministic games**: Without random noop starts, agent memorizes fixed sequences instead of learning to play.
2. **Wrong joystick port**: Videopac games use either port 0 or port 1. If the agent can't move, check joystick_index.
3. **Wrong score addresses**: RAM addresses differ per game. Verify with manual play + RAM watching.
4. **Score format mismatch**: Some games use BCD, others use individual digit bytes, others use binary. Check the ROM disassembly.
5. **Config merge override**: If training config sets reward_mode to the default value ("survival"), the game profile's value wins. Use explicit field tracking.
6. **Hang detection kills long episodes**: As the agent improves, episodes get longer, PPO rollouts take longer. Increase hang timeout.
