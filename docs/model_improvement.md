# Model Improvement — Sample Efficiency & Agent Quality

Ideas for training better agents with fewer timesteps. These don't make
each step faster — they make each step count more.

## Implemented

### Reward Clipping (B2) ✅
- Clip rewards to [-1, +1], normalizes reward scale
- Stabilizes training, standard in Atari benchmarks
- Config: `reward_clip: 1.0`
- Effort: low | Impact: low-medium

### Sticky Actions (B3) ✅
- With probability p (0.25), repeat previous action
- Prevents memorization of deterministic games
- We have random noop starts but this adds per-step stochasticity
- Config: `sticky_actions: 0.25`
- Effort: low | Impact: medium (for deterministic games)

## Not Yet Tested

### Prioritized Experience Replay (B1)
- Replay important transitions more often
- Requires off-policy algorithm (DQN, not PPO)
- Could switch to DQN + PER for sample efficiency
- Effort: medium | Impact: medium

### Curiosity-Driven Exploration / RND (B4)
- Intrinsic reward for visiting novel states
- Random Network Distillation (RND) is simpler than ICM
- Key for sparse reward games like Satellite Attack
- Effort: medium | Impact: high (for sparse reward games)

### Epsilon-Greedy Exploration (B5)
- With probability epsilon, take random action
- Standard in DQN, less common in PPO (uses entropy bonus instead)
- Effort: low | Impact: low-medium

### Target Networks (B6)
- Stabilize value function estimation
- Built into SB3 DQN already
- Relevant if we switch from PPO to DQN
- Effort: low | Impact: medium

### SimPLe — Simulated Policy Learning (C1)
- Learn a world model from real experience
- Generate synthetic experience from the world model
- Train policy on synthetic + real data
- Kaiser et al. 2020: 100k real steps to match 10M model-free on Atari
- Effort: high | Impact: very high

### Supervised Pre-training / Behavioral Cloning (C2)
- Record human gameplay, train policy to imitate
- Use as initialization before RL fine-tuning
- Requires human demonstrations
- Effort: medium | Impact: medium-high

### Data Augmentation / DrQ (C3)
- Random crops, color jitter on observations
- DrQ showed big gains with simple augmentation
- Works with both PPO and DQN
- Effort: low | Impact: medium

### Efficient Replay Buffers (C4)
- For off-policy methods (DQN): better buffer management
- Hindsight Experience Replay (HER) for goal-conditioned tasks
- Not directly applicable to PPO (on-policy)
- Effort: medium | Impact: medium

## Priority Ranking

1. **Sticky actions (B3)** — already implemented ✅
2. **Reward clipping (B2)** — already implemented ✅
3. **Curiosity/RND (B4)** — medium effort, could unlock sparse reward games
4. **Data augmentation (C3)** — low effort, proven gains
5. **DQN + PER (B1)** — alternative to PPO for better sample efficiency
6. **SimPLe (C1)** — high effort but game-changing sample efficiency
7. **Behavioral cloning (C2)** — needs human demos, good for bootstrapping
