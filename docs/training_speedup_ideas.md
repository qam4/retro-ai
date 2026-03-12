# Training Speedup Ideas — Brainstorm

Ideas for reducing the number of iterations needed to train competent agents, and for making each iteration faster.

## A. Faster Training Loop (wall-clock speed)

### A1. SBX (SB3 + JAX)
- Drop-in replacement for SB3 using JAX instead of PyTorch
- JAX JIT-compiles the entire training loop
- Reported 10-50x speedup for Atari-scale problems
- Almost zero code changes: `from sbx import PPO`
- Effort: low | Impact: high

### A2. EnvPool
- C++ vectorized environment pool, replaces Python SubprocVecEnv
- Zero Python overhead for env stepping
- Our emulator is already C++, natural fit for an adapter
- Could give 5-10x throughput on the env side
- Effort: medium | Impact: high

### A3. Async Training (IMPALA-style)
- Decouple env stepping from model updates
- Actors collect experience in parallel, learner updates asynchronously
- Better GPU utilization, no waiting for slowest env
- Effort: high | Impact: medium-high

### A4. Custom Lean PPO (CleanRL-style)
- Single-file PPO without SB3 abstraction layers
- Tailored to our specific use case
- Smaller gains than SBX but more control
- Effort: medium | Impact: low-medium

## B. Better Sample Efficiency (fewer iterations needed)

### B1. Prioritized Experience Replay (PER)
- Replay important transitions more often
- Requires off-policy algorithm (DQN, not PPO)
- Could switch to DQN + PER for sample efficiency
- Effort: medium | Impact: medium

### B2. Reward Clipping
- Clip rewards to [-1, +1], normalizes reward scale
- Stabilizes training, standard in Atari benchmarks
- Effort: low | Impact: low-medium

### B3. Sticky Actions
- With probability p (0.25), repeat previous action
- Prevents memorization of deterministic games
- We have random noop starts but this adds per-step stochasticity
- Effort: low | Impact: medium (for deterministic games)

### B4. Curiosity-Driven Exploration (ICM/RND)
- Intrinsic reward for visiting novel states
- Random Network Distillation (RND) is simpler than ICM
- Key for sparse reward games like Satellite Attack
- Effort: medium | Impact: high (for sparse reward games)

### B5. Epsilon-Greedy Exploration
- With probability epsilon, take random action
- Standard in DQN, less common in PPO (uses entropy bonus instead)
- Effort: low | Impact: low-medium

### B6. Target Networks
- Stabilize value function estimation
- Built into SB3 DQN already
- Relevant if we switch from PPO to DQN
- Effort: low | Impact: medium

## C. Learning from Less Data (sample efficiency)

### C1. SimPLe (Simulated Policy Learning)
- Learn a world model from real experience
- Generate synthetic experience from the world model
- Train policy on synthetic + real data
- Kaiser et al. 2020: 100k real steps to match 10M model-free steps on Atari
- Effort: high | Impact: very high

### C2. Supervised Pre-training / Behavioral Cloning
- Record human gameplay, train policy to imitate
- Use as initialization before RL fine-tuning
- Requires human demonstrations
- Effort: medium | Impact: medium-high

### C3. Data Augmentation (DrQ)
- Random crops, color jitter on observations
- DrQ showed big gains with simple augmentation
- Works with both PPO and DQN
- Effort: low | Impact: medium

### C4. Efficient Replay Buffers
- For off-policy methods (DQN): better buffer management
- Hindsight Experience Replay (HER) for goal-conditioned tasks
- Not directly applicable to PPO (on-policy)
- Effort: medium | Impact: medium

## D. GPU / Hardware (not currently available)

### D1. GPU Training
- Move neural network to CUDA
- 10-100x faster for CNN forward/backward passes
- mixed_precision support already implemented
- Blocked by: no GPU on current dev instance
- Effort: low (just need hardware) | Impact: very high

### D2. Multiple GPUs / Distributed
- Data-parallel training across GPUs
- Effort: high | Impact: high (with hardware)

## Priority Ranking (CPU-only, no GPU)

1. **SBX (A1)** — easiest win, one import change, 10x+ potential
2. **Sticky actions (B3)** — low effort, helps with deterministic games
3. **Curiosity/RND (B4)** — medium effort, could unlock sparse reward games
4. **Data augmentation (C3)** — low effort, proven gains
5. **SimPLe (C1)** — high effort but game-changing sample efficiency
6. **EnvPool (A2)** — medium effort, big throughput gains
7. **DQN + PER (B1)** — alternative to PPO for better sample efficiency
