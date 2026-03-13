# Training Speed — Wall-Clock Optimizations

How to make each training iteration faster (more timesteps per second).

## Benchmark Results

All benchmarks run on the same EC2 instance with 10,000 timesteps,
PPO CnnPolicy, Course Automobile game profile. Reproducible via:

```bash
RETRO_AI_ROM_DIR=roms PYTHONPATH=python:build/ci-linux \
  python3.9 scripts/benchmark_speedup.py --name <name> [options]
```

Results accumulate in `benchmarks/speedup_results.json`.

Hardware: NVIDIA A10G (23 GB), 8 vCPUs, Python 3.9,
PyTorch 2.8 + CUDA 12.8, JAX 0.4.30 + CUDA 12.

### Round 1: GPU + Parallel Environments (2026-03-13)

| Configuration              | FPS   | Speedup | Wall clock |
|----------------------------|-------|---------|------------|
| CPU baseline (1 env)       | 34.1  | 1.0x    | 293s       |
| GPU only (1 env)           | 39.2  | 1.15x   | 255s       |
| GPU + FP16 (1 env)         | 39.6  | 1.16x   | 253s       |
| GPU + FP16 + 4 envs        | 117.9 | 3.46x   | 85s        |
| GPU + FP16 + 8 envs        | 160.0 | 4.69x   | 63s        |

Findings:
- With a single env, the bottleneck is the C++ emulator (CPU-bound).
  Moving the neural net to GPU gives only ~15% improvement.
- Mixed precision (FP16) adds negligible benefit at this model size.
- Parallel envs via SubprocVecEnv are the real win: the GPU batches
  inference across all envs, and emulator stepping runs in parallel
  across CPU cores.
- 8 envs on 8 vCPUs hits ~4.7x. Diminishing returns expected beyond
  the CPU core count.

### Round 2: SBX / JAX (2026-03-13)

| Configuration              | FPS   | Speedup | Wall clock |
|----------------------------|-------|---------|------------|
| SB3 PPO, GPU, 1 env       | 39.2  | 1.15x   | 255s       |
| SBX PPO, GPU, 1 env       | 38.2  | 1.12x   | 262s       |
| SB3 PPO, GPU+FP16, 4 envs | 117.9 | 3.46x   | 85s        |
| SBX PPO, GPU, 4 envs      | 103.9 | 3.05x   | 96s        |
| SB3 PPO, GPU+FP16, 8 envs | 160.0 | 4.69x   | 63s        |
| SBX PPO, GPU, 8 envs      | 137.6 | 4.04x   | 73s        |

Findings:
- SBX (JAX) does NOT provide a speedup over SB3 (PyTorch) for our
  workload. In fact it's slightly slower across all configurations.
- The "10-50x" speedup reported for SBX is for pure-RL benchmarks
  where the environment is trivially fast (e.g. Brax physics). In our
  case the emulator is the bottleneck, not the RL framework.
- Verdict: SBX is not worth the added dependency. Stick with SB3.

### Round 3: ThreadedVecEnv vs SubprocVecEnv (2026-03-13)

Our C++ emulator releases the GIL during step/reset, so threads can
run emulators in parallel without subprocess IPC overhead.

| Configuration                     | FPS   | Speedup | Wall clock |
|-----------------------------------|-------|---------|------------|
| SubprocVecEnv, GPU+FP16, 4 envs  | 117.9 | 3.46x   | 85s        |
| ThreadedVecEnv, GPU+FP16, 4 envs | 115.0 | 3.37x   | 87s        |
| SubprocVecEnv, GPU+FP16, 8 envs  | 160.0 | 4.69x   | 63s        |
| ThreadedVecEnv, GPU+FP16, 8 envs | 165.5 | 4.85x   | 60s        |
| SubprocVecEnv, GPU+FP16, 16 envs | 146.2 | 4.29x   | 68s        |
| ThreadedVecEnv, GPU+FP16, 16 envs| 163.2 | 4.79x   | 61s        |

Findings:
- At 4 envs: roughly equal (SubprocVecEnv slightly ahead).
- At 8 envs: ThreadedVecEnv edges ahead by ~3% (165 vs 160 FPS).
- At 16 envs: ThreadedVecEnv wins clearly (163 vs 146 FPS, +12%).
  SubprocVecEnv actually regresses past 8 envs due to IPC overhead
  exceeding the CPU core count. ThreadedVecEnv holds steady.
- ThreadedVecEnv avoids process spawning and observation serialization,
  which matters more as env count grows beyond CPU cores.
- Verdict: ThreadedVecEnv is the better default for our GIL-releasing
  emulator, especially at higher env counts.

### Summary Table

| Configuration                              | FPS   | vs baseline |
|--------------------------------------------|-------|-------------|
| CPU baseline (1 env)                       | 34.1  | 1.0x        |
| SB3 + GPU + FP16 + 8 envs (threaded)      | 165.5 | 4.85x       |
| **SB3 + GPU + FP16 + 8 envs + PGO**       | 191.0 | **5.60x**   |

Best configuration: SB3 PPO + CUDA + FP16 + 8 envs + ThreadedVecEnv + PGO.

Best configuration: SB3 PPO + CUDA + FP16 + 8 envs + ThreadedVecEnv.

### Round 5: PGO (Profile-Guided Optimization) (2026-03-13)

Two-pass build: instrument with `-fprofile-generate`, run 5000 steps to
collect branch/call data, rebuild with `-fprofile-use`.

| Configuration                          | FPS   | vs baseline | vs non-PGO |
|----------------------------------------|-------|-------------|------------|
| Non-PGO, GPU+FP16+8env (threaded)     | 165.5 | 4.85x       | —          |
| **PGO, GPU+FP16+8env (threaded)**     | 191.0 | **5.60x**   | **+15.4%** |

PGO gives a clean 15% improvement by letting the compiler optimize
branch prediction and code layout for the emulator's actual hot paths
(VDC pixel rendering, MasterClock tick, CPU instruction dispatch).

Build instructions:
```bash
# Pass 1: instrumented build
cmake --preset ci-linux -DPGO_GENERATE=ON [python flags]
cmake --build --preset ci-linux
# Run representative workload
python3.9 -c "... run 5000 steps ..."
# Pass 2: optimized build
cmake --preset ci-linux -DPGO_USE=ON [python flags]
cmake --build --preset ci-linux
```

### Round 4: Where's the Bottleneck? (2026-03-13)

Per-step cost breakdown (single env, framebuffer mode):

| Component | Time | Notes |
|-----------|------|-------|
| C++ emulator (4 frames) | 18.8ms | frame_skip=4, ~4.7ms per frame |
| Python preprocessing | 0.5ms | grayscale + resize + frame_stack |
| **Total per agent step** | **19.3ms** | = ~52 agent FPS single-threaded |

The Python preprocessing (grayscale, resize, frame_stack) is negligible.
The C++ emulator dominates the per-step cost.

Raw env throughput (no neural network, full preprocessing pipeline):

| Parallel envs | Agent FPS | Emulator FPS | Scaling |
|---------------|-----------|--------------|---------|
| 1             | 55        | 219          | 1.0x    |
| 4             | 216       | 862          | 3.9x   |
| 8             | 289       | 1157         | 5.3x   |

**Training throughput vs env ceiling (8 envs, framebuffer):**

```
Env ceiling (with preprocessing):  289 agent FPS
Training throughput:               165 agent FPS
NN overhead:                       43%
```

The split is ~57% emulator / 43% NN. Both are significant contributors.
To go faster: optimize the C++ emulator hot paths, or reduce NN overhead
with async training.

Reproduce with:
```bash
RETRO_AI_ROM_DIR=roms PYTHONPATH=python:build/ci-linux \
  python3.9 scripts/benchmark_emulator.py --steps 1000 --max-envs 16
```

### Round 6: torch.compile() (2026-03-13)

PyTorch 2.x `torch.compile()` JIT-compiles the policy network.

| Configuration (PGO + GPU + FP16 + 8 threaded envs) | 10k FPS | 50k FPS |
|-----------------------------------------------------|---------|---------|
| Without torch.compile                               | 191     | 163     |
| With torch.compile                                  | 141     | 154     |

Findings:
- Steady-state FPS during rollout collection is slightly higher with
  compile (~230 vs ~250), but JIT compilation overhead (first steps +
  periodic recompilation) drags the average down.
- At 10k steps: 26% slower due to upfront JIT cost.
- At 50k steps: 6% slower, still not amortized.
- Would likely break even around 200k+ steps, but the gain is marginal.
- Verdict: not worth it for our model size and typical run lengths.


```bash
# GPU + mixed precision + threaded parallel envs
RETRO_AI_ROM_DIR=roms PYTHONPATH=python:build/ci-linux \
  python scripts/benchmark_speedup.py --name my-run \
    --device cuda --mixed-precision --num-envs 8 --vec-env-type threaded
```

Config fields: `device` ("auto"/"cuda"/"cpu"), `mixed_precision` (bool),
`num_envs` (int), `vec_env_type` ("subproc"/"threaded"),
`algorithm.name` ("PPO"/"DQN"/"SBX_PPO"/"SBX_DQN").

## Ideas — Not Yet Benchmarked

### EnvPool — Priority: low (revised)
- C++ vectorized environment pool, replaces Python SubprocVecEnv
- Our ThreadedVecEnv already eliminates most IPC overhead since the
  emulator releases the GIL. EnvPool's main advantage (C++ env pool)
  is less relevant when the env is already C++ with GIL release.
- Effort: medium | Expected impact: low (given ThreadedVecEnv results)

### Async Training (IMPALA-style) — Priority: medium
- Decouple env stepping from model updates
- Actors collect experience in parallel, learner updates asynchronously
- Better GPU utilization, no waiting for slowest env
- Effort: high | Expected impact: medium-high

### Custom Lean PPO (CleanRL-style) — Priority: low
- Single-file PPO without SB3 abstraction layers
- Tailored to our specific use case
- Effort: medium | Expected impact: low-medium

### Multiple GPUs / Distributed — Priority: low
- Data-parallel training across GPUs
- Only relevant if we add more GPUs
- Effort: high | Expected impact: high (with hardware)

## Priority Ranking

1. **GPU + parallel envs** — proven 4.85x with 8 envs ✅
2. **ThreadedVecEnv** — slight edge over SubprocVecEnv, better at 16+ envs ✅
3. **PGO** — +15% from profile-guided optimization of C++ emulator ✅
4. ~~SBX~~ — tested, no benefit for emulator-bound workloads ❌
5. ~~torch.compile~~ — tested, JIT overhead exceeds gains at typical run lengths ❌
6. ~~EnvPool~~ — deprioritized, ThreadedVecEnv covers the same ground
7. **IMPALA / async training** — 43% of time is NN, async could overlap with env stepping
