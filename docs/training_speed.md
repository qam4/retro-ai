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
| SB3 PPO, GPU, 1 env       | 39.2  | 1.0x*   | 255s       |
| SBX PPO, GPU, 1 env       | 38.2  | 0.97x*  | 262s       |
| SB3 PPO, GPU+FP16, 4 envs | 117.9 | 3.46x   | 85s        |
| SBX PPO, GPU, 4 envs      | 103.9 | 3.05x   | 96s        |
| SB3 PPO, GPU+FP16, 8 envs | 160.0 | 4.69x   | 63s        |
| SBX PPO, GPU, 8 envs      | 137.6 | 4.04x   | 73s        |

*Speedup relative to CPU baseline (34.1 FPS).

Findings:
- SBX (JAX) does NOT provide a speedup over SB3 (PyTorch) for our
  workload. In fact it's slightly slower across all configurations.
- The "10-50x" speedup reported for SBX is for pure-RL benchmarks
  where the environment is trivially fast (e.g. Brax physics). In our
  case the emulator is the bottleneck, not the RL framework.
- JAX JIT compilation overhead is visible in the first few steps but
  amortizes quickly. Even after warmup, no advantage over PyTorch.
- Verdict: SBX is not worth the added dependency for our use case.
  Stick with SB3 + PyTorch.

### Summary Table

| Configuration                  | FPS   | vs baseline |
|--------------------------------|-------|-------------|
| CPU baseline (1 env)           | 34.1  | 1.0x        |
| **SB3 + GPU + FP16 + 8 envs** | 160.0 | **4.69x**   |
| SBX + GPU + 8 envs             | 137.6 | 4.04x       |

Best configuration so far: SB3 PPO + CUDA + FP16 + 8 parallel envs.

### How to use

```bash
# GPU + mixed precision + parallel envs
python examples/gpu_training.py --rom roms/game.bin --bios roms/bios.bin --num-envs 8

# Or via YAML config
python -m retro_ai.training.cli train examples/configs/gpu_training.yaml
```

Config fields: `device` ("auto"/"cuda"/"cpu"), `mixed_precision` (bool),
`num_envs` (int), `algorithm.name` ("PPO"/"DQN"/"SBX_PPO"/"SBX_DQN").

## Ideas — Not Yet Benchmarked

### EnvPool — Priority: high (next)
- C++ vectorized environment pool, replaces Python SubprocVecEnv
- Zero Python overhead for env stepping
- Our emulator is already C++, natural fit for an adapter
- Could give 5-10x throughput on the env side
- Effort: medium | Expected impact: high

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

1. **GPU + parallel envs** — proven 4.7x with 8 envs ✅
2. ~~SBX~~ — tested, no benefit for emulator-bound workloads ❌
3. **EnvPool** — next candidate, could reduce env stepping overhead
4. **IMPALA** — high effort, better GPU utilization
5. **CleanRL PPO** — more control, smaller gains
