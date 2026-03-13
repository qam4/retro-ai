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

### GPU + Parallel Environments (2026-03-13)

Hardware: NVIDIA A10G (23 GB), 8 vCPUs, PyTorch 2.8 + CUDA 12.8

| Configuration              | FPS   | Speedup | Wall clock |
|----------------------------|-------|---------|------------|
| CPU baseline (1 env)       | 34.1  | 1.0x    | 293s       |
| GPU only (1 env)           | 39.2  | 1.15x   | 255s       |
| GPU + FP16 (1 env)         | 39.6  | 1.16x   | 253s       |
| GPU + FP16 + 4 envs        | 117.9 | 3.46x   | 85s        |
| GPU + FP16 + 8 envs        | 160.0 | 4.69x   | 63s        |

Key findings:
- With a single env, the bottleneck is the C++ emulator (CPU-bound).
  Moving the neural net to GPU gives only ~15% improvement.
- Mixed precision (FP16) adds negligible benefit at this model size.
- Parallel envs via SubprocVecEnv are the real win: the GPU batches
  inference across all envs, and emulator stepping runs in parallel
  across CPU cores.
- 8 envs on 8 vCPUs hits ~4.7x. Diminishing returns expected beyond
  the CPU core count.

### How to use

```bash
# GPU + mixed precision + 4 parallel envs
python examples/gpu_training.py --rom roms/game.bin --bios roms/bios.bin --num-envs 4

# Or via YAML config
python -m retro_ai.training.cli train examples/configs/gpu_training.yaml
```

Config fields: `device` ("auto"/"cuda"/"cpu"), `mixed_precision` (bool),
`num_envs` (int).


## Ideas — Not Yet Benchmarked

### SBX (SB3 + JAX) — Priority: high
- Drop-in replacement for SB3 using JAX instead of PyTorch
- JAX JIT-compiles the entire training loop (reported 10-50x for Atari)
- Almost zero code changes: `from sbx import PPO`
- Effort: low | Expected impact: high
- Status: not yet tested

### EnvPool — Priority: medium
- C++ vectorized environment pool, replaces Python SubprocVecEnv
- Zero Python overhead for env stepping
- Our emulator is already C++, natural fit for an adapter
- Could give 5-10x throughput on the env side
- Effort: medium | Expected impact: high
- Status: not yet tested

### Async Training (IMPALA-style) — Priority: low
- Decouple env stepping from model updates
- Actors collect experience in parallel, learner updates asynchronously
- Better GPU utilization, no waiting for slowest env
- Effort: high | Expected impact: medium-high
- Status: not yet tested

### Custom Lean PPO (CleanRL-style) — Priority: low
- Single-file PPO without SB3 abstraction layers
- Tailored to our specific use case
- Smaller gains than SBX but more control
- Effort: medium | Expected impact: low-medium
- Status: not yet tested

### Multiple GPUs / Distributed — Priority: low
- Data-parallel training across GPUs
- Only relevant if we add more GPUs
- Effort: high | Expected impact: high (with hardware)
- Status: not yet tested

## Priority Ranking

1. **Parallel envs (D1)** — already proven, 4.7x with 8 envs ✅
2. **SBX** — easiest next win, one import change, 10x+ potential
3. **EnvPool** — medium effort, big throughput gains
4. **IMPALA** — high effort, better GPU utilization
5. **CleanRL PPO** — more control, smaller gains
