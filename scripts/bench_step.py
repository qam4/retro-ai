#!/usr/bin/env python3
"""Quick benchmark to isolate where time goes in a training step.

Usage:
  RETRO_AI_ROM_DIR=... PYTHONPATH=build/ci-linux:python python scripts/bench_step.py
"""

import os
import time
import retro_ai_native

ROM_DIR = os.environ.get("RETRO_AI_ROM_DIR", "roms")
BIOS = os.path.join(ROM_DIR, "videopac/Philips C52 BIOS (19xx)(Philips)(FR).bin")
ROM = os.path.join(ROM_DIR, "videopac/Course de Voitures + Autodrome + Cryptogramme (1980)(Philips)(FR).bin")

WARMUP = 100
N = 200

def bench(label, fn, n=N):
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    elapsed = time.perf_counter() - t0
    fps = n / elapsed
    us = elapsed / n * 1e6
    print(f"  {label:30s}  {elapsed:.3f}s  {fps:.0f} fps  {us:.0f} µs/call")
    return elapsed

def main():
    print("Creating VideopacRLInterface (memory reward)...")
    rl = retro_ai_native.VideopacRLInterface(BIOS, ROM, "memory", 1, reward_params={
        "score_address_count": "1",
        "score_address_0_addr": "54",
        "score_address_0_bytes": "2",
        "score_address_0_bcd": "1",
        "score_address_0_le": "1",
        "timer_minutes_addr": "65",
        "timer_seconds_addr": "66",
        "done_when_timer_zero": "true",
    })
    rl.reset_numpy(-1)
    for _ in range(WARMUP):
        rl.step_numpy([1])

    print(f"\nBenchmarking {N} iterations each:\n")

    # 1. Full step with memory reward
    t_full = bench("step_numpy (memory reward)", lambda: rl.step_numpy([1]))

    # 2. read_ram alone
    t_ram = bench("read_ram()", lambda: rl.read_ram())

    # 3. Switch to survival (no RAM reads) and re-bench
    rl.set_reward_mode("survival")
    t_surv = bench("step_numpy (survival reward)", lambda: rl.step_numpy([1]))

    # 4. Switch to no reward at all — just emulator + framebuffer
    # We can approximate by using survival (reward is trivial)
    # The difference between survival and memory tells us RAM overhead

    print(f"\n--- Analysis ---")
    print(f"  Full step (memory):    {t_full/N*1000:.2f} ms/frame")
    print(f"  Full step (survival):  {t_surv/N*1000:.2f} ms/frame")
    print(f"  RAM read alone:        {t_ram/N*1000:.2f} ms/frame")
    print(f"  Memory reward overhead: {(t_full-t_surv)/N*1000:.2f} ms/frame")
    print(f"  Emulator+FB+Python:    {t_surv/N*1000:.2f} ms/frame")

if __name__ == "__main__":
    main()
