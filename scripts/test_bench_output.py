#!/usr/bin/env python3
"""Quick test to see what bench_training.py outputs."""
import os
import subprocess
import sys

env = os.environ.copy()
result = subprocess.run(
    [sys.executable, "scripts/bench_training.py",
     "--game-profile", "course_automobile",
     "--timesteps", "100"],
    capture_output=True, text=True, env=env,
)
with open("bench_output_test.txt", "w") as f:
    f.write(f"RETURNCODE: {result.returncode}\n")
    f.write(f"STDOUT:\n{result.stdout}\n")
    f.write(f"STDERR:\n{result.stderr}\n")
print("Written to bench_output_test.txt")
