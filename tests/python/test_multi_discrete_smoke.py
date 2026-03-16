"""Smoke test for multi-discrete and discrete action modes.

Validates Requirements R1, R2, R3, R5, R6, R7.
"""

import os
import sys

import pytest

# Add build dir and python source to path
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "build", "ci-linux")
)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

try:
    import retro_ai_native as native
except ImportError:
    pytest.skip("retro_ai_native not available (no C++ build)", allow_module_level=True)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BIOS = os.path.join(ROOT, "roms/videopac/BIOS/bios_O2rom.bin")
ROM = os.path.join(ROOT, "roms/videopac/ROMS/ntsc_moto-crash.bin")

passed = 0
failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  PASS: {name}")
        passed += 1
    else:
        print(f"  FAIL: {name} — {detail}")
        failed += 1


# =========================================================================
# 1. Multi-discrete mode (default)
# =========================================================================
print("\n=== Multi-Discrete Mode (default) ===")

env = native.VideopacRLInterface(BIOS, ROM)
env.reset()

# R1: action_space returns MULTI_DISCRETE with shape [2,2,2,2,2]
aspace = env.action_space()
check(
    "action_space type is MULTI_DISCRETE",
    aspace.type == native.ActionType.MULTI_DISCRETE,
    f"got {aspace.type}",
)
check(
    "action_space shape is [2,2,2,2,2]",
    aspace.shape == [2, 2, 2, 2, 2],
    f"got {aspace.shape}",
)

# R1: NOOP step succeeds
result = env.step([0, 0, 0, 0, 0])
check(
    "NOOP [0,0,0,0,0] succeeds (not truncated)",
    not result.truncated,
    f"truncated={result.truncated}",
)

# R1: Diagonal up+left
result = env.step([1, 0, 1, 0, 0])
check(
    "Diagonal up+left [1,0,1,0,0] succeeds",
    not result.truncated,
    f"truncated={result.truncated}",
)

# R1: Diagonal down+right+fire
result = env.step([0, 1, 0, 1, 1])
check(
    "Diagonal down+right+fire [0,1,0,1,1] succeeds",
    not result.truncated,
    f"truncated={result.truncated}",
)

# R6: Invalid value (2 instead of 0/1) → truncated
result = env.step([1, 2, 0, 0, 0])
check(
    "Invalid value [1,2,0,0,0] → truncated=True",
    result.truncated,
    f"truncated={result.truncated}",
)

# R6: Wrong size → truncated
result = env.step([1, 0, 0])
check(
    "Wrong size [1,0,0] → truncated=True",
    result.truncated,
    f"truncated={result.truncated}",
)

# R7: step_n with multi-discrete
result = env.step_n([1, 0, 0, 1, 0], 4)
check(
    "step_n([1,0,0,1,0], 4) succeeds",
    not result.truncated,
    f"truncated={result.truncated}",
)


# =========================================================================
# 2. Discrete mode
# =========================================================================
print("\n=== Discrete Mode ===")

env2 = native.VideopacRLInterface(BIOS, ROM, "survival", 0, {}, "discrete")
env2.reset()

# R2: action_space returns DISCRETE with shape [18]
aspace2 = env2.action_space()
check(
    "action_space type is DISCRETE",
    aspace2.type == native.ActionType.DISCRETE,
    f"got {aspace2.type}",
)
check("action_space shape is [18]", aspace2.shape == [18], f"got {aspace2.shape}")

# R2: NOOP
result2 = env2.step([0])
check(
    "Discrete NOOP [0] succeeds",
    not result2.truncated,
    f"truncated={result2.truncated}",
)

# R2: Up
result2 = env2.step([1])
check(
    "Discrete Up [1] succeeds", not result2.truncated, f"truncated={result2.truncated}"
)

# R7: step_n with discrete
result2 = env2.step_n([4], 4)
check(
    "step_n([4], 4) discrete succeeds",
    not result2.truncated,
    f"truncated={result2.truncated}",
)


# =========================================================================
# 3. Invalid action_mode → InitializationError
# =========================================================================
print("\n=== Invalid action_mode ===")

try:
    bad = native.VideopacRLInterface(BIOS, ROM, "survival", 0, {}, "invalid")
    check(
        "Invalid action_mode raises InitializationError", False, "no exception raised"
    )
except native.InitializationError:
    check("Invalid action_mode raises InitializationError", True)
except Exception as e:
    check(
        "Invalid action_mode raises InitializationError",
        False,
        f"wrong exception: {type(e).__name__}: {e}",
    )


# =========================================================================
# Summary
# =========================================================================
print(f"\n{'='*50}")
print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
if failed:
    sys.exit(1)
else:
    print("All smoke tests passed!")
