#!/usr/bin/env python3
"""Quick test of the native module and emulator."""
import sys
import os

print("Python:", sys.executable)
print("Testing native module import...")
sys.stdout.flush()

try:
    import retro_ai_native as n
    print("OK: retro_ai_native imported")
    print("  Symbols:", [s for s in dir(n) if not s.startswith("_")])
except Exception as e:
    print(f"FAIL: {e}")
    sys.exit(1)

rom_dir = os.environ.get("RETRO_AI_ROM_DIR", "roms")
rom = os.path.join(rom_dir, "videopac", "Course de Voitures + Autodrome + Cryptogramme (1980)(Philips)(FR).bin")
bios = os.path.join(rom_dir, "videopac", "Philips C52 BIOS (19xx)(Philips)(FR).bin")

print(f"\nROM:  {rom}  exists={os.path.exists(rom)}")
print(f"BIOS: {bios}  exists={os.path.exists(bios)}")
sys.stdout.flush()

if not os.path.exists(rom) or not os.path.exists(bios):
    print("FAIL: ROM or BIOS file missing")
    sys.exit(1)

print("\nCreating VideopacRLInterface...")
sys.stdout.flush()
try:
    v = n.VideopacRLInterface(bios, rom)
    print("OK: interface created")
except Exception as e:
    print(f"FAIL creating interface: {e}")
    sys.exit(1)

print("Calling reset()...")
sys.stdout.flush()
try:
    result = v.reset()
    print(f"OK: reset returned obs size={len(result.observation)}, reward={result.reward}, done={result.done}")
except Exception as e:
    print(f"FAIL reset: {e}")
    sys.exit(1)

print("Calling step(0)...")
sys.stdout.flush()
try:
    result = v.step(0)
    print(f"OK: step returned obs size={len(result.observation)}, reward={result.reward}, done={result.done}")
except Exception as e:
    print(f"FAIL step: {e}")
    sys.exit(1)

print("\nAll tests passed!")
