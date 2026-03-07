#!/usr/bin/env python3
"""Dump emulator frames as PNG images for visual inspection."""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build', 'ci-linux'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import retro_ai_native

bios = os.path.join(os.environ.get('RETRO_AI_ROM_DIR', 'roms'),
                    'videopac/Philips C52 BIOS (19xx)(Philips)(FR).bin')
rom = os.path.join(os.environ.get('RETRO_AI_ROM_DIR', 'roms'),
                   'videopac/Course de Voitures + Autodrome + Cryptogramme (1980)(Philips)(FR).bin')

emu = retro_ai_native.VideopacRLInterface(bios, rom, 'survival', 1)

os.makedirs('output/frames', exist_ok=True)

def save_frame(obs_bytes, space, name):
    obs = np.frombuffer(obs_bytes, dtype=np.uint8).reshape(space.height, space.width, space.channels)
    try:
        from PIL import Image
        img = Image.fromarray(obs, 'RGB')
        path = f'output/frames/{name}.png'
        img.save(path)
        print(f'Saved {path} ({space.width}x{space.height})')
    except ImportError:
        print('PIL not available, skipping image save')

space = emu.observation_space()

# Frame 0: right after reset (C++ reset presses Key1 once for game select)
result = emu.reset()
save_frame(bytes(result.observation), space, '01_after_reset')

# Frame ~60: after some time
for _ in range(60):
    result = emu.step([0])
save_frame(bytes(result.observation), space, '02_after_60_noop')

# Press Key1 (level select) and wait
result = emu.step([11])
for _ in range(120):
    result = emu.step([0])
save_frame(bytes(result.observation), space, '03_after_key1_level_select')

# Wait more
for _ in range(120):
    result = emu.step([0])
save_frame(bytes(result.observation), space, '04_waiting_on_track')

# Push Up for 2 seconds
for _ in range(120):
    result = emu.step([1])
save_frame(bytes(result.observation), space, '05_after_2sec_up')

# Push Up for 2 more seconds
for _ in range(120):
    result = emu.step([1])
save_frame(bytes(result.observation), space, '06_after_4sec_up')

print('Done. Check output/frames/')
