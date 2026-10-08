# Champion reproducibility and emulator provenance (commit `2b0a45d`)

**Status: RESOLVED 2026-08-12.** Root cause found, confirmed by bisecting the
native core, and proven with a policy-free RAM comparison.

Cross-refs: `experiments/003-yeti-training.md` H-AM (the run-history entry),
H-AC (the fix that caused this), H-AE (the level-2 twin), TODO.md
"BLOCKER [CONFIRMED]" (follow-ups).

---

## 1. What you need to know (read this if nothing else)

The L1 champion `champions/v15_phase2_4500k`, documented at **99.7% princess
from reset**, scores **0/40 today**. That is not a regression to fix, and the
old behaviour must not be restored.

- **Physics never changed.** Cold-boot emulator dynamics are byte-identical
  before and after `2b0a45d`.
- **The old state-restore was broken**, and `reset()` is implemented as a state
  restore, so the bug was in the path of *every* training and eval episode after
  the first one.
- **The champion is overfit to that bug.** It was trained in a world where a
  hazard counter was frozen and the HUD font was blank. On a correct emulator its
  navigation still works (4 fruits, 39/40) but the hazard-sensitive final leg
  collapses (princess 0/40).
- **The current core is the correct one.** On it, a restored episode start is
  bit-identical to a real boot. That is now checkable in one command (§5).

**So: any L1 princess number recorded before `2b0a45d` is void.** Re-measure or
retrain on the current core. Do not chase the old figure, and do not try to
recover it with observation tricks — the dominant component is game state, not
pixels (§4.3).

## 2. Measurements

Documented eval, unchanged, on four core builds:

```
eval_from_reset.py --model output/mo5/yeti/champions/v15_phase2_4500k/final_model.zip \
                   --profile yeti_fruit --episodes 40 --stochastic
```

| core build | princess | >= 4 fruits | note |
|---|---|---|---|
| `2b0a45d~1` (= `7f8d8c7`) | **39/40 (97.5%)** | 39/40 | reproduces the documented result |
| `2b0a45d` | 0/40 (0.0%) | 39/40 | **the commit that changed it** |
| `HEAD` (`f542839`) | 0/40 (0.0%) | 39/40 | current source |
| stale Jul-2 `.so` | 0/40 (0.0%) | 39/40 | the unversioned binary in `build/ci-linux` |

Two things this settles:

1. The 99.7% **was** real and reproducible — against the pre-`2b0a45d` core. The
   figure was never fabricated, which was the alternative hypothesis.
2. A freshly built HEAD matches the stale `.so` exactly, so nothing was hiding in
   the unversioned binary. The bisect is trustworthy.

Note the shape of the failure is identical in all four builds up to the last
step: 39/40 reach 4 fruits, with the same single 3-fruit outlier. Only the
princess leg differs.

## 3. Why a "load_state only" commit changed a from-RESET eval

This is the part that made the bug look impossible, and it is the reusable
lesson.

`MO5RLInterface::reset()` (`src/mo5_rl.cpp`) boots the emulator only the FIRST
time. After that it replays a cached snapshot:

```cpp
StateResult reset(int /*seed*/) {
    if (!startup_state_.empty()) {
        // Fast path: restore cached post-startup state
        emulator_->load_state_from_buffer(startup_state_.data(), startup_state_.size());
    } else {
        emulator_->reset();
        // ... run_startup_sequence(...); cache_startup_state();
    }
```

A training run or eval does thousands of episodes, so effectively **all** of them
take the restore path. "Save/load only" is therefore never eval-neutral here.
`2b0a45d` changed exactly that path (`mo5_rl.cpp` now calls
`load_state_from_buffer`, plus the crayon bump to `136d6b9` which changed
`set_state` semantics).

## 4. Root cause

### 4.1 The proof: the old restore did not reproduce a real boot

Policy removed from the loop — fixed no-op action sequence, full 48K RAM
snapshot per step, 150 steps (`scripts/mo5/yeti/core_determinism_probe.py`):

| comparison | result |
|---|---|
| old vs new core, **episode 1 (real boot)** | **identical, all 150 steps** |
| old vs new core, episode 2 (restore) | diverges at step 44 |
| **new** core, boot vs restore | **identical — restore is faithful** |
| **old** core, boot vs restore | diverges at step 44, 10 bytes by step 149 |

Read the last two rows together: the new core's restore reproduces a real boot
bit-exactly, and the old core's did not. The fix is correct; the old behaviour
was the bug.

### 4.2 What the old restore got wrong, as the policy saw it

**(a) Dynamics — the important one.** ~57 RAM addresses drift from a true boot,
including the 8-byte-strided object table at `0x2B60/68/70/78/80` (hazards).
Signature byte: **`0x2B24` is FROZEN at 148** for all 150 steps on the old core,
where a real boot has it live and varying (13..251) — a counter/RNG driver was
left inert, so snowball timing was quieter and more predictable than the real
game. Player position (`0x2B52`/`0x2B51`) is unaffected, so this is the hazards,
not the avatar.

In a deterministic champion episode the first play-area pixel difference is a
falling snowball at step 76 (bbox `y=[96,107] x=[168,179]`, drifting down-right
over following steps) while the player's RAM is still identical through step 84
and the actions still match for 100 steps.

**(b) Rendering — the visible but minor one.** `MemorySystem::set_state` wiped
the monitor ROM, which holds the MO5 character font, so from episode 2 onward
every HUD glyph drew BLANK. Fixed now, so the HUD paints real digits: exactly
**105 pixels differ, all inside rows `y=1..14`**, from step 2 onward. The frame
at reset is still byte-identical between cores; the difference appears with the
first HUD repaint.

### 4.3 Why masking or cropping the HUD does not recover the champion

Tested, not assumed: with rows 0..15 cropped out of the observation, the two
cores still diverge — at step 76, via the snowball. Difference (a) is in game
state, not in the frame, so no observation-level trick reaches it.

Cropping the HUD is still worth doing on its own merits (see TODO.md), just not
as a fix for this.

## 5. Reproduce it

### Standing guard on the current build (~30s, one build, no champion needed)

```bash
env PYTHONPATH=python:build/ci-linux RETRO_AI_ROM_DIR=roms \
  python3 scripts/mo5/yeti/core_determinism_probe.py selfcheck --out debug/ram_head.npz
```

Expected: `IDENTICAL for all 150 steps` and exit 0. A non-zero exit means
`reset()` no longer reproduces a real boot — stop and fix that before trusting
any training run, because every episode would be training against a state the
game never reaches.

### Full A/B against an old core

```bash
# 1. Worktree at the pre-change commit (submodules resolve offline from the
#    shared object store; the mo5 core is vendored in-tree, not a submodule).
git worktree add out/oldcore-2b0a45d 2b0a45d~1
git -C out/oldcore-2b0a45d submodule update --init --recursive

# 2. Build. The preset alone is NOT enough: it picks python3.12 and cannot find
#    pybind11, so the module would not match the 3.9 runtime. Pass both.
cmake -S out/oldcore-2b0a45d -B out/oldcore-2b0a45d/build/ci-linux --preset ci-linux \
  -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF \
  -DPython3_EXECUTABLE=/usr/bin/python3.9 \
  -Dpybind11_DIR="$(python3.9 -m pybind11 --cmakedir)"
cmake --build out/oldcore-2b0a45d/build/ci-linux -j8

# 3. The .so lands in build/ci-linux/ (NOT build/), so PYTHONPATH must include
#    that subdirectory.
env PYTHONPATH=python:out/oldcore-2b0a45d/build/ci-linux RETRO_AI_ROM_DIR=roms \
  python3 scripts/mo5/yeti/core_determinism_probe.py capture --out debug/ram_old.npz --frames
env PYTHONPATH=python:build/ci-linux RETRO_AI_ROM_DIR=roms \
  python3 scripts/mo5/yeti/core_determinism_probe.py capture --out debug/ram_new.npz --frames
python3 scripts/mo5/yeti/core_determinism_probe.py compare --a debug/ram_old.npz --b debug/ram_new.npz
```

The champion eval itself is ~50s for 40 episodes; run it with
`PYTHONPATH=python:<core-build-dir>` to pick the core under test.

## 6. Artifacts

Tracked, so they survive:

- `experiments/003-yeti/data/champion_repro_2b0a45d/` — the four eval result
  JSONs (per-episode rows, `max_cp` counts) plus `summary.json`.
- `scripts/mo5/yeti/core_determinism_probe.py` — regenerates all the evidence
  above.

Machine-local and regenerable (gitignored, present on this box as of
2026-08-12): worktrees `out/oldcore-2b0a45d` and `out/headcore` (~213M each,
already built); `debug/{oldcore,midcore,headcore,newcore}_l1_eval/` raw eval
logs; `debug/probe_{old,head}.npz` RAM/frame captures (~2.8M each).

## 7. Follow-ups

- **Record the native build's SHA** in champion dirs and run manifests. The
  manifest already captures git info; add the built artifact's SHA/mtime. Without
  it, a policy cannot be tied to the emulator it was trained on — which is what
  made this take a full investigation instead of one command.
- **Re-measure or retrain L1 on the current core.** Navigation transfers
  (reach-4 is 39/40 everywhere); the final leg needs to relearn live hazards.
  Warm-starting from the champion's weights is the obvious first attempt.
- **Keep `selfcheck` in the loop** after any change to save/load, the crayon
  submodule, or the startup sequence. It is ~30s and needs no champion.
- **Consider cropping the HUD** for the next training generation (TODO.md has
  the rect and the caveats). Hygiene, not a fix for this.
