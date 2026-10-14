# Yeti snowballs: what decides them, and why l1_v17 does not react

Measured 2026-10-13/14 on the fixed emulator (native fdd7e006...). Layout 1 only (L1, L5)
unless stated. Tools: `python/retro_ai/games/yeti_hazards.py` (decoding),
`scripts/mo5/yeti/harvest_respawns.py`, `python/retro_ai/training/root_pool.py`,
`eval_from_reset.py --start-pool`. The first probes were scratch scripts, not kept; the
numbers below are what they printed.

## The short version

1. From a given state the game is deterministic: same inputs, same game, bit for bit.
2. Within one life the player's inputs do NOT change the snowballs' route choices. They
   shift spawn times by a few steps (the spawn clock skips ticks when the game loop is
   busier), nothing else.
3. The choices come from a pseudo-random generator. A level start fixes its state, so
   **L1's first life always plays the same sequence**. Every DEATH leads to a respawn
   with a new sequence, which depends on when and where the life ended.
4. l1_v17 memorised L1's first-life sequence. It reaches the princess 215/300 from
   reset and **0/300 from respawns with other sequences**.
5. Fix being tried: train on respawns too (`curriculum.root_pool`), score on held-out
   sequences.

User check in RetroArch (2026-10-14): the game always starts with the same sequence;
after a death it changes; a new game after game over gets a different sequence.

## RAM

Six snowball slots at `0x2B60 + 8k`: `+0` y (px; rolling on a floor = standing y + 10),
`+1` x (RAM units, px = x*4 + 8), `+4` active. Confirmed by drawing them over frames.
`0x2B21` is the spawn clock (28 -> 1, a snowball when it wraps). `0x2B24` changes twice
just before each spawn and is part of the route generator (below); not all of it.

## Evidence

**Inputs within a life.** 17 L1 first-life runs (8 l1_v17 episodes with different seeds,
standing still, walking, 7 title-screen waits): 0 conflicting choices in 2,089 pairwise
comparisons of where each snowball left each floor. L5, 8 runs: 0 in 462. Spawn steps
standing still 17, 47, 78, 111, 144, 178; in policy episodes 17, 50, 83, 116, 151, 187
(within +-1 of each other). Spawn-clock skips: 13.5% of steps standing, 16.2% walking,
22.4% in policy episodes. A held direction that does not move the player changes game
RAM 0x2AE0-0x2B8F for a few steps, then the whole game RAM matches again; deaths on the
same step.

**Title screen.** +1, +2 and +333 frames before pressing start: identical routes and
`0x2B24` sequence, spawns within 1 step. (+3, +7, +50 hit another screen; L1 never
started.)

**The generator.** Writing 180, 0 or 77 into `0x2B24` in L1's start save (CRC32 at the
end of the save recomputed, `emulators/crayon/src/savestate.cpp`) changed the routes of
the first three snowballs every time and no spawn time. Writing L1's value (148) into
L5's start did not give L1's routes, so other generator state lives elsewhere.

**L5.** Its own fixed sequence: e.g. the 2nd snowball goes down L45 on L5 (8/8 runs) and
rolls past it on L1 (10/10). Not tested: whether L5's sequence depends on how L4 was
played.

**Deaths.** Playing on after a death: player back at the start, all snowballs cleared,
lives 5 -> 4, about 380 steps of death and restart. 56 respawns from l1_v17 first lives
that ended at random points gave 19 different second-life sequences. Deaths close in
time mostly share one (steps 169-232 on floors 1-4: one sequence for 13 of 14); place
matters too (step 226 floor 1 vs 223 floor 4: different). Falls vs hits not settled
(one fall in the sample).

**What a respawn is.** Bonus back at 1000 and ticking, fruit counter back to 4 (also
after collecting all 4), score kept, lives one less. The wait before respawn depends on
the floor of death: ~375-395 steps (floor 1), ~340-358 (2), ~300-320 (3), ~258-270 (4).
The first frame differs from a fresh reset only in the score bar (43 pixels: lives and
score).

## l1_v17 does not react (all stochastic, l1_v17's own settings)

| start | princess |
|---|---|
| L1 game reset (registry) | 215/300 = 0.72 |
| same reset, policy starts 4 steps later | 75/150 = 0.50 |
| reset with `0x2B24` rewritten (30 values) | 6/300 = 0.02 |
| 198 scratch-harvested respawns, 26 sequences | 0/330 |
| `l1_r1` held-out respawns (10 sequences, 114 states) | **0/300**, never 4 fruits |
| control l1_v18 best (2.75M), same held-out | **0/300**, never 4 fruits |

A dummy standing still lives longer on the rewritten-generator starts (median 229 steps)
than on the plain one (191), so those starts are not harder in themselves.

## Respawn pool `l1_r1`

`experiments/003-yeti/data/respawns/l1_r1.json`: 299 respawns from 4 x 75 l1_v17
episodes (seeds 10000-40000), 31 sequences; every 3rd sequence held out (10 sequences,
114 states); 21 train sequences, 185 states. The states are local
(`output/mo5/yeti/respawns/l1_r1/states/`, a save holds the game); the manifest pins
each by sha256. Sequence identity is judged on each respawn's first 10-24 snowball
choices (330 steps standing still), so two respawns grouped together could still differ
later.

## Open

- Does L5's sequence depend on the L4 playthrough? Matters for the full-game runner.
- The rest of the generator state (beyond `0x2B24`).
- L2-L4 hazards (goats, kangaroos) not studied.
- Respawns show 4 lives in the score bar; only 1st-death respawns harvested so far.
