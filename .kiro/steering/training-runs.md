---
inclusion: auto
description: Ergonomics for launching long training runs and monitoring them
---

# Training Run Ergonomics

Rules for any long-running training script (anything using
`scripts/train_*.py` or `scripts/go_explore*.py`).

## Launching

- Use `kiro-monitor` per the long-running-tasks rules.
- **Monitor output goes in ONE place: `output/monitor/<job-id>/`.** Pass it
  explicitly, and pass `--job-id` so the path is predictable:

  ```
  nohup kiro-monitor output/monitor/<job-id> <timeout_min> --job-id <job-id> -- \
    <cmd> > /dev/null 2>&1 &
  ```

  This applies to everything monitored — training runs, sweeps, evals,
  diagnostics — so there is one directory to look in and one naming scheme.

- **Never put monitor output inside a run's own output directory.**

  This reverses an earlier version of this rule, which said to write
  `status.json`/`output.log` into `training.output` and explicitly forbade a
  central tree. Two failures killed it:

  1. It contradicted the freshness rule below. A launch script that did
     `rm -rf "$training_output"` deleted the monitor's own `status.json` and
     `output.log` while the monitor was writing them. This is not
     hypothetical: it was caught mid-flight on the v16c run, ~3 minutes
     before the `rm -rf` would have fired.
  2. It disagreed with `kiro-monitor.md`, which sends output to
     `<runs_dir>/<job-id>` when the directory is omitted. With two rules
     giving two answers, monitor logs ended up scattered across
     `debug/<name>/` for diagnostics and `output/mo5/.../<run>/` for
     training, and finding a given run's log meant guessing which.

- Make the RUN's output directory fresh before launch (`rm -rf` + `mkdir -p`)
  unless the run is a resume. Do this in the launching command, BEFORE
  starting the monitor — never inside the monitored script, which cannot
  clear a directory it is being watched from.

## TensorBoard

A TensorBoard server is usually already running pointed at `output/`.
Before starting a new one, check:

```
ss -ltnp | grep 6006
```

If nothing's listening, start it:

```
tensorboard --logdir output --port 6006 --bind_all
```

Launch it via `controlBashProcess` (it's a long-running UI).

### Domain metrics on TB

`EpisodeMetricsCallback` (in `python/retro_ai/training/callbacks.py`)
is wired into every training script. During a run it writes scalar
tags to the same TB event file as the SB3 defaults:

- `reach/from_<S>/ge_<L>` — fraction of episodes that started at level
  S and reached ≥ L.
- `length/from_<S>/reached_<R>/mean` — mean PPO steps for that
  start→end pair.
- `end_reason/<reason>/fraction` — per-reason episode termination.
- `n_episodes/from_<S>` / `n_episodes/total` — sample sizes
  (noise-checking helper).

Use the regex filter in the TB sidebar (e.g. `reach/from_0`) to see
just the reset-start reach rates across all overlaid runs.

For runs that finished before the callback existed, replay the CSV
into a new TB dir with `scripts/episodes_to_tb.py`; it uses the same
aggregator so the tags match.

## Reporting progress

After launch, do the `status.json` check described in
`long-running-tasks.md`. When a run finishes, before launching the
next one:

1. Report the last line of `output.log` (final `cp=`, `saves=`,
   `success=`).
2. Compute end-to-end chaining from `episodes.csv` (last 20% of rows)
   and show per-start-level `reached_level` distribution. The
   `success=[N→N+1:x%]` metric in the log is lossy — it doesn't
   distinguish "reached exactly N+1" from "reached N+2".
3. Update `experiments/003-yeti-training.md` approach section with
   the new result.

## Smoke policy — how much run to buy before a long run

Three tiers. Pick by what the change touches, not by how nervous you feel.

| Change | Check | Cost |
|---|---|---|
| Mechanical (logging, refactor with no semantic intent, config plumbing) | normal ~40k smoke: does it run without exceptions | ~1 min |
| Reward semantics, or the start distribution (curriculum gates, pools, seeding) | ~100k **reading the chain**: `scripts/mo5/yeti/smoke_train.py` | ~3 min |
| "Did it help?" | full run | hours |

The middle tier exists because of L3 v14: a 40k smoke passed, then a 15M run
had already destroyed the from-reset chain by its first route table and we spent
6h finding out. The smoke was not too SHORT — it only looked for exceptions and
never read `reset_reach`. The signal was legible at 90k.

```
# assert a warm-started run keeps the chain it inherited
python3 scripts/mo5/yeti/smoke_train.py --config <cfg> --timesteps 100000 --min-chain 6

# or judge a log a run already produced (no emulator needed)
python3 scripts/mo5/yeti/smoke_train.py \
  --check-log output/monitor/<job-id>/output.log --min-chain 6
```

Exit code is 1 on regression, so it can gate the long run. Set `--min-chain` to
what the parent run achieved (`route[N]: k/N` in its last progress line).

Assert on ROUTE POINTS, not rung indices. Route points are named map points and
are stable across runs; rung indices are not — the ladder re-keyed from fruit
count to mandatory-target count and L3 went from 3 rungs to 15, so the same depth
number means different things on either side of that change. `--min-depth` is
opt-in for that reason.

## Attribution

One change per run. Before writing "one lever vs <baseline>", check the
BASELINE's run date against the commit dates of everything since — v14 was
labelled one-lever while carrying a whole refactor, and cost 6h of
unattributable compute. When two things did move, run the baseline's semantics
forward on current code first (a ~600k control is enough to answer
"did we break it", never "did it help").

## Trusting a number

Confidence comes from averaging independent samples, and there are two levels of
it. Both were ignored for five 6M runs (~15h) spent explaining a difference that
was inside the error bars.

**Judging one model — the samples are EPISODES.** The evaluator prints the 95%
half-width next to every rate. At the default 30 episodes a rate near 0.5 is
known to only **±0.17**:

| rate | n=30 | n=100 | n=300 |
|---|---|---|---|
| 0.20 | ±0.14 | ±0.08 | ±0.05 |
| 0.50 | ±0.17 | ±0.10 | ±0.06 |
| 0.70 | ±0.16 | ±0.09 | ±0.05 |

**Comparing two configs — the samples are RUNS, and the spread is MEASURED.**
Three 6M replicates of one L4 config (identical yaml but the output dir, same
parent, same empty pools, same `seed: 42`, same commit) gave:

| metric | the three runs | sd | range |
|---|---|---|---|
| `mean_rung` | 3.998, 5.009, 4.782 | 0.530 | 1.011 |
| headline frontier rate | 0.149, 0.219, 0.239 | 0.047 | 0.090 |

So **two single runs cannot resolve a difference below 1.47 rungs or 0.131**, and
3 runs per arm detects 1.21 rungs / 0.108. Minimum 3 runs per arm, or do not
claim an effect. Five 6M runs (~15h) were spent attributing a gap of 1.24 rungs /
0.129 — at the resolution floor, so unanswerable by construction.

An earlier version of this section asserted the same 0.13 threshold but derived
it from how much a single run wanders between its own 1.2M blocks. That is the
wrong reference distribution for a difference of run means, the rule got
retracted on that basis, and the measurement then vindicated the number. Keep the
threshold; it now rests on the three runs above, not on that argument.

**Do not read a run's own halves as signal.** Those three replicates swing −1.58,
+1.61 and +1.61 rungs between their first and second 3M. Slicing ONE run into
chunks measures the same experiment repeatedly, not independent experiments. Runs
are not reproducible even at a fixed seed (8 subprocess envs), so a genuine
repeat is a genuine replicate.

**The exception worth exploiting.** When an arm's readout is a quantity pinned at
exactly 0 across every run so far, one run IS informative, because the null is
"never happened". Spend replicates on mean shifts; spend single runs on
does-this-ever-happen.

**A champion's headline number is inflated.** We keep the best of ~60 snapshots
scored on 30 episodes each, so the winner is partly the one that got lucky.
Measured: v23's champion was selected at 0.967 and re-measures at **0.85 ±0.04**
on 300 episodes. Re-measure a champion before quoting it, and assume every
champion figure in older notes is high by ~0.1.

## Do not gate a training run on "no improvement"

Measured across five 6M runs, the gaps between successive new-bests (in 100k
units) are `1 1 1 1 1 1 1 1 1 2 2 3 3 3 4 4 5 5 8 26 32 41`. Median 2.5, tail to
**41 — 4.1M steps of nothing, then an improvement.** Three of the five runs found
their champion after a dry spell of 2.6M+.

So a safe patience exceeds ~45 evals, which on a 6M budget leaves nothing worth
saving. `on_regression: stop` at the default patience 3 cut v25 off at 1.2M with
a champion of 0.533; the same config run to 6M reached **0.700 at 5.7M**. Keep it
off for training. It stays in the code as the control arm for the unbuilt
`revert` variant, not as something to enable.

A dry spell is indistinguishable from convergence until it ends, so for training:
run to budget and keep the best. Stopping rules belong on EXPERIMENTS, where you
want an average and extra steps on one seed buy almost nothing.
