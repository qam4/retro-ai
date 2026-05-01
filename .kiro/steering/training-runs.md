---
inclusion: auto
description: Ergonomics for launching long training runs and monitoring them
---

# Training Run Ergonomics

Rules for any long-running training script (anything using
`scripts/train_*.py` or `scripts/go_explore*.py`).

## Launching

- Use `kiro-monitor` per the long-running-tasks rules.
- Write `status.json` and `output.log` into the run's own output
  directory (the path from `training.output` in the config). Do not
  create a parallel `output/kiro-monitor/` tree. Example:

  ```
  nohup kiro-monitor <training.output> <timeout_min> -- \
    <cmd>
  ```

- Make the output directory fresh before launch (`rm -rf` + `mkdir -p`)
  unless the run is a resume.

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
