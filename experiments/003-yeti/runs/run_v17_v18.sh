#!/usr/bin/env bash
# Two 15M runs back to back, each ONE lever from v16c's 15M (which reached mean rung
# 8.49 / 73.0% at rung 10, cold). SEQUENTIAL: 8 cores and each run uses num_envs 8, so
# running them together halves each one's throughput and makes the wall clock meaningless.
#
#   v17  curriculum.seed_waypoint_skip: [Lclimb3_top]
#        Floor 11 is the only exposed ground in that stretch (NOOP death in 3-13 frames,
#        always) and `admit_requires_survival` admits only arrivals that landed in a
#        benign hazard phase -- pool seeds survive a median 8 NOOP frames against 3 for
#        the policy's own arrivals. The approach to the floor-12 jump needs 8 steps, so a
#        3-frame arrival is doomed. Seeding there hands the agent a survivable phase for
#        free, i.e. it never practises the decision the level turns on. Excluding it
#        pushes practice back to `Step` (same px, one floor down, off the patrol route,
#        pool == reset arrivals). TESTS: is representative practice enough on its own?
#
#   v18  resume from v13's final_model.zip (+ v13's pools)
#        v6/v13 crossed floor 11 and walled at rope 2; v16c walls at floor 11. Same map,
#        same reward in all three. The measured difference is arrival survivability:
#        v6/v13 median 12 frames, v16c median 3. TESTS: does that timing TRANSFER? If it
#        survives under v16c's reward and sprite geometry the skill was inherited and cold
#        runs never find it; if it decays toward 3, something in the v16 configuration
#        destroys it and box-vs-sprite or the absent target_kl is next.
#
# WHAT TO READ AFTERWARDS, for both:
#   * champion on the shared stick: experiments/003-yeti/runs/run_champion_eval.sh <run>
#     (v6 8.52/75.3%, v13 8.49/63.3%, v16c 8.49/73.0%)
#   * arrival survivability from reset, the quantity that separates the runs:
#     diag/l4_seed_determinism.py --pools Lclimb3_top --noop-safety --vs-reset 30 \
#       --models <run>/best/best_model.zip
#   * `Low1` pool size (8 in v16c, 100 in v6/v13) and whether `Low2` moves at all
#   * for v17 specifically: does `Lclimb3_top`'s REACH hold up without its pool? It is
#     still detected and tracked, so the route table row stays -- only the pool goes.
#
# Read peaks and windows, NOT endpoints: v16c peaked over 9M-12M and decayed by 15M.
set -uo pipefail
cd "$(dirname "$0")/../../.."

C=experiments/003-yeti/configs
for arm in v17_noseed_climb3_cold_15m v18_warm_v13_15m; do
  cfg="$C/yeti_curriculum_l4_${arm}.yaml"
  out="output/mo5/yeti/training/yeti_curriculum_l4_${arm}"
  echo "=============================================================="
  echo "ARM: $arm   ($(date -u +%Y-%m-%d\ %H:%M:%S) UTC)   expect ~7h"
  echo "=============================================================="
  # The RUN's directory is cleared here, in the launcher, BEFORE the trainer starts.
  # Never inside a monitored script: kiro-monitor writes status.json/output.log into
  # output/monitor/<job-id>/, and an rm -rf of the wrong directory once came within
  # minutes of deleting a live monitor's own log.
  rm -rf "$out"
  python3 -u scripts/mo5/yeti/train_checkpoint_curriculum.py --config "$cfg"
  echo "$arm exit: $?"
  echo
done
echo "BOTH ARMS DONE"
