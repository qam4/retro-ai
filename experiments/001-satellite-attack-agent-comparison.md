# Experiment 001: Satellite Attack Agent Comparison

**Date:** 2026-03-16
**Commit:** 92924af
**Game:** Satellite Attack (Videopac/Odyssey²)
**Budget:** 100,000 timesteps per agent

## Goal

Compare four RL agent configurations for Satellite Attack within a fixed
100k-timestep budget. All agents use memory-based reward (score read from RAM
addresses 0x80–0x83) with `done_when_score_drops` for episode termination.

## Configurations

All configs are checked into `game_profiles/` and can be re-run with:

```
RETRO_AI_ROM_DIR=roms python3 -m retro_ai.training.cli train game_profiles/<config>.yaml
```

| Config | File | Algorithm | Key Features |
|--------|------|-----------|--------------|
| RND+DrQ Baseline | `satellite_attack_rnd_drq.yaml` | PPO | RND exploration, DrQ augmentation, sticky actions 0.25, 4 envs |
| Tuned PPO | `satellite_attack_ppo_tuned.yaml` | PPO | Higher LR (0.001), ent_coef 0.05, n_steps 512, 4 envs |
| DQN+PER | `satellite_attack_dqn_per.yaml` | DQN | Prioritized replay (α=0.6, β=0.4), RND, 1 env |
| SimPLe | `satellite_attack_simple.yaml` | PPO+WorldModel | 15 rounds, world model with synthetic rollouts |

All configs share: `reward_mode: memory`, `frame_maxpool: true`, `grayscale: true`,
`resize: [84, 84]`, `frame_stack: 4`, `frame_skip: 4`, `reward_clip: 0`.

## Training Results

```
Rank | Config                   | Episodes | Mean Reward | Best Reward | Wall Clock
-----+--------------------------+----------+-------------+-------------+-----------
   1 | DQN+PER                 |      365 |         6.3 |        47.0 |    8997s
   2 | Tuned PPO               |      514 |         2.6 |        34.1 |    5104s
   3 | SimPLe                  |      517 |         2.3 |        31.6 |    5420s
   4 | RND+DrQ Baseline        |      583 |         1.7 |        27.4 |    5332s
```

## Deterministic Evaluation (3 episodes, seed 42–44)

| Config | Eval Mean Reward | Eval Episode Length |
|--------|------------------|--------------------|
| DQN+PER | 13.0 | 238 |
| SimPLe | 3.0 | 221 |
| Tuned PPO | 1.0 | 98 |

RND+DrQ baseline was not evaluated (no significant difference from Tuned PPO expected).

## Observations

- DQN+PER is the clear winner at 100k steps, with 2.4× the mean reward of the
  next best agent and the highest peak reward (47.0).
- DQN's off-policy nature and prioritized replay appear to be much more
  sample-efficient than PPO variants at this low timestep budget.
- SimPLe and Tuned PPO performed similarly during training, but SimPLe showed
  better eval performance (3.0 vs 1.0), suggesting the world model helped
  generalization despite similar training curves.
- DQN+PER took ~1.7× longer wall-clock time (single env vs 4 parallel envs
  for PPO variants), but the quality improvement justified the cost.
- All agents showed high reward variance (std > mean), typical for early
  training on sparse-reward games.

## Post-Mortem: None of the Agents Learned to Play

Deterministic evaluation of all four trained models reveals that none of them
learned meaningful gameplay in 100k steps. The training reward numbers are
misleading because they include the RND intrinsic reward bonus (curiosity),
which rewards visiting novel states — not scoring points.

### Deterministic Action Analysis

| Config | Eval Reward | Unique Actions | Dominant Behavior |
|--------|-------------|----------------|-------------------|
| DQN+PER | 13.0 | 1 | Always action 10 (fire in one direction). Completely degenerate — Q-values collapsed to a single action. |
| Tuned PPO | 4.0 | 11 | Mostly noop `(0,0,0)` and move right `(0,1,0)`. Rarely fires. |
| RND+DrQ Baseline | 1.0 | 10 | Moves left and fires `(1,2,1)`, but ineffectively. |
| SimPLe | 1.0 | 9 | Moves up/down `(1,0,0)/(2,0,0)`, almost never fires. |

### Root Causes

1. **RND reward dominates**: With `intrinsic_reward.coefficient: 1.0`, the
   curiosity bonus overwhelms the sparse game score signal. The agents learn
   to explore (visit novel frames) rather than score. Training mean rewards
   of 1.7–6.3 are mostly intrinsic, not game score.

2. **DQN Q-value collapse**: The DQN model converged to always selecting
   action 10 (100% of the time deterministically). This is a known failure
   mode where the Q-network assigns near-identical values to all actions and
   one wins by a tiny margin. It scored 13.0 only because action 10 happens
   to fire, and the game's starting position lines up a few hits.

3. **PPO variants don't fire**: The joystick action space is `[3,3,2]`
   (vertical, horizontal, fire). The PPO models learned to move but not to
   coordinate firing with positioning. The fire dimension is binary and
   sparse-reward, making it hard to discover via random exploration alone.

4. **100k steps is very low**: For a game requiring coordinated movement +
   firing with sparse rewards (+10 per hit, death on score drop), 100k steps
   is insufficient for any of these algorithms to converge to useful policies.

### Recommendations for Next Experiment

- Reduce `intrinsic_reward.coefficient` to 0.1–0.3 so game score dominates
- Try a shaped reward (e.g. small bonus for firing, proximity to targets)
- Increase budget to 500k–1M steps
- For DQN: increase `exploration_fraction` and `exploration_final_eps` to
  prevent premature convergence
- Consider curriculum learning: train on easier scenarios first

## Reproducing

### Prerequisites

```bash
# 1. Build the native module
cmake --preset ci-linux
cmake --build --preset ci-linux

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Optional: for video recording of evaluation replays
pip install opencv-python
# and for browser-compatible H.264 encoding:
sudo apt install ffmpeg   # Linux
# brew install ffmpeg      # macOS

# 4. Set environment
export RETRO_AI_ROM_DIR=roms
export PYTHONPATH=python:build/ci-linux
```

ROMs are not included in the repository. Place them in `roms/videopac/`.

### Training Configs

All four configs are checked into `game_profiles/` and are self-contained
(no external overrides needed). Each config specifies its own `output_dir`.

| Config | File |
|--------|------|
| RND+DrQ Baseline | `game_profiles/satellite_attack_rnd_drq.yaml` |
| Tuned PPO | `game_profiles/satellite_attack_ppo_tuned.yaml` |
| DQN+PER | `game_profiles/satellite_attack_dqn_per.yaml` |
| SimPLe | `game_profiles/satellite_attack_simple.yaml` |

### Run Training

```bash
python3 -m retro_ai.training.cli train game_profiles/satellite_attack_rnd_drq.yaml
python3 -m retro_ai.training.cli train game_profiles/satellite_attack_ppo_tuned.yaml
python3 -m retro_ai.training.cli train game_profiles/satellite_attack_dqn_per.yaml
python3 -m retro_ai.training.cli train game_profiles/satellite_attack_simple.yaml
```

### Compare Results

```bash
python3 -m retro_ai.training.cli compare \
  output/satellite_attack_rnd_drq \
  output/satellite_attack_ppo_tuned \
  output/satellite_attack_dqn_per \
  output/satellite_attack_simple
```

### Evaluate Best Agent

```bash
python3 -m retro_ai.training.cli evaluate \
  output/satellite_attack_dqn_per/final_model.zip \
  --profile satellite_attack_memory \
  --episodes 10 --seed 42 \
  --output output/satellite_attack_dqn_per/eval \
  --video output/satellite_attack_dqn_per/eval/replay.mp4 \
  --action-mode discrete
```

## Notes

- The original `satellite_attack` baseline (survival reward, no memory-based
  scoring) logged 0 episodes because `survival_bonus: 0` with no episode
  termination signal meant episodes never ended. Not a valid comparison point.
- Training times measured on EC2 (Linux, CPU-only). GPU would be faster.
- Dependencies: see `requirements.txt` for core packages, `requirements-dev.txt`
  for testing. Video recording needs `opencv-python`; H.264 encoding needs
  `ffmpeg` system package.
- Each training run saves a copy of its config and `summary.json` in its
  output directory for post-hoc reproducibility.
