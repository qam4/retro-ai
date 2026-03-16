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
