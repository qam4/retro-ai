# Implementation Plan: Satellite Attack Agent Improvement

## Overview

Graduated improvement strategy for training a Satellite Attack agent within 100k timesteps. Organized in three tiers (config-only → DQN+PER → SimPLe) plus a comparison script. All code under `python/retro_ai/training/`, configs under `game_profiles/`, tests under `tests/python/test_training/`.

## Tasks

### Tier 1: Config-Only (No Code Changes)

- [x] 1. Create Tier 1 training config YAML files
  - [x] 1.1 Create `game_profiles/satellite_attack_rnd_drq.yaml` baseline config
    - Set `reward_mode: memory`, `game_profile: satellite_attack_memory`, `survival_bonus: 0`, `reward_clip: 0`
    - Set `action_mode: joystick`, `intrinsic_reward.enabled: true`, `intrinsic_reward.coefficient: 1.0`
    - Set `augmentation: true`, `sticky_actions: 0.25`, `num_envs: 4`, `total_timesteps: 100000`
    - Set `algorithm.name: PPO`, `algorithm.learning_rate: 0.0003`, `algorithm.batch_size: 64`
    - Set `output_dir: output/satellite_attack_rnd_drq`
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9_

  - [x] 1.2 Create `game_profiles/satellite_attack_ppo_tuned.yaml` tuned PPO config
    - Inherit all baseline settings from 1.1
    - Set `algorithm.learning_rate: 0.001`
    - Set `algorithm.extra.ent_coef: 0.05`, `algorithm.extra.n_steps: 512`, `algorithm.extra.n_epochs: 4`, `algorithm.extra.clip_range: 0.2`
    - Set `output_dir: output/satellite_attack_ppo_tuned`
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_

  - [ ]* 1.3 Write property tests for Tier 1 configs (Properties 1, 2, 3)
    - **Property 1: Tier 1 Baseline Config Correctness**
    - Parse `satellite_attack_rnd_drq.yaml`, assert all field values match design spec
    - **Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9**
    - **Property 2: Tuned PPO Config Correctness**
    - Parse `satellite_attack_ppo_tuned.yaml`, assert PPO hyperparameters match design spec
    - **Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5**
    - **Property 3: Config Inheritance Across Tiers**
    - Verify shared fields (`reward_mode`, `survival_bonus`, `action_mode`, `reward_clip`) match baseline values
    - **Validates: Requirements 2.6, 3.9**

- [x] 2. Checkpoint — Verify Tier 1 configs
  - Ensure all tests pass, ask the user if questions arise.
  - Verify both YAML files parse correctly via `TrainingConfigParser.from_yaml()`


### Tier 2: DQN + Prioritized Experience Replay

- [x] 3. Add PER config fields to TrainingConfig
  - [x] 3.1 Add `prioritized_replay`, `prioritized_replay_alpha`, `prioritized_replay_beta` fields to `TrainingConfig` in `python/retro_ai/training/config.py`
    - Add `prioritized_replay: bool = False`
    - Add `prioritized_replay_alpha: float = 0.6`
    - Add `prioritized_replay_beta: float = 0.4`
    - _Requirements: 4.1, 4.3, 4.4_

  - [x] 3.2 Extend `_build_model()` in `python/retro_ai/training/pipeline.py` to wire PER
    - When `config.prioritized_replay` is `True` and algorithm is DQN, import `PrioritizedReplayBuffer` from `sb3_contrib.common.buffers`
    - Pass `replay_buffer_class=PrioritizedReplayBuffer` and `replay_buffer_kwargs={"alpha": ..., "beta": ...}` to DQN constructor
    - Raise `ImportError` with install instructions if `sb3_contrib` is missing
    - Log a warning if `prioritized_replay` is set with a non-DQN algorithm (ignore silently)
    - _Requirements: 4.1, 4.2, 4.5, 4.6_

  - [x] 3.3 Create `game_profiles/satellite_attack_dqn_per.yaml` DQN+PER config
    - Set `algorithm.name: DQN`, `algorithm.learning_rate: 0.0001`, `algorithm.batch_size: 32`
    - Set `algorithm.extra`: `buffer_size: 50000`, `learning_starts: 1000`, `exploration_fraction: 0.3`, `exploration_final_eps: 0.05`, `target_update_interval: 500`, `train_freq: 4`
    - Set `num_envs: 1` (DQN is single-env)
    - Set `prioritized_replay: true`, `prioritized_replay_alpha: 0.6`, `prioritized_replay_beta: 0.4`
    - Inherit reward/action/exploration settings from Tier 1 baseline
    - Set `output_dir: output/satellite_attack_dqn_per`
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.9, 3.10, 4.1, 4.3, 4.4_

  - [ ]* 3.4 Write property tests for DQN+PER (Properties 4, 5)
    - **Property 4: DQN Config Correctness**
    - Parse `satellite_attack_dqn_per.yaml`, assert DQN hyperparameters and `intrinsic_reward.enabled == True`
    - **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.10**
    - **Property 5: PER Integration Correctness**
    - With `prioritized_replay=True` and DQN, verify `_build_model` kwargs include `PrioritizedReplayBuffer` with correct alpha/beta
    - Use Hypothesis to generate random alpha ∈ (0,1), beta ∈ (0,1)
    - **Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.6**

- [ ] 4. Checkpoint — Verify Tier 2 DQN+PER
  - Ensure all tests pass, ask the user if questions arise.
  - Verify DQN config parses and PER fields are wired correctly


### Tier 3: SimPLe World Model

- [x] 5. Implement SimPLe data structures and config
  - [x] 5.1 Add `SimpleConfig` dataclass and `simple` field to `TrainingConfig` in `python/retro_ai/training/config.py`
    - Define `SimpleConfig` with `enabled`, `num_rounds`, `world_model_epochs`, `world_model_lr`, `world_model_batch_size`, `synthetic_ratio`, `rollout_horizon`, `quality_threshold`
    - Add `simple: SimpleConfig = field(default_factory=SimpleConfig)` to `TrainingConfig`
    - Update `TrainingConfigParser.from_dict()` to handle nested `simple` dict → `SimpleConfig` conversion
    - _Requirements: 7.6_

  - [x] 5.2 Implement `TransitionBuffer` in `python/retro_ai/training/simple.py`
    - Pre-allocate numpy arrays for `observations`, `actions`, `rewards`, `next_observations`, `dones` with configurable capacity
    - Implement `add()` with circular buffer semantics
    - Implement `sample(batch_size)` returning random batch of transitions
    - Implement `sample_starts(n)` returning n random observations for rollout starting states
    - _Requirements: 5.1_

  - [ ]* 5.3 Write property tests for TransitionBuffer and SimpleConfig (Properties 6, 12)
    - **Property 6: Transition Buffer Round-Trip**
    - Add random transitions (uint8 obs, int actions, float rewards, bool dones), sample back, verify all fields match stored values
    - **Validates: Requirements 5.1**
    - **Property 12: SimpleConfig YAML Round-Trip**
    - Serialize TrainingConfig with random SimpleConfig fields to YAML, deserialize, verify equivalence
    - **Validates: Requirements 7.6**

- [x] 6. Implement WorldModel
  - [x] 6.1 Implement `WorldModel(nn.Module)` in `python/retro_ai/training/simple.py`
    - Encoder: 3-layer CNN (Conv2d → Conv2d → Conv2d → flatten → FC)
    - Action embedding: `nn.Embedding(num_actions, 64)` concatenated with latent
    - Observation decoder: FC → reshape → 3-layer transposed CNN
    - Reward head: FC → ReLU → FC → scalar
    - Forward returns `(predicted_next_obs, predicted_reward)`
    - _Requirements: 5.2, 5.3, 5.4_

  - [x] 6.2 Implement world model training loop in `python/retro_ai/training/simple.py`
    - Train on TransitionBuffer data using MSE loss for both observation and reward prediction
    - Implement validation function computing MSE on held-out transitions
    - Log warning when validation MSE exceeds `quality_threshold`
    - _Requirements: 5.5, 5.6, 5.7_

  - [ ]* 6.3 Write property tests for WorldModel (Properties 7, 8)
    - **Property 7: World Model Output Shape Correctness**
    - Forward pass with random (batch, C, 84, 84) obs and (batch,) actions, verify output shapes (batch, C, 84, 84) and (batch, 1)
    - **Validates: Requirements 5.2, 5.4**
    - **Property 8: World Model Validation Error**
    - Compute validation error on random transitions, verify MSE ≥ 0
    - **Validates: Requirements 5.6**

- [ ] 7. Implement SyntheticGenerator
  - [ ] 7.1 Implement `SyntheticGenerator` in `python/retro_ai/training/simple.py`
    - Constructor takes `WorldModel`, `horizon` (default 50)
    - `generate()` unrolls world model from starting observations using current policy
    - Sample starting states from TransitionBuffer via `sample_starts()`
    - Terminate rollout early if world model predicts `done=True` or horizon reached
    - Store synthetic transitions in same format as real transitions (Transition dataclass)
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

  - [ ]* 7.2 Write property tests for SyntheticGenerator (Properties 9, 10)
    - **Property 9: Synthetic Rollout Generation Constraints**
    - Verify first obs of each rollout exists in real buffer, rollout length ≤ horizon, total synthetic steps ≈ synthetic_ratio × buffer_size
    - **Validates: Requirements 6.1, 6.2, 6.3, 6.4**
    - **Property 10: Synthetic Transition Format Consistency**
    - Verify synthetic transitions have same fields and compatible dtypes as real transitions
    - **Validates: Requirements 6.5**

- [x] 8. Implement SimplePipeline
  - [x] 8.1 Implement `SimplePipeline` in `python/retro_ai/training/simple.py`
    - Constructor takes `TrainingConfig` with `simple.enabled == True`
    - `run()` executes iterative rounds: collect real data → train world model → generate synthetic data → train PPO policy
    - Allocate `total_timesteps / num_rounds` real steps per round
    - Log round number, real steps, synthetic steps, and policy performance after each round
    - Save final model + `summary.json` in same format as `TrainingPipeline`
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

  - [x] 8.2 Create `game_profiles/satellite_attack_simple.yaml` SimPLe config
    - Build on Tier 1 baseline settings
    - Add `simple` section with `enabled: true`, `num_rounds: 15`, `world_model_epochs: 50`, etc.
    - Set `output_dir: output/satellite_attack_simple`
    - _Requirements: 7.6_

  - [ ]* 8.3 Write property test for SimPLe budget allocation (Property 11)
    - **Property 11: SimPLe Budget Allocation Per Round**
    - For random T and N, verify each round collects ≈ T/N real steps (±10% tolerance for last round)
    - **Validates: Requirements 7.2**

- [ ] 9. Checkpoint — Verify Tier 3 SimPLe module
  - Ensure all tests pass, ask the user if questions arise.
  - Verify WorldModel forward pass, TransitionBuffer round-trip, and SimplePipeline structure


### Comparison Script

- [ ] 10. Implement run comparison and CLI integration
  - [ ] 10.1 Implement `RunComparator` in `python/retro_ai/training/compare.py`
    - `load_summaries()` loads `summary.json` from each output directory, skipping missing/malformed files with warnings
    - `compare()` returns a formatted comparison table ranked by `mean_reward` descending
    - `flag_nonfunctional()` returns `True` when `total_episodes == 0`
    - Output includes rank, config name, episodes, mean reward, best reward, wall clock
    - Exit with code 1 if no valid runs found
    - _Requirements: 8.3, 8.4, 8.5, 8.6_

  - [ ] 10.2 Add `compare` subcommand to CLI in `python/retro_ai/training/cli.py`
    - `retro-ai compare <output_dir1> <output_dir2> ...` accepts multiple output directories
    - Wire to `RunComparator.load_summaries()` → `compare()` → print table
    - _Requirements: 8.3_

  - [ ]* 10.3 Write property tests for comparison (Properties 13, 14, 15)
    - **Property 13: Metrics Output Completeness**
    - Record random episodes in MetricsTracker, verify `summary.json` keys and `metrics.csv` rows
    - **Validates: Requirements 8.1, 8.2**
    - **Property 14: Comparison Ranking Correctness**
    - Create summaries with random distinct `mean_reward` values, verify descending sort order
    - **Validates: Requirements 8.3, 8.4, 8.5**
    - **Property 15: Non-Functional Run Flagging**
    - Create summaries with random `total_episodes` values, verify `functional == (episodes > 0)`
    - **Validates: Requirements 8.6**

- [x] 11. Final checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.
  - Verify all four YAML configs parse correctly
  - Verify PER wiring in pipeline
  - Verify SimPLe module structure
  - Verify comparison script output

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Tier 1 requires zero code changes — only YAML config files
- Tier 2 is a small pipeline extension (3 config fields + ~15 lines in `_build_model`)
- Tier 3 is a new module (`simple.py`) with WorldModel, TransitionBuffer, SyntheticGenerator, SimplePipeline
- Property tests use Hypothesis for Python
- Build/test with: `python -m pytest tests/python/ -v --tb=short`
