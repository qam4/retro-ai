# Requirements Document

## Introduction

Improve the Satellite Attack agent to achieve a non-zero score within a 100k timestep training budget. The current agent produced 0 completed episodes in 100k steps using a survival reward, which was the wrong approach. This spec defines a graduated improvement strategy: start with config-only quick wins (correct reward signal, exploration aids, action mode), then add DQN with Prioritized Experience Replay for sample efficiency, then optionally pursue SimPLe (Simulated Policy Learning) for maximum sample efficiency. Each tier builds on the previous one, and training configs are the primary deliverable at each stage.

## Glossary

- **Pipeline**: The `TrainingPipeline` class that orchestrates end-to-end RL training runs
- **Training_Config**: A YAML file specifying algorithm, reward mode, preprocessing, and hyperparameters for a training run
- **Game_Profile**: A YAML file defining emulator, ROM, reward addresses, startup sequence, and preprocessing defaults for a specific game
- **Score_Delta_Reward**: The reward signal computed as the change in game score between consecutive frames, read from emulator RAM
- **RND**: Random Network Distillation, a curiosity-driven exploration method that rewards visiting novel states
- **DrQ**: Data-regularized Q-learning, a data augmentation technique applying random crops and jitter to observations
- **Sticky_Actions**: A stochasticity mechanism where the previous action is repeated with probability p, preventing memorization
- **DQN**: Deep Q-Network, an off-policy RL algorithm with a replay buffer for better sample efficiency
- **PER**: Prioritized Experience Replay, a replay buffer strategy that samples high-error transitions more frequently
- **SimPLe**: Simulated Policy Learning, a world-model approach that generates synthetic experience to amplify real data
- **Joystick_Mode**: An action mode supporting simultaneous directional movement and fire button, mapped to multi-discrete actions
- **Episode**: One complete game playthrough from start to death (score drop)
- **Timestep_Budget**: The fixed 100k real environment steps available for training


## Requirements

### Requirement 1: Config-Only Baseline with Correct Reward Signal

**User Story:** As a researcher, I want a training config that uses pure score delta reward with exploration aids and joystick actions, so that the agent receives the correct learning signal and can explore effectively within 100k steps without any code changes.

#### Acceptance Criteria

1. THE Training_Config SHALL use `reward_mode: memory` with the `satellite_attack_memory` Game_Profile to obtain Score_Delta_Reward from RAM
2. THE Training_Config SHALL set `survival_bonus: 0` so that the agent receives reward only from scoring hits
3. THE Training_Config SHALL set `action_mode: joystick` so that the agent can move and fire simultaneously via Joystick_Mode
4. THE Training_Config SHALL set `reward_clip: 0` so that different hit values are preserved without clipping
5. THE Training_Config SHALL enable RND exploration with `intrinsic_reward.enabled: true` and `intrinsic_reward.coefficient: 1.0`
6. THE Training_Config SHALL enable DrQ augmentation with `augmentation: true`
7. THE Training_Config SHALL enable Sticky_Actions with `sticky_actions: 0.25`
8. THE Training_Config SHALL use `num_envs: 4` parallel environments to increase throughput
9. THE Training_Config SHALL set `total_timesteps: 100000` to stay within the Timestep_Budget
10. WHEN the Pipeline executes this Training_Config, THE Pipeline SHALL complete training and produce a saved model file


### Requirement 2: PPO Hyperparameter Tuning Config

**User Story:** As a researcher, I want a tuned PPO config with adjusted entropy coefficient, learning rate schedule, and rollout length, so that the on-policy agent explores more aggressively and learns faster in the sparse-reward Satellite Attack environment.

#### Acceptance Criteria

1. THE Training_Config SHALL set the PPO entropy coefficient to 0.05 via `algorithm.extra.ent_coef: 0.05` to encourage broader exploration
2. THE Training_Config SHALL set `algorithm.extra.n_steps: 512` to collect longer rollouts before each policy update
3. THE Training_Config SHALL set `algorithm.learning_rate: 0.001` to use a higher initial learning rate for faster early learning
4. THE Training_Config SHALL set `algorithm.extra.n_epochs: 4` to perform multiple passes over each rollout batch
5. THE Training_Config SHALL set `algorithm.extra.clip_range: 0.2` to constrain policy updates within a stable range
6. THE Training_Config SHALL build on Requirement 1 by inheriting the same reward mode, action mode, exploration aids, and parallel environment settings


### Requirement 3: DQN with Prioritized Experience Replay

**User Story:** As a researcher, I want a DQN-based training config with prioritized experience replay, so that the agent can reuse past experience more efficiently and learn from the most informative transitions within the 100k step budget.

#### Acceptance Criteria

1. THE Training_Config SHALL set `algorithm.name: DQN` to use the off-policy Deep Q-Network algorithm
2. THE Training_Config SHALL set `algorithm.extra.buffer_size: 50000` to maintain a replay buffer sized for the 100k Timestep_Budget
3. THE Training_Config SHALL set `algorithm.extra.learning_starts: 1000` so that the agent collects initial experience before training begins
4. THE Training_Config SHALL set `algorithm.extra.exploration_fraction: 0.3` to dedicate 30% of training to epsilon-greedy exploration
5. THE Training_Config SHALL set `algorithm.extra.exploration_final_eps: 0.05` to maintain a minimum 5% random action rate
6. THE Training_Config SHALL set `algorithm.extra.target_update_interval: 500` to stabilize value estimation with periodic target network updates
7. THE Training_Config SHALL set `algorithm.extra.train_freq: 4` to update the network every 4 environment steps
8. THE Pipeline SHALL support SB3 DQN algorithm selection through the existing `ALGORITHM_MAP` without new code
9. THE Training_Config SHALL use `reward_mode: memory` with `survival_bonus: 0`, `action_mode: joystick`, and `reward_clip: 0` consistent with Requirement 1
10. THE Training_Config SHALL enable RND exploration alongside DQN epsilon-greedy exploration for combined novelty-seeking behavior


### Requirement 4: Prioritized Experience Replay Integration

**User Story:** As a researcher, I want the DQN training to use prioritized experience replay instead of uniform sampling, so that high-error transitions are replayed more often and the agent learns faster from rare scoring events.

#### Acceptance Criteria

1. THE Pipeline SHALL support a `prioritized_replay: true` configuration option for DQN training
2. WHEN `prioritized_replay` is enabled, THE Pipeline SHALL use SB3-contrib's `PrioritizedReplayBuffer` as the DQN replay buffer class
3. THE Training_Config SHALL set PER alpha to 0.6 to control the degree of prioritization
4. THE Training_Config SHALL set PER beta to 0.4 as the initial importance-sampling correction weight
5. IF the `sb3_contrib` package is not installed, THEN THE Pipeline SHALL raise a clear error message indicating the missing dependency
6. THE Pipeline SHALL pass the PER buffer class to DQN via the `replay_buffer_class` parameter without modifying the core DQN algorithm


### Requirement 5: SimPLe World Model Training

**User Story:** As a researcher, I want to train a world model from real environment interactions, so that the model can generate synthetic experience to amplify the 100k real steps into much more training data.

#### Acceptance Criteria

1. THE SimPLe_Trainer SHALL collect real environment transitions (observation, action, reward, next_observation, done) during an initial data collection phase
2. THE SimPLe_Trainer SHALL train a neural network world model to predict next_observation and reward given current observation and action
3. THE World_Model SHALL use a convolutional architecture that accepts 84x84 grayscale frame-stacked observations
4. THE World_Model SHALL predict both the next observation frame and the scalar reward for a given state-action pair
5. WHEN training the World_Model, THE SimPLe_Trainer SHALL use mean squared error loss for observation prediction and reward prediction
6. THE SimPLe_Trainer SHALL validate World_Model quality by comparing predicted observations against held-out real transitions
7. IF the World_Model prediction error exceeds a configurable threshold, THEN THE SimPLe_Trainer SHALL log a warning about model quality


### Requirement 6: SimPLe Synthetic Experience Generation

**User Story:** As a researcher, I want the world model to generate synthetic rollouts, so that the policy can train on many more transitions than the 100k real steps allow.

#### Acceptance Criteria

1. THE SimPLe_Generator SHALL produce synthetic rollouts by unrolling the World_Model from real starting states using the current policy
2. THE SimPLe_Generator SHALL generate a configurable number of synthetic steps per real data collection round (default: 8x amplification)
3. THE SimPLe_Generator SHALL limit synthetic rollout length to a configurable horizon (default: 50 steps) to prevent compounding model errors
4. WHEN generating synthetic rollouts, THE SimPLe_Generator SHALL sample starting states uniformly from the real experience buffer
5. THE SimPLe_Generator SHALL store synthetic transitions in the same format as real transitions so the policy trainer can consume both interchangeably


### Requirement 7: SimPLe Training Loop Orchestration

**User Story:** As a researcher, I want an iterative training loop that alternates between real data collection, world model training, and policy training on synthetic data, so that the agent progressively improves using the SimPLe algorithm.

#### Acceptance Criteria

1. THE SimPLe_Pipeline SHALL execute iterative rounds consisting of: (a) collect real data, (b) train World_Model, (c) generate synthetic data, (d) train policy on synthetic + real data
2. THE SimPLe_Pipeline SHALL allocate the 100k real Timestep_Budget across a configurable number of rounds (default: 15 rounds of ~6600 real steps each)
3. THE SimPLe_Pipeline SHALL train the policy using PPO on the combined real and synthetic experience after each round
4. THE SimPLe_Pipeline SHALL save the final trained policy model in the same format as the standard Pipeline output
5. WHEN a round completes, THE SimPLe_Pipeline SHALL log the round number, real steps used, synthetic steps generated, and current policy performance
6. THE SimPLe_Pipeline SHALL accept configuration via a Training_Config YAML file with a `simple` section for world model and generation parameters


### Requirement 8: Training Run Comparison and Evaluation

**User Story:** As a researcher, I want to compare the results of different training configurations side by side, so that I can determine which approach produces the best agent within the 100k step budget.

#### Acceptance Criteria

1. WHEN a training run completes, THE Pipeline SHALL write a `summary.json` containing total_episodes, mean_reward, best_reward, mean_episode_length, and wall_clock_seconds
2. THE Pipeline SHALL write per-episode metrics to `metrics.csv` with columns for timestep, episode_reward, and episode_length
3. WHEN comparing runs, THE Evaluation_Script SHALL load summary.json files from multiple output directories and display a comparison table
4. THE Evaluation_Script SHALL rank configurations by mean_reward as the primary metric
5. THE Evaluation_Script SHALL report the number of completed episodes as a secondary metric to verify the agent is actually playing the game
6. IF a training run produces zero completed episodes, THEN THE Evaluation_Script SHALL flag the run as non-functional