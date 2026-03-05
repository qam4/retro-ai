# Requirements Document

## Introduction

The Agent Training Pipeline extends the retro-ai framework with end-to-end capabilities for training, evaluating, and deploying reinforcement learning agents on retro game emulators. It builds on the existing infrastructure (BaseEnv, GymnasiumWrapper, preprocessing, reward systems) and adds a training orchestration layer using Stable-Baselines3, game-agnostic configuration, real-time inference at 60 FPS, and evaluation/metrics tracking. The pipeline is designed to work with any game on any supported emulator (Videopac now, MO5/Crayon later, and future emulators).

## Glossary

- **Training_Pipeline**: The orchestration module that configures and runs RL training using Stable-Baselines3 algorithms on GymnasiumWrapper environments
- **Training_Config**: A configuration object specifying all parameters for a training run (algorithm, hyperparameters, environment settings, reward mode, preprocessing, checkpointing)
- **Agent**: A trained RL model (neural network policy) that selects actions given observations
- **Inference_Runner**: The module that loads a trained Agent and runs it in real-time against a live emulator at the target frame rate
- **Evaluation_Module**: The module that runs a trained Agent for multiple episodes and collects performance metrics
- **Metrics_Tracker**: The component that records, aggregates, and persists training and evaluation metrics (episode reward, length, score, FPS)
- **Game_Profile**: A configuration describing a specific game on a specific emulator, including ROM paths, action mappings, reward mode, startup sequence, and preprocessing defaults
- **Startup_Sequence**: An ordered list of actions (button presses, delays) required to navigate from emulator boot to gameplay for a specific game
- **Checkpoint**: A saved snapshot of the Agent's policy weights and optimizer state at a specific training step
- **Callback**: A Stable-Baselines3 callback object that executes custom logic during training (e.g. logging, checkpointing, early stopping)
- **Algorithm_Config**: The subset of Training_Config specifying the RL algorithm name and its hyperparameters

## Requirements

### Requirement 1: Training Pipeline Orchestration

**User Story:** As a researcher, I want a single entry point to configure and launch RL training on any game, so that I can start training agents without writing boilerplate code.

#### Acceptance Criteria

1. WHEN a valid Training_Config is provided, THE Training_Pipeline SHALL create the environment, apply preprocessing, wrap it with GymnasiumWrapper, instantiate the selected algorithm, and begin training
2. THE Training_Pipeline SHALL support PPO and DQN algorithms from Stable-Baselines3
3. WHEN training is launched, THE Training_Pipeline SHALL log the algorithm name, total timesteps, and environment configuration at info level
4. THE Training_Pipeline SHALL accept Training_Config from a Python dictionary, a JSON file, or a YAML file
5. WHEN training completes, THE Training_Pipeline SHALL save the final Agent to the configured output directory
6. IF training is interrupted by a keyboard interrupt, THEN THE Training_Pipeline SHALL save the current Agent state before exiting

### Requirement 2: Training Configuration

**User Story:** As a researcher, I want to specify all training parameters in a single configuration, so that I can reproduce experiments and iterate on hyperparameters.

#### Acceptance Criteria

1. THE Training_Config SHALL include fields for algorithm name, total timesteps, learning rate, batch size, and algorithm-specific hyperparameters
2. THE Training_Config SHALL include fields for emulator type, ROM path, BIOS path, and reward mode
3. THE Training_Config SHALL include fields for preprocessing options (grayscale, resize dimensions, frame stack count, frame skip count)
4. THE Training_Config SHALL include fields for output directory, checkpoint interval, and log interval
5. THE Training_Config SHALL provide default values for all optional fields
6. WHEN a required field is missing from Training_Config, THE Training_Pipeline SHALL raise a descriptive error naming the missing field
7. THE Config_Serializer SHALL format Training_Config objects back into valid configuration files
8. FOR ALL valid Training_Config objects, parsing then serializing then parsing SHALL produce an equivalent object (round-trip property)

### Requirement 3: Game Profile System

**User Story:** As a user, I want to define game-specific profiles, so that I can train on different games without manually configuring ROM paths, actions, and startup sequences each time.

#### Acceptance Criteria

1. THE Game_Profile SHALL specify the emulator type, ROM path, BIOS path, action space description, reward mode, and preprocessing defaults for a specific game
2. THE Game_Profile SHALL include an optional Startup_Sequence that lists actions and delays needed to reach gameplay from emulator boot
3. WHEN a Startup_Sequence is defined, THE Training_Pipeline SHALL execute the Startup_Sequence after each environment reset before returning the initial observation
4. THE Game_Profile SHALL be loadable from a JSON or YAML file
5. WHEN a Training_Config references a Game_Profile by name, THE Training_Pipeline SHALL merge the Game_Profile settings with the Training_Config, with Training_Config values taking precedence
6. THE Game_Profile SHALL support per-game reward parameters (memory addresses for memory reward, screen regions for vision reward)

### Requirement 4: Checkpoint Management

**User Story:** As a researcher, I want periodic model checkpoints during training, so that I can resume interrupted training and select the best-performing snapshot.

#### Acceptance Criteria

1. WHEN the checkpoint interval is reached during training, THE Training_Pipeline SHALL save the Agent weights and optimizer state to the output directory
2. THE Training_Pipeline SHALL name checkpoint files with the training step number for identification
3. WHEN training is resumed from a checkpoint, THE Training_Pipeline SHALL load the Agent weights and optimizer state and continue training from the saved step
4. THE Training_Pipeline SHALL retain the most recent N checkpoints as configured, deleting older checkpoints to limit disk usage
5. WHEN a checkpoint file is corrupted or missing, THE Training_Pipeline SHALL log an error and fall back to the most recent valid checkpoint

### Requirement 5: Training Callbacks and Logging

**User Story:** As a researcher, I want real-time visibility into training progress, so that I can monitor convergence and detect problems early.

#### Acceptance Criteria

1. THE Training_Pipeline SHALL register a Callback that logs episode reward, episode length, and FPS at the configured log interval
2. THE Training_Pipeline SHALL support TensorBoard logging of training metrics when TensorBoard is available
3. THE Metrics_Tracker SHALL record per-episode reward, episode length, and wall-clock time
4. THE Metrics_Tracker SHALL compute rolling averages over a configurable window size for reward and episode length
5. WHEN the rolling average reward does not improve for a configurable number of steps, THE Training_Pipeline SHALL log a warning indicating potential training stagnation

### Requirement 6: Real-Time Inference

**User Story:** As a user, I want to watch a trained agent play a game in real-time at 60 FPS, so that I can evaluate agent behavior visually.

#### Acceptance Criteria

1. WHEN a trained Agent path is provided, THE Inference_Runner SHALL load the Agent and run it against the emulator at the target frame rate
2. THE Inference_Runner SHALL default to 60 FPS as the target frame rate and accept a configurable override
3. THE Inference_Runner SHALL output each frame as an RGB NumPy array suitable for rendering or video recording
4. WHILE the Inference_Runner is active, THE Inference_Runner SHALL execute the Startup_Sequence from the Game_Profile before gameplay begins
5. IF the Agent inference takes longer than the frame budget, THEN THE Inference_Runner SHALL skip frames to maintain real-time pacing and log the number of skipped frames
6. WHEN the game episode ends, THE Inference_Runner SHALL reset the environment and start a new episode automatically

### Requirement 7: Evaluation Module

**User Story:** As a researcher, I want to evaluate a trained agent over multiple episodes with deterministic seeds, so that I can measure performance reliably.

#### Acceptance Criteria

1. WHEN an Agent path and episode count are provided, THE Evaluation_Module SHALL run the Agent for the specified number of episodes and collect metrics
2. THE Evaluation_Module SHALL record total reward, episode length, and game-specific score (when available from info dict) for each episode
3. THE Evaluation_Module SHALL compute mean, standard deviation, minimum, and maximum for each metric across all episodes
4. THE Evaluation_Module SHALL support deterministic evaluation by accepting a base seed and incrementing it for each episode
5. THE Evaluation_Module SHALL save evaluation results to a JSON file in the output directory
6. THE Evaluation_Module SHALL complete evaluation without requiring a display or GUI

### Requirement 8: Metrics Persistence and Export

**User Story:** As a researcher, I want training and evaluation metrics saved to disk, so that I can analyze results after training and compare experiments.

#### Acceptance Criteria

1. THE Metrics_Tracker SHALL save training metrics (episode rewards, episode lengths, timestamps) to a CSV file during training
2. THE Metrics_Tracker SHALL flush metrics to disk at the configured log interval to prevent data loss on crash
3. WHEN training completes, THE Metrics_Tracker SHALL write a summary JSON file containing final statistics (total episodes, mean reward, best reward, total timesteps, wall-clock duration)
4. THE Metrics_Tracker SHALL append to existing metrics files when resuming training from a checkpoint

### Requirement 9: Multi-Emulator Game-Agnostic Design

**User Story:** As a framework developer, I want the training pipeline to work identically across emulators, so that adding a new emulator does not require changes to training code.

#### Acceptance Criteria

1. THE Training_Pipeline SHALL interact with emulators exclusively through the BaseEnv and GymnasiumWrapper interfaces
2. THE Training_Pipeline SHALL NOT contain emulator-specific logic or conditional branches based on emulator type
3. WHEN a new emulator is added to the framework, THE Training_Pipeline SHALL support training on the new emulator without code modifications, requiring only a new Game_Profile
4. THE Game_Profile system SHALL support Videopac and MO5 emulator types
5. THE Training_Pipeline SHALL validate that the specified emulator type is available before starting training

### Requirement 10: Reward Tuning Support

**User Story:** As a researcher, I want to easily switch and combine reward modes per game, so that I can find the reward signal that produces the best agent behavior.

#### Acceptance Criteria

1. THE Training_Config SHALL allow specifying the reward mode (survival, memory, vision, intrinsic) per training run
2. THE Training_Config SHALL allow specifying reward-specific parameters (memory addresses, vision regions, intrinsic novelty scale) that override Game_Profile defaults
3. WHEN multiple reward modes are specified with weights, THE Training_Pipeline SHALL combine the reward signals using the configured weights
4. THE Training_Pipeline SHALL log the active reward mode and parameters at the start of each training run

### Requirement 11: Video Recording

**User Story:** As a researcher, I want to record agent gameplay as video files, so that I can share results and review agent behavior offline.

#### Acceptance Criteria

1. WHEN video recording is enabled in the Inference_Runner or Evaluation_Module, THE Training_Pipeline SHALL save gameplay frames as an MP4 video file
2. THE Training_Pipeline SHALL record video at the native emulator resolution and frame rate
3. THE Training_Pipeline SHALL overlay episode reward and step count on recorded frames when the overlay option is enabled
4. IF the video encoding library is not available, THEN THE Training_Pipeline SHALL log a warning and continue without recording

### Requirement 12: CLI Entry Point

**User Story:** As a user, I want command-line tools to train, evaluate, and run inference, so that I can use the pipeline without writing Python scripts.

#### Acceptance Criteria

1. THE CLI SHALL provide a "train" command that accepts a Training_Config file path and starts training
2. THE CLI SHALL provide an "evaluate" command that accepts an Agent path, a Game_Profile, and an episode count
3. THE CLI SHALL provide an "play" command that loads an Agent and runs real-time inference with optional video recording
4. THE CLI SHALL provide a "list-games" command that lists all available Game_Profiles
5. WHEN an invalid command or missing argument is provided, THE CLI SHALL display a usage message with available commands and required arguments
