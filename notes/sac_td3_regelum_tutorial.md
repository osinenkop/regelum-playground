Here's a comprehensive tutorial on how SAC (Soft Actor-Critic) and TD3 (Twin Delayed Deep Deterministic Policy Gradient) are implemented in regelum-playground, formatted in markdown:

# SAC and TD3 Implementation in regelum-playground

## Overview

Both SAC and TD3 are implemented as scenarios in regelum-playground, inheriting from the base `CleanRLScenario` class. They are adapted from the CleanRL repository to work within the regelum framework.

## Base Structure: CleanRLScenario

The `CleanRLScenario` class (`src/scenario/base.py`) provides a foundation for both algorithms:

- Initializes common components (simulator, running objective, environment)
- Implements methods for action computation, episode management, and logging
- Uses the `@apply_callbacks()` decorator for integrating with regelum's callback system

## SAC Implementation (src/scenario/sac.py)

### Network Architecture
1. `SoftQNetwork`: Implements the Q-function
2. `Actor`: Implements the policy network with mean and log standard deviation outputs

### SACScenario class
- Initializes networks, optimizers, and replay buffer
- Implements the `run` method containing the main training loop

### Key Features
- Automatic entropy tuning (optional)
- Uses stable-baselines3 ReplayBuffer for efficient experience storage
- Implements soft updates for target networks

### Training Loop
1. Collects experiences using the current policy
2. Updates Q-functions and policy network
3. Optionally updates the temperature parameter (alpha) if autotune is enabled

## TD3 Implementation (src/scenario/td3.py)

### Network Architecture
1. `Actor`: Implements the deterministic policy network
2. `QNetwork`: Implements the Q-function

### TD3Scenario class
- Initializes actor, critic networks, target networks, optimizers, and replay buffer
- Implements the `run` method for the main training loop

### Key Features
- Uses two critic networks to reduce overestimation bias
- Implements delayed policy updates
- Adds clipped noise to target actions for smoothing

### Training Loop
1. Collects experiences using the current policy with added exploration noise
2. Updates critic networks
3. Updates the actor network and target networks at a lower frequency

## Integration with regelum

Both implementations integrate with regelum through:

1. Simulator and RunningObjective:
   - Use regelum's `Simulator` and `RunningObjective` classes for environment interaction and reward calculation

2. Callback System:
   - Utilize the `@apply_callbacks()` decorator for logging and visualization
   - Implement `save_episodic_return` and `save_losses` methods for logging metrics

3. Configuration:
   - Use regelum's configuration system (rehydra) for easy hyperparameter tuning

## Usage

To run SAC or TD3 experiments:

1. Configure the experiment in `presets/main.yaml`:
   - Set the appropriate scenario (sac or td3)
   - Configure system, simulator, and other parameters

2. Run the experiment:
   ```bash
   python run.py scenario=sac  # or td3
   ```

3. For SAC with specific hyperparameters:
   ```bash
   python run.py \
       scenario=sac \
       system=pendulum_with_gym_observation \
       running_objective=gym_pendulum \
       simulator=casadi_random_state_init \
       scenario.autotune=False \
       scenario.policy_lr=0.00079 \
       scenario.q_lr=0.00025 \
       scenario.alpha=0.0085 \
       +seed=4 
   ```

4. For TD3:
   ```bash
   python run.py \
       scenario=td3 \
       system=pendulum_with_gym_observation \
       running_objective=gym_pendulum \
       simulator=casadi_random_state_init  \
       +seed=4   
   ```

## Logging and Visualization

- Both implementations use MLflow for logging metrics
- The `CleanRLCallback` class handles the logging of metrics during training
- Episodic returns and various losses are logged and can be visualized using MLflow's UI

This tutorial provides an overview of how SAC and TD3 are implemented in regelum-playground, highlighting their integration with the regelum framework and key features of each algorithm.