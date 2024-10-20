"""Twin Delayed Deep Deterministic Policy Gradient (TD3) implementation.

This file contains the implementation of the TD3 algorithm, adapted from the CleanRL repository
(https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/td3_continuous_action.py) to work with
the regelum framework.

Key components:
1. Actor network
2. Critic networks (Q-functions)
3. TD3Scenario class, which inherits from CleanRLScenario

The TD3 algorithm improves upon DDPG by using two critic networks to reduce overestimation bias,
delayed policy updates, and target policy smoothing.

Main features:
- Integration with regelum's Simulator and RunningObjective classes
- Use of stable-baselines3 ReplayBuffer for efficient experience storage
- Customizable hyperparameters for easy experimentation

This implementation allows for seamless integration with regelum's ecosystem while maintaining
the core TD3 algorithm structure.
"""

from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
import gymnasium as gym

from regelum.simulator import Simulator
from .base import CleanRLScenario
from regelum.objective import RunningObjective
from src.rgenv import RgEnv


class Actor(nn.Module):
    def __init__(
        self, dim_action: int, dim_observation: int, action_bounds: np.ndarray
    ):
        super().__init__()
        self.fc1 = nn.Linear(dim_observation, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, dim_action)
        self.action_bounds = action_bounds

        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (action_bounds[:, 1] - action_bounds[:, 0]) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (action_bounds[:, 1] + action_bounds[:, 0]) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc_mu(x)) * self.action_scale + self.action_bias


class QNetwork(nn.Module):
    def __init__(self, dim_action: int, dim_observation: int):
        super().__init__()
        self.fc1 = nn.Linear(dim_observation + dim_action, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class TD3Scenario(CleanRLScenario):
    def __init__(
        self,
        simulator: Simulator,
        running_objective: RunningObjective,
        device: str = "cuda:0",
        total_timesteps: int = 1000000,
        buffer_size: int = 1000000,
        gamma: float = 0.99,
        tau: float = 0.005,
        batch_size: int = 256,
        learning_starts: int = 25000,
        policy_frequency: int = 2,
        noise_clip: float = 0.5,
        exploration_noise: float = 0.1,
        learning_rate: float = 3e-4,
        policy_noise: float = 0.2,
    ):
        """
        Initialize the TD3Scenario.

        Args:
            simulator: The simulator object.
            running_objective: The running objective for the scenario.
            device: The device to run the computations on.
            total_timesteps: Total number of timesteps for the scenario.
            buffer_size: Size of the replay buffer.
            gamma: Discount factor for future rewards.
            tau: Soft update coefficient for target networks.
            batch_size: Batch size for training.
            learning_starts: Number of timesteps before learning starts.
            policy_frequency: Frequency of policy updates.
            noise_clip: Maximum value of the noise added to target policy.
            exploration_noise: Standard deviation of Gaussian exploration noise.
            learning_rate: Learning rate for the optimizer.
            policy_noise: Standard deviation of Gaussian noise added to policy.
        """
        super().__init__(
            simulator=simulator,
            running_objective=running_objective,
            total_timesteps=total_timesteps,
            device=device,
        )
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.policy_frequency = policy_frequency
        self.noise_clip = noise_clip
        self.exploration_noise = exploration_noise
        self.learning_rate = learning_rate
        self.policy_noise = policy_noise

        dim_action, dim_observation, self.action_bounds = (
            simulator.system._dim_inputs,
            simulator.system._dim_observation,
            np.array(simulator.system._action_bounds),
        )

        self.actor = Actor(dim_action, dim_observation, self.action_bounds).to(device)
        self.actor_target = Actor(dim_action, dim_observation, self.action_bounds).to(
            device
        )
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)

        self.qf1 = QNetwork(dim_action, dim_observation).to(device)
        self.qf2 = QNetwork(dim_action, dim_observation).to(device)
        self.qf1_target = QNetwork(dim_action, dim_observation).to(device)
        self.qf2_target = QNetwork(dim_action, dim_observation).to(device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        self.q_optimizer = optim.Adam(
            list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=learning_rate
        )

        self.rb = ReplayBuffer(
            buffer_size,
            observation_space=gym.spaces.Box(-np.inf, np.inf, shape=(dim_observation,)),
            action_space=gym.spaces.Box(
                low=self.action_bounds[:, 0], high=self.action_bounds[:, 1]
            ),
            device=device,
            handle_timeout_termination=False,
        )

    def run(self):
        obs, _ = self.envs.reset()
        for global_step in range(self.total_timesteps):
            # ALGO LOGIC: put action logic here
            if global_step < self.learning_starts:
                actions = np.array(
                    [
                        np.random.uniform(
                            low=self.action_bounds[:, 0],
                            high=self.action_bounds[:, 1],
                        )
                    ]
                )
            else:
                with torch.no_grad():
                    actions = self.actor(torch.Tensor(obs).to(self.device))
                    actions += torch.normal(
                        0, self.actor.action_scale * self.exploration_noise
                    )
                    actions = (
                        actions.cpu()
                        .numpy()
                        .clip(self.action_bounds[:, 0], self.action_bounds[:, 1])
                    )

            self.state = self.envs.envs[0].env.state.reshape(1, -1)
            self.time = self.envs.envs[0].env.simulator.time
            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, terminations, truncations, infos = self.envs.step(
                actions
            )
            # We need state and time for logging, so we extracted it 2 lines above
            # before calling the step method
            self.post_compute_action(
                self.state,
                obs,
                actions,
                float(rewards.reshape(-1)),
                self.time,
                global_step,
            )
            # TRY NOT TO MODIFY: record rewards for plotting purposes
            if "final_info" in infos:
                for info in infos["final_info"]:
                    self.save_episodic_return(
                        global_step=global_step, episodic_return=info["episode"]["r"]
                    )
                    self.reload_scenario()
                    self.reset_episode()
                    self.reset_iteration()

            # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
            real_next_obs = next_obs.copy()
            for idx, trunc in enumerate(truncations):
                if trunc:
                    real_next_obs[idx] = infos["final_observation"][idx]
            self.rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs

            # ALGO LOGIC: training.
            if global_step > self.learning_starts:
                data = self.rb.sample(self.batch_size)
                with torch.no_grad():
                    clipped_noise = (
                        torch.randn_like(data.actions, device=self.device)
                        * self.policy_noise
                    ).clamp(
                        -self.noise_clip, self.noise_clip
                    ) * self.actor_target.action_scale

                    next_state_actions = (
                        self.actor_target(data.next_observations) + clipped_noise
                    ).clamp(
                        self.envs.single_action_space.low[0],
                        self.envs.single_action_space.high[0],
                    )

                    qf1_next_target = self.qf1_target(
                        data.next_observations, next_state_actions
                    )
                    qf2_next_target = self.qf2_target(
                        data.next_observations, next_state_actions
                    )
                    min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
                    next_q_value = data.rewards.flatten() + (
                        1 - data.dones.flatten()
                    ) * self.gamma * (min_qf_next_target).view(-1)

                qf1_a_values = self.qf1(data.observations, data.actions).view(-1)
                qf2_a_values = self.qf2(data.observations, data.actions).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                qf_loss = qf1_loss + qf2_loss

                # optimize the model
                self.q_optimizer.zero_grad()
                qf_loss.backward()
                self.q_optimizer.step()

                actor_loss = None
                if global_step % self.policy_frequency == 0:
                    actor_loss = -self.qf1(
                        data.observations, self.actor(data.observations)
                    ).mean()
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()

                    # update the target network
                    for param, target_param in zip(
                        self.actor.parameters(), self.actor_target.parameters()
                    ):
                        target_param.data.copy_(
                            self.tau * param.data + (1 - self.tau) * target_param.data
                        )
                    for param, target_param in zip(
                        self.qf1.parameters(), self.qf1_target.parameters()
                    ):
                        target_param.data.copy_(
                            self.tau * param.data + (1 - self.tau) * target_param.data
                        )
                    for param, target_param in zip(
                        self.qf2.parameters(), self.qf2_target.parameters()
                    ):
                        target_param.data.copy_(
                            self.tau * param.data + (1 - self.tau) * target_param.data
                        )

                if global_step % 100 and actor_loss is not None:
                    # We use callbacks functionality to save losses and metrics
                    # The save_losses method is implemented in the base class
                    # and handles the logging through callbacks
                    self.save_losses(
                        global_step=global_step,
                        actor_loss=actor_loss.item(),
                        qf1_loss=qf1_loss.item(),
                        qf2_loss=qf2_loss.item(),
                        qf_loss=qf_loss.item() / 2.0,
                        qf1_values=qf1_a_values.mean().item(),
                        qf2_values=qf2_a_values.mean().item(),
                    )
        self.envs.close()
