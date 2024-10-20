"""This file contains the implementation of the Soft Actor-Critic (SAC) algorithm.
The implementation is based on the CleanRL repository:
https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py

We have adapted the original implementation to work with the regelum framework.
Key changes include:
1. Using mlflow for logging instead of Weights & Biases and tensorboard.
2. Utilizing regelum's callback features for smooth logging of metrics and trajectories.
3. Integrating with regelum's Simulator and RunningObjective classes.

The file structure is as follows:
1. Import statements
2. SoftQNetwork class definition
3. Actor class definition
4. SACScenario class definition (which inherits from CleanRLScenario)

The SACScenario class contains the main logic for the SAC algorithm, including:
- Initialization of networks, optimizers, and replay buffer
- Training loop
- Evaluation and logging functions

This implementation allows for easy integration with regelum's ecosystem while
maintaining the core SAC algorithm structure from CleanRL.
"""

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer
from src.rgenv import RgEnv
from torch.distributions.normal import Normal
from regelum.simulator import Simulator
from regelum.objective import RunningObjective
import gymnasium as gym
import mlflow
from .base import CleanRLScenario


class SoftQNetwork(nn.Module):
    def __init__(self, dim_action, dim_observation):
        super().__init__()
        self.fc1 = nn.Linear(
            dim_action + dim_observation,
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(
        self,
        dim_action: int,
        dim_observation: int,
        action_bounds: list[list[float]],
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        self.fc1 = nn.Linear(dim_observation, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, dim_action)
        self.fc_logstd = nn.Linear(256, dim_action)
        self.action_bounds = np.array(action_bounds)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

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
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (
            log_std + 1
        )  # From SpinUp / Denis Yarats
        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


class SACScenario(CleanRLScenario):
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
        learning_starts: int = 5000,
        policy_lr: float = 3.0e-4,
        q_lr: float = 1.0e-3,
        policy_frequency: int = 2,
        target_network_frequency: int = 1,
        alpha: float = 0.2,
        autotune: bool = True,
    ):
        """
        Initializes the Soft Actor-Critic (SAC) scenario.

        Args:
            simulator: The simulator object.
            running_objective: The running objective for the scenario.
            device: The device to run the computations on.
            total_timesteps: Total number of timesteps for the scenario.
            buffer_size: Size of the replay buffer.
            gamma: Discount factor for future rewards.
            tau: Soft update coefficient for target networks.
            batch_size: Batch size for training.
            learning_starts: Number of steps before actor learning starts.
            policy_lr: Learning rate for the policy network.
            q_lr: Learning rate for the Q-networks.
            policy_frequency: Frequency of policy network updates.
            target_network_frequency: Frequency of target network updates.
            alpha: Temperature parameter for entropy regularization.
            autotune: Whether to automatically tune the temperature parameter.
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
        self.policy_lr = policy_lr
        self.q_lr = q_lr
        self.policy_frequency = policy_frequency
        self.target_network_frequency = target_network_frequency
        self.alpha = alpha
        self.autotune = autotune

        dim_action, dim_observation, self.action_bounds = (
            simulator.system._dim_inputs,
            simulator.system._dim_observation,
            np.array(simulator.system._action_bounds),
        )
        self.actor = Actor(dim_action, dim_observation, self.action_bounds).to(device)
        self.qf1 = SoftQNetwork(dim_action, dim_observation).to(device)
        self.qf2 = SoftQNetwork(dim_action, dim_observation).to(device)
        self.qf1_target = SoftQNetwork(dim_action, dim_observation).to(device)
        self.qf2_target = SoftQNetwork(dim_action, dim_observation).to(device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        self.q_optimizer = optim.Adam(
            list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=self.q_lr
        )
        self.actor_optimizer = optim.Adam(
            list(self.actor.parameters()), lr=self.policy_lr
        )
        # Note: This implementation uses the ReplayBuffer from Stable Baselines 3 (SB3).
        # CleanRL utilizes this SB3 replay buffer for efficient experience storage and sampling.
        self.rb = ReplayBuffer(
            buffer_size,
            observation_space=gym.spaces.Box(-np.inf, np.inf, shape=(dim_observation,)),
            action_space=gym.spaces.Box(
                low=self.action_bounds[:, 0], high=self.action_bounds[:, 1]
            ),
            device=device,
            handle_timeout_termination=False,
        )

        if autotune:
            self.target_entropy = -torch.prod(
                torch.Tensor(self.envs.single_action_space.shape).to(device)
            ).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=q_lr)
        else:
            self.alpha = alpha

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
                actions, _, _ = self.actor.get_action(torch.Tensor(obs).to(self.device))
                actions = actions.detach().cpu().numpy()

            self.state = self.envs.envs[0].env.state.reshape(1, -1)
            self.time = self.envs.envs[0].env.simulator.time
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
            if "final_info" in infos:
                for info in infos["final_info"]:
                    self.save_episodic_return(
                        global_step=global_step, episodic_return=self.value
                    )
                    self.reload_scenario()
                    self.reset_episode()
                    self.reset_iteration()
                    break
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
                    next_state_actions, next_state_log_pi, _ = self.actor.get_action(
                        data.next_observations
                    )
                    qf1_next_target = self.qf1_target(
                        data.next_observations, next_state_actions
                    )
                    qf2_next_target = self.qf2_target(
                        data.next_observations, next_state_actions
                    )
                    min_qf_next_target = (
                        torch.min(qf1_next_target, qf2_next_target)
                        - self.alpha * next_state_log_pi
                    )
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
                if (
                    global_step % self.policy_frequency == 0
                ):  # TD 3 Delayed update support
                    for _ in range(
                        self.policy_frequency
                    ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                        pi, log_pi, _ = self.actor.get_action(data.observations)
                        qf1_pi = self.qf1(data.observations, pi)
                        qf2_pi = self.qf2(data.observations, pi)
                        min_qf_pi = torch.min(qf1_pi, qf2_pi)
                        actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
                        self.actor_optimizer.zero_grad()
                        actor_loss.backward()
                        self.actor_optimizer.step()
                        if self.autotune:
                            with torch.no_grad():
                                _, log_pi, _ = self.actor.get_action(data.observations)
                            alpha_loss = (
                                -self.log_alpha.exp() * (log_pi + self.target_entropy)
                            ).mean()
                            self.a_optimizer.zero_grad()
                            alpha_loss.backward()
                            self.a_optimizer.step()
                            self.alpha = self.log_alpha.exp().item()
                # update the target networks
                if global_step % self.target_network_frequency == 0:
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
                if global_step % 100 == 0:
                    # We use callbacks functionality to save losses and metrics
                    # The save_losses method is implemented in the base class
                    # and handles the logging through callbacks
                    self.save_losses(
                        global_step=global_step,
                        qf1_values=qf1_a_values.mean().item(),
                        qf2_values=qf2_a_values.mean().item(),
                        qf1_loss=qf1_loss.item(),
                        qf2_loss=qf2_loss.item(),
                        actor_loss=actor_loss.item(),
                        alpha=self.alpha,
                    )
                    if self.autotune:
                        self.save_losses(
                            global_step=global_step,
                            alpha_loss=alpha_loss.item(),
                        )
        self.envs.close()
