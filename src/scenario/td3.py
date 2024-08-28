# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/td3/#td3_continuous_actionpy
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
import gymnasium as gym

from regelum.simulator import Simulator
from regelum.scenario import Scenario
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


class TD3Scenario(Scenario):
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
        policy_frequency : int = 2,
        noise_clip: float = 0.5,
        exploration_noise: float = 0.1,
        learning_rate: float = 3e-4,
        policy_noise: float = 0.2,
    ):
        self.simulator = simulator
        self.running_objective = running_objective
        self.device = device
        self.total_timesteps = total_timesteps
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.policy_frequency  = policy_frequency 
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

        def make_env(env):
            def thunk():
                wrapped_env = gym.wrappers.RecordEpisodeStatistics(env)
                return wrapped_env

            return thunk

        self.envs = gym.vector.SyncVectorEnv(
            [make_env(RgEnv(simulator, running_objective))]
        )
        self.N_episodes = int(
            total_timesteps / simulator.time_final * simulator.max_step
        )
        self.episode_id = 1
        self.N_iterations = 1
        self.value = 0

    @apply_callbacks()
    def post_compute_action(self, state, obs, action, reward, time, global_step):
        self.current_running_objective = reward
        self.value += reward
        return {
            "estimated_state": state,
            "observation": obs,
            "time": time,
            "episode_id": self.episode_id,
            "iteration_id": 1,
            "step_id": global_step,
            "action": action,
            "running_objective": reward,
            "current_value": self.value,
        }

    @apply_callbacks()
    def save_episodic_return(self, episodic_return, global_step):
        return {
            "global_step": global_step,
            "charts/episodic_return": episodic_return,
        }

    @apply_callbacks()
    def save_losses(self, global_step, **losses):
        return {"losses/" + loss: losses[loss] for loss in losses} | {
            "global_step": global_step
        }

    @apply_callbacks()
    def reset_episode(self):
        self.episode_id += 1

    @apply_callbacks()
    def reload_scenario(self):
        self.recent_value = self.value
        self.value = 0

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
                        0, actor.action_scale * self.exploration_noise
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
                    print(
                        f"global_step={global_step}, episodic_return={info['episode']['r']}"
                    )
                    self.save_episodic_return(
                        global_step=global_step, episodic_return=info["episode"]["r"]
                    )
                    self.reload_scenario()
                    self.reset_episode()
                    # Add logging here if needed

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
                    ).clamp(self.action_bounds[:, 0], self.action_bounds[:, 1])

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

                if global_step % self.policy_frequency  == 0:
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
                if global_step % 100:
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
