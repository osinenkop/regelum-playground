import torch
import numpy as np
import gymnasium as gym
from regelum.simulator import Simulator
from regelum.objective import RunningObjective
from regelum.scenario import Scenario
from regelum.callback import Callback
from src.rgenv import RgEnv
import mlflow


class CleanRLScenario(Scenario):
    def __init__(
        self,
        simulator: Simulator,
        running_objective: RunningObjective,
        total_timesteps: int,
        device: str,
    ):
        self.total_timesteps = total_timesteps
        self.device = device
        self.simulator = simulator
        self.running_objective = running_objective

        def make_env(env):
            def thunk():
                return gym.wrappers.RecordEpisodeStatistics(env)

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
        raise NotImplementedError("Subclasses must implement the run method")


class CleanRLCallback(Callback):
    def log_metrics(self, metrics):
        global_step = metrics["global_step"]
        for metric, value in metrics.items():
            if metric != "global_step":
                mlflow.log_metric(metric, value, global_step)

    def is_target_event(self, obj, method, output, triggers):
        return isinstance(obj, CleanRLScenario) and method in [
            "save_episodic_return",
            "save_losses",
        ]

    def on_function_call(self, obj, method, output):
        self.log_metrics(metrics=output)
