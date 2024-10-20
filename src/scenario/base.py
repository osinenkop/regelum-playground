"""This file contains the base scenario classes for reinforcement learning algorithms implemented in the CleanRL style.

The file is structured as follows:
1. Import statements for necessary libraries and modules.
2. Definition of the CleanRLScenario base class, which provides a foundation for implementing RL scenarios.
3. The CleanRLScenario class includes:
   - Initialization method to set up the scenario with simulator, objective, and other parameters.
   - Methods for environment creation, episode management, and logging.
   - Placeholder methods for action computation and scenario-specific logic.
"""

import numpy as np
import gymnasium as gym
from regelum.simulator import Simulator
from regelum.objective import RunningObjective
from regelum.scenario import Scenario
from regelum.callback import Callback
from src.rgenv import RgEnv
import mlflow
from typing import Any
import torch


class CleanRLScenario(Scenario):
    """Base class for CleanRL-style scenarios.

    https://docs.cleanrl.dev/

    This class provides a foundation for implementing reinforcement learning scenarios
    using the CleanRL style within the regelum framework. It sets up the environment,
    handles episode management, and provides methods for action computation and
    episodic return logging.
    """

    def __init__(
        self,
        simulator: Simulator,
        running_objective: RunningObjective,
        total_timesteps: int,
        device: str,
    ):
        """Initialize the CleanRLScenario.

        This method sets up the scenario with the given simulator, running objective,
        total timesteps, and device. It initializes the environment, episode management,
        and other necessary attributes for running the scenario.

        Args:
            simulator: The simulator object used for the environment.
            running_objective: The running objective function for reward calculation.
            total_timesteps: The total number of timesteps to run the scenario.
            device: The device (e.g., 'cpu' or 'cuda') to run computations on.
        """
        self.total_timesteps = total_timesteps
        self.device = (
            "cpu"
            if device.startswith("cuda") and not torch.cuda.is_available()
            else device
        )
        self.simulator = simulator
        self.running_objective = running_objective

        def make_env(env):
            def thunk():
                return gym.wrappers.RecordEpisodeStatistics(env)

            return thunk

        self.envs = gym.vector.SyncVectorEnv(
            [make_env(RgEnv(simulator, running_objective))]
        )

        self.N_iterations = int(
            total_timesteps / simulator.time_final * simulator.max_step
        )
        self.episode_id = 1
        self.iteration_id = 1
        self.N_episodes = 1
        self.value = 0

    @apply_callbacks()
    def post_compute_action(
        self, state, obs, action, reward, time, global_step
    ) -> dict[str, Any]:
        """Post-process the computed action and update scenario state.

        Called after an action is computed and applied to the environment.
        Updates the current running objective and cumulative value.

        This method is decorated with @apply_callbacks(), which triggers it
        within the scope of callbacks defined in presets/main.yaml. The output
        is used by ScenarioStepLogger to print logs and HistoricalDataCallback
        to save episodic trajectories in h5 and svg formats in the regelum_data folder.

        Args:
            state: Current system state.
            obs: Current observation.
            action: Action taken.
            reward: Reward received for the action.
            time: Current simulation time.
            global_step: Global step count.

        Returns:
            dict: Information about the current scenario state, including:
                - estimated_state
                - observation
                - time
                - episode_id
                - iteration_id
                - step_id
                - action
                - running_objective
                - current_value (discounted but `None` due to not needing to track the discounted value)
                - current_undiscounted_value
        """
        self.current_running_objective = reward
        self.value += reward
        return {
            "estimated_state": state,
            "observation": obs,
            "time": time,
            "episode_id": self.episode_id,
            "iteration_id": self.iteration_id,
            "step_id": global_step,
            "action": action,
            "running_objective": reward,
            "current_value": None,
            "current_undiscounted_value": self.value,
        }

    @apply_callbacks()
    def save_episodic_return(self, episodic_return: float, global_step: int):
        """Save the episodic return and log it.

        This method is decorated with @apply_callbacks(), which ensures it's triggered
        within the scope of callbacks defined in presets/main.yaml. The output of this
        method will be used by CleanRLCallback to log metrics.

        Args:
            episodic_return: The return (cumulative reward) for the episode.
            global_step: The global step count.

        Returns:
            A dictionary containing the global step and the episodic return for logging.
        """
        return {
            "global_step": global_step,
            "charts/episodic_return": episodic_return,
        }

    @apply_callbacks()
    def save_losses(self, global_step, **losses):
        """Save the losses and log them.

        This method is decorated with @apply_callbacks(), which ensures it's triggered
        within the scope of callbacks defined in presets/main.yaml. The output of this
        method will be used by CleanRLCallback to log metrics.

        Args:
            global_step: The global step count.
            **losses: Arbitrary keyword arguments representing different losses.

        Returns:
            A dictionary containing the global step and the losses for logging.
        """
        return {"losses/" + loss: losses[loss] for loss in losses} | {
            "global_step": global_step
        }

    @apply_callbacks()
    def reset_episode(self):
        """Reset the episode.

        This method triggers callbacks inside regelum, allowing the callbacks
        to be aware of the current state in the pipeline. This is a technical
        feature that facilitates proper callback execution and tracking.
        """
        pass

    @apply_callbacks()
    def reload_scenario(self):
        """Reload the scenario.

        This method triggers callbacks inside regelum, allowing the callbacks
        to be aware of the current state in the pipeline. This is a technical
        feature that facilitates proper callback execution and tracking.
        """
        self.recent_undiscounted_value = self.value
        self.value = 0

    def run(self):
        raise NotImplementedError("Subclasses must implement the run method")

    @apply_callbacks()
    def reset_iteration(self):
        """Reset the iteration and trigger callbacks in Regelum.

        This method updates the iteration ID and allows callbacks to track the
        current state in the pipeline. In Regelum, an iteration refers to the point
        when policy gradient methods perform their updates.
        """
        self.iteration_id += 1


class CleanRLCallback(Callback):
    """A callback for logging metrics in CleanRL scenarios.

    This class inherits from Callback to utilize regelum's callback feature
    (https://regelum.aidynamic.group/notebooks/callbacks/). It's designed to work
    in conjunction with the @apply_callbacks() decorator used in CleanRLScenario
    methods. When methods decorated with @apply_callbacks() are called, this
    callback will be triggered if it's added to the main.yaml configuration.

    The primary purpose of this callback is to log metrics using MLflow. It
    intercepts the output of specific methods in CleanRLScenario and logs
    the metrics accordingly.

    Note: This callback should be added to the main.yaml configuration to be
    triggered during the execution of CleanRLScenario methods.
    """

    def log_metrics(self, metrics: dict[str, Any]):
        """Log metrics using MLflow.

        This method takes a dictionary of metrics and logs them using MLflow.
        It extracts the global step from the metrics dictionary and logs each
        metric with its corresponding value and the global step.

        Args:
            metrics: A dictionary containing metrics to be logged.
                The dictionary should have a 'global_step' key and other keys
                representing different metrics.

        Returns:
            None

        Note:
            This method assumes that MLflow has been properly initialized and
            configured before calling.
        """
        global_step = metrics["global_step"]
        for metric, value in metrics.items():
            if metric != "global_step":
                mlflow.log_metric(metric, value, global_step)

    def is_target_event(self, obj, method, output, triggers):
        """Determines if the current event is a target event for logging.

        This method should be overridden in subclasses due to inheritance.
        It is used by the @apply_callbacks decorator to determine whether
        to trigger the callback.

        Args:
            obj: The object instance that triggered the event.
            method: The name of the method that was called.
            output: The output of the method call.
            triggers: Additional trigger conditions (not used in this implementation).

        Returns:
            A boolean indicating whether this is a target event for logging.
        """
        return isinstance(obj, CleanRLScenario) and method in [
            "save_episodic_return",
            "save_losses",
        ]

    def on_function_call(self, obj, method, output):
        """Called when a target event is triggered.

        This method is invoked when is_target_event() returns True for a particular
        function call. It logs the metrics output by the triggered method.

        Args:
            obj: The object instance that triggered the event.
            method: The name of the method that was called.
            output: The output of the method call, expected to be a dictionary of metrics.

        Returns:
            None

        Note:
            This method assumes that the output is a dictionary of metrics suitable
            for logging via the log_metrics method.
        """
        self.log_metrics(metrics=output)
