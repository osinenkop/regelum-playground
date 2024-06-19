from regelum.scenario import RLScenario, Scenario
from regelum.utils import Clock, AwaitedParameter, calculate_value
import numpy as np
import torch
import torch.multiprocessing as mp
import random

from regelum.data_buffers import DataBuffer

from regelum.utils import Clock, AwaitedParameter, calculate_value
from regelum import RegelumBase
from regelum.policy import (
    Policy,
    RLPolicy,
    PolicyPPO,
    PolicyReinforce,
    PolicySDPG,
    PolicyDDPG,
)
from .critic import CriticPPO
from regelum.critic import Critic, CriticCALF, CriticTrivial
from typing import Optional, Union, Type, Dict, List, Any, Callable
from regelum.objective import RunningObjective
from regelum.data_buffers import DataBuffer
from regelum.event import Event
from regelum.simulator import Simulator
from regelum.constraint_parser import ConstraintParser, ConstraintParserTrivial
from regelum.observer import Observer, ObserverTrivial
from regelum.model import (
    Model,
    ModelNN,
    ModelPerceptron,
    ModelWeightContainer,
    ModelWeightContainerTorch,
    PerceptronWithTruncatedNormalNoise,
    ModelQuadLin,
)
from regelum.predictor import Predictor, EulerPredictor
from regelum.optimizable.core.configs import (
    TorchOptimizerConfig,
    CasadiOptimizerConfig,
)
from regelum.data_buffers.batch_sampler import RollingBatchSampler, EpisodicSampler
from copy import deepcopy
from typing_extensions import Self


def get_policy_gradient_kwargs(
    sampling_time: float,
    running_objective: RunningObjective,
    simulator: Simulator,
    discount_factor: float,
    observer: Optional[Observer],
    N_episodes: int,
    N_iterations: int,
    value_threshold: float,
    policy_type: Type[Policy],
    policy_model: PerceptronWithTruncatedNormalNoise,
    policy_n_epochs: int,
    policy_opt_method_kwargs: Dict[str, Any],
    policy_opt_method: Type[torch.optim.Optimizer],
    is_reinstantiate_policy_optimizer: bool,
    critic_model: Optional[ModelPerceptron] = None,
    critic_opt_method: Optional[Type[torch.optim.Optimizer]] = None,
    critic_opt_method_kwargs: Optional[Dict[str, Any]] = None,
    critic_n_epochs: Optional[int] = None,
    critic_td_n: Optional[int] = None,
    critic_kwargs: Dict[str, Any] = None,
    critic_is_value_function: Optional[bool] = None,
    is_reinstantiate_critic_optimizer: Optional[bool] = None,
    policy_kwargs: Dict[str, Any] = None,
    scenario_kwargs: Dict[str, Any] = None,
    is_use_critic_as_policy_kwarg: bool = True,
    stopping_criterion: Optional[Callable[[DataBuffer], bool]] = None,
    device: str = "cpu",
):
    system = simulator.system
    if critic_model is not None:
        assert (
            critic_n_epochs is not None
            and critic_td_n is not None
            and critic_opt_method is not None
            and critic_opt_method_kwargs is not None
            and is_reinstantiate_critic_optimizer is not None
            and critic_is_value_function is not None
        ), "critic_n_epochs, critic_td_n, critic_opt_method, critic_opt_method_kwargs, is_reinstantiate_critic_optimizer, critic_is_value_function should be set"
        critic = CriticPPO(
            discount_factor=discount_factor,
            sampling_time=sampling_time,
            model=critic_model,
            td_n=critic_td_n,
            optimizer_config=TorchOptimizerConfig(
                critic_n_epochs,
                data_buffer_iter_bathes_kwargs={
                    "batch_sampler": RollingBatchSampler,
                    "dtype": torch.FloatTensor,
                    "mode": "full",
                    "n_batches": 1,
                    "device": device,
                },
                opt_method_kwargs=critic_opt_method_kwargs,
                opt_method=critic_opt_method,
                is_reinstantiate_optimizer=is_reinstantiate_critic_optimizer,
            ),
            **(critic_kwargs if critic_kwargs is not None else dict()),
        )
    else:
        critic = CriticTrivial()

    return dict(
        stopping_criterion=stopping_criterion,
        simulator=simulator,
        discount_factor=discount_factor,
        policy_optimization_event=Event.reset_iteration,
        critic_optimization_event=Event.reset_iteration,
        N_episodes=N_episodes,
        N_iterations=N_iterations,
        running_objective=running_objective,
        observer=observer,
        critic=critic,
        is_critic_first=True,
        value_threshold=value_threshold,
        sampling_time=sampling_time,
        policy=policy_type(
            model=policy_model,
            system=system,
            device=device,
            discount_factor=discount_factor,
            optimizer_config=TorchOptimizerConfig(
                n_epochs=policy_n_epochs,
                opt_method=policy_opt_method,
                opt_method_kwargs=policy_opt_method_kwargs,
                is_reinstantiate_optimizer=is_reinstantiate_policy_optimizer,
                data_buffer_iter_bathes_kwargs={
                    "batch_sampler": RollingBatchSampler,
                    "dtype": torch.FloatTensor,
                    "mode": "full",
                    "n_batches": 1,
                    "device": device,
                },
            ),
            **(dict(critic=critic) if is_use_critic_as_policy_kwarg else dict()),
            **(policy_kwargs if policy_kwargs is not None else dict()),
        ),
        **(scenario_kwargs if scenario_kwargs is not None else dict()),
    )


class Scenario(Scenario):
    def __init__(
        self,
        termination_criterion: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if termination_criterion is not None:
            self.termination_criterion = termination_criterion
        else:
            self.termination_criterion = lambda *args, **kwargs: False

    def step(self):
        if isinstance(self.action_init, AwaitedParameter) and isinstance(
            self.state_init, AwaitedParameter
        ):
            (
                self.state_init,
                self.action_init,
            ) = self.simulator.get_init_state_and_action()

        if (not self.is_episode_ended) and (self.value <= self.value_threshold):
            (
                self.time,
                self.state,
                self.observation,
                self.simulation_metadata,
            ) = self.simulator.get_sim_step_data()

            self.delta_time = (
                self.time - self.time_old
                if self.time_old is not None and self.time is not None
                else 0
            )
            self.time_old = self.time
            if len(list(self.constraint_parser)) > 0:
                self.constraint_parameters = self.constraint_parser.parse_constraints(
                    simulation_metadata=self.simulation_metadata
                )
                self.substitute_constraint_parameters(**self.constraint_parameters)
            estimated_state = self.observer.get_state_estimation(
                self.time, self.observation, self.action
            )

            self.action = self.compute_action_sampled(
                self.time,
                estimated_state,
                self.observation,
            )
            self.simulator.receive_action(self.action)
            self.is_episode_ended = (
                self.simulator.do_sim_step() == -1
                or self.termination_criterion(self.time, self.state, self.action)
            )
            return "episode_continues"
        else:
            return "episode_ended"


class RLScenario(RLScenario):
    def __init__(
        self,
        policy: Policy,
        critic: Critic,
        running_objective: RunningObjective,
        simulator: Simulator,
        policy_optimization_event: Event,
        critic_optimization_event: Event = None,
        discount_factor: float = 1.0,
        is_critic_first: bool = False,
        sampling_time: float = 0.1,
        constraint_parser: Optional[ConstraintParser] = None,
        observer: Optional[Observer] = None,
        N_episodes: int = 1,
        N_iterations: int = 1,
        value_threshold: float = np.inf,
        stopping_criterion: Optional[Callable[[DataBuffer], bool]] = None,
        termination_criterion: Optional[
            Callable[[float, np.ndarray, np.ndarray], bool]
        ] = None,
    ):
        super().__init__(
            policy=policy,
            critic=critic,
            running_objective=running_objective,
            simulator=simulator,
            policy_optimization_event=policy_optimization_event,
            critic_optimization_event=critic_optimization_event,
            discount_factor=discount_factor,
            is_critic_first=is_critic_first,
            sampling_time=sampling_time,
            constraint_parser=constraint_parser,
            observer=observer,
            N_episodes=N_episodes,
            N_iterations=N_iterations,
            value_threshold=value_threshold,
            stopping_criterion=stopping_criterion,
        )
        if termination_criterion is not None:
            self.termination_criterion = termination_criterion
        else:
            self.termination_criterion = lambda *args, **kwargs: False

    def step(self):
        if isinstance(self.action_init, AwaitedParameter) and isinstance(
            self.state_init, AwaitedParameter
        ):
            (
                self.state_init,
                self.action_init,
            ) = self.simulator.get_init_state_and_action()

        if (not self.is_episode_ended) and (self.value <= self.value_threshold):
            (
                self.time,
                self.state,
                self.observation,
                self.simulation_metadata,
            ) = self.simulator.get_sim_step_data()

            self.delta_time = (
                self.time - self.time_old
                if self.time_old is not None and self.time is not None
                else 0
            )
            self.time_old = self.time
            if len(list(self.constraint_parser)) > 0:
                self.constraint_parameters = self.constraint_parser.parse_constraints(
                    simulation_metadata=self.simulation_metadata
                )
                self.substitute_constraint_parameters(**self.constraint_parameters)
            estimated_state = self.observer.get_state_estimation(
                self.time, self.observation, self.action
            )

            self.action = self.compute_action_sampled(
                self.time,
                estimated_state,
                self.observation,
            )
            self.simulator.receive_action(self.action)
            self.is_episode_ended = (
                self.simulator.do_sim_step() == -1
                or self.termination_criterion(self.time, self.state, self.action)
            )
            return "episode_continues"
        else:
            return "episode_ended"


class PPO(RLScenario):
    """Scenario for Proximal Polizy Optimization.

    PPOScenario is a reinforcement learning scenario implementing the Proximal Policy Optimization (PPO) algorithm.
    This algorithm uses a policy gradient approach with an objective function designed to reduce the variance
    of policy updates, while ensuring that the new policy does not deviate significantly from the old one.
    """

    def __init__(
        self,
        policy_model: PerceptronWithTruncatedNormalNoise,
        critic_model: ModelPerceptron,
        sampling_time: float,
        running_objective: RunningObjective,
        simulator: Simulator,
        critic_n_epochs: int,
        policy_n_epochs: int,
        critic_opt_method_kwargs: Dict[str, Any],
        policy_opt_method_kwargs: Dict[str, Any],
        critic_opt_method: Type[torch.optim.Optimizer] = torch.optim.Adam,
        policy_opt_method: Type[torch.optim.Optimizer] = torch.optim.Adam,
        running_objective_type: str = "cost",
        critic_td_n: int = 1,
        cliprange: float = 0.2,
        discount_factor: float = 0.7,
        observer: Optional[Observer] = None,
        N_episodes: int = 2,
        N_iterations: int = 100,
        value_threshold: float = np.inf,
        stopping_criterion: Optional[Callable[[DataBuffer], bool]] = None,
        gae_lambda: float = 0.0,
        is_normalize_advantages: bool = True,
        device: str = "cpu",
        entropy_coeff: float = 0.0,
        termination_criterion: Optional[
            Callable[[float, np.ndarray, np.ndarray], bool]
        ] = None,
    ):
        """Initialize the object with the given parameters.

        Args:
            policy_model (PerceptronWithTruncatedNormalNoise): The
                neural network model parameterizing the policy.
            critic_model (ModelPerceptron): The neural network model
                used for the value function approximation.
            sampling_time (float): Time interval between two consecutive
                actions.
            running_objective (RunningObjective): A function that
                returns the scalar running cost or reward associated
                with an observation-action pair.
            simulator (Simulator): The simulation environment where the
                agent performs actions.
            critic_n_epochs (int): The number of epochs for which the
                critic is trained per iteration.
            policy_n_epochs (int): The number of epochs for which the
                policy is trained per iteration.
            critic_opt_method_kwargs (Dict[str, Any]): A dictionary of
                keyword arguments for the optimizer used for the critic.
            policy_opt_method_kwargs (Dict[str, Any]): A dictionary of
                keyword arguments for the optimizer used for the policy.
            critic_opt_method (Type[torch.optim.Optimizer]): The
                optimization algorithm class used for training the
                critic, e.g. torch.optim.Adam.
            policy_opt_method (Type[torch.optim.Optimizer]): The
                optimization algorithm class used for training the
                policy, e.g. torch.optim.Adam.
            running_objective_type (str): Specifies whether the running
                objective represents a 'cost' to minimize or a 'reward'
                to maximize.
            critic_td_n (int): The n-step temporal-difference parameter
                used for critic updates.
            epsilon (float): Clipping parameter that restricts the
                deviation of the new policy from the old policy.
            discount_factor (float): A factor applied to future rewards
                or costs to discount their value relative to immediate
                ones.
            observer (Optional[Observer]): Object responsible for state
                estimation from observations.
            N_episodes (int): The number of episodes to run in every
                iteration.
            N_iterations (int): The number of iterations to run in the
                scenario.
            value_threshold (float): Threshold of the value to
                end an episode.

        Raises:
            AssertionError: If the `running_objective_type` is invalid.

        Returns:
            None
        """
        assert (
            running_objective_type == "cost" or running_objective_type == "reward"
        ), f"Invalid 'running_objective_type' value: '{running_objective_type}'. It must be either 'cost' or 'reward'."
        super().__init__(
            **get_policy_gradient_kwargs(
                sampling_time=sampling_time,
                running_objective=running_objective,
                simulator=simulator,
                discount_factor=discount_factor,
                observer=observer,
                N_episodes=N_episodes,
                N_iterations=N_iterations,
                value_threshold=value_threshold,
                policy_type=PolicyPPO,
                policy_model=policy_model,
                policy_opt_method=policy_opt_method,
                policy_opt_method_kwargs=policy_opt_method_kwargs,
                is_reinstantiate_policy_optimizer=True,
                policy_kwargs=dict(
                    cliprange=cliprange,
                    running_objective_type=running_objective_type,
                    sampling_time=sampling_time,
                    gae_lambda=gae_lambda,
                    is_normalize_advantages=is_normalize_advantages,
                    entropy_coeff=entropy_coeff,
                ),
                policy_n_epochs=policy_n_epochs,
                critic_model=critic_model,
                critic_opt_method=critic_opt_method,
                critic_opt_method_kwargs=critic_opt_method_kwargs,
                critic_td_n=critic_td_n,
                critic_n_epochs=critic_n_epochs,
                critic_is_value_function=True,
                is_reinstantiate_critic_optimizer=True,
                stopping_criterion=stopping_criterion,
                device=device,
            ),
            termination_criterion=termination_criterion,
        )
