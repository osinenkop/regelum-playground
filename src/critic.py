from regelum.critic import Critic
from regelum.utils import rg
from regelum.system import System, ComposedSystem
from regelum.model import ModelNN, PerceptronWithTruncatedNormalNoise
from regelum.optimizable.core.configs import OptimizerConfig
from regelum.optimizable import Optimizable
from typing import Union, Optional
from regelum.objective import temporal_difference_objective_full
import torch


def calculate_critic_targets(
    running_objectives: torch.Tensor,
    critic_model_outputs: torch.Tensor,
    times: torch.Tensor,
    td_n: int,
    discount_factor: float,
    sampling_time: float,
) -> torch.Tensor:
    """Calculates critic targets for MSE loss.

    Args:
        running_objectives: running objectives in episode
        observations: observations in episode

    Returns:
        Torch tensor of critic targets.
    """
    episode_size = running_objectives.shape[0]
    discount_factors = discount_factor**times

    return torch.vstack(
        [
            torch.sum(
                running_objectives[i : i + td_n]
                * discount_factors[: min(td_n, episode_size - i)]
            )
            + (
                discount_factors[td_n, 0] * critic_model_outputs[i + td_n, 0]
                if i + td_n < episode_size
                else 0
            )
            for i in range(episode_size)
        ]
    )


def calculate_critic_targets_old(
    running_objective, discount_factor, sampling_time, td_n, critic_model_output
):

    batch_size = running_objective.shape[0]
    assert batch_size > td_n, f"batch size {batch_size} too small for such td_n {td_n}"
    discount_factors = rg.array(
        [[discount_factor ** (sampling_time * i)] for i in range(td_n)],
        prototype=running_objective,
        _force_numeric=True,
    )
    discounted_tdn_sum_of_running_objectives = rg.vstack(
        [
            rg.sum(running_objective[i : i + td_n, :] * discount_factors)
            for i in range(batch_size - td_n)
        ]
    )
    critic_targets = critic_model_output[td_n:, :]

    return (
        discounted_tdn_sum_of_running_objectives
        + discount_factor ** (td_n * sampling_time) * critic_targets
    )


class CriticPPO(Critic):
    def __init__(
        self,
        model: Union[ModelNN],
        device: str = "cpu",
        optimizer_config: Optional[OptimizerConfig] = None,
        discount_factor: float = 1.0,
        sampling_time: float = 0.01,
        entropy_coef: float = 0.02,
        td_n: int = 1,
    ):
        Optimizable.__init__(self, optimizer_config=optimizer_config)
        self.model = model
        self.device = device
        self.discount_factor = discount_factor
        self.sampling_time = sampling_time
        self.entropy_coef = entropy_coef
        self.td_n = td_n
        self.is_on_policy = True
        self.instantiate_optimization_procedure()

    def instantiate_optimization_procedure(self):
        """Instantilize optimization procedure via Optimizable functionality."""
        self.batch_size = self.get_data_buffer_batch_size()
        (
            self.running_objective_var,
            self.observation_var,
            self.episode_id_var,
            self.time_var,
            self.critic_weights_var,
        ) = (
            self.create_variable(
                name="running_objective",
                is_constant=True,
            ),
            self.create_variable(
                name="observation",
                is_constant=True,
            ),
            self.create_variable(
                name="episode_id",
                is_constant=True,
            ),
            self.create_variable(
                name="time",
                is_constant=True,
            ),
            self.create_variable(
                name="critic_weights",
                like=self.model.named_parameters,
                is_constant=False,
            ),
        )

        self.register_objective(
            func=self.objective_function,
            variables=[
                self.running_objective_var,
                self.observation_var,
                self.episode_id_var,
                self.time_var,
            ],
        )

    def data_buffer_objective_keys(self):
        return ["observation", "running_objective", "time", "episode_id"]

    def objective_function(
        self,
        running_objective: torch.Tensor,
        observation: torch.Tensor,
        episode_id: torch.Tensor,
        time: torch.Tensor,
    ):
        # return temporal_difference_objective_full(
        #     self.model(observation),
        #     running_objective,
        #     self.td_n,
        #     discount_factor=self.discount_factor,
        #     sampling_time=self.sampling_time,
        #     episode_ids=episode_id,
        # )

        episode_ids = episode_id.unique()
        total_batch_size = len(running_objective)

        objective = 0
        for ep_id in episode_ids:
            mask = (ep_id == episode_id).reshape(-1)
            episodic_running_objectives = running_objective[mask]
            episodic_observations = observation[mask]
            episodic_times = time[mask]

            critic_model_outputs = self.model(episodic_observations)

            # critic_targets_old = calculate_critic_targets_old(
            #     running_objective=episodic_running_objectives,
            #     critic_model_output=critic_model_outputs,
            #     td_n=self.td_n,
            #     discount_factor=self.discount_factor,
            #     sampling_time=self.sampling_time,
            # )

            critic_targets = calculate_critic_targets(
                running_objectives=episodic_running_objectives,
                critic_model_outputs=critic_model_outputs,
                times=episodic_times,
                td_n=self.td_n,
                discount_factor=self.discount_factor,
                sampling_time=self.sampling_time,
            )

            # objective += (
            # torch.nn.functional.mse_loss(
            #     critic_model_outputs[: -self.td_n],
            #     critic_targets_old,
            #     reduction="sum",
            # )
            #     / total_batch_size
            # )
            objective += (
                torch.nn.functional.mse_loss(
                    critic_model_outputs,
                    critic_targets,
                    reduction="sum",
                )
                / total_batch_size
            )

            # assert torch.allclose(critic_targets_old, critic_targets[: -self.td_n])
            # assert torch.allclose(
            #     torch.nn.functional.mse_loss(
            #         critic_model_outputs[: -self.td_n],
            #         critic_targets[: -self.td_n],
            #         reduction="sum",
            #     ),
            #     torch.nn.functional.mse_loss(
            #         critic_model_outputs[: -self.td_n],
            #         critic_targets_old,
            #         reduction="sum",
            #     ),
            # )

        return objective
