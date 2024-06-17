from regelum.critic import Critic
from regelum.utils import rg
from regelum.system import System, ComposedSystem
from regelum.model import ModelNN, PerceptronWithTruncatedNormalNoise
from regelum.optimizable.core.configs import OptimizerConfig
from regelum.optimizable import Optimizable
from typing import Union, Optional
import torch


class CriticPPO(Critic):
    def __init__(
        self,
        system: Union[System, ComposedSystem],
        model: Union[ModelNN],
        policy_model: PerceptronWithTruncatedNormalNoise,
        device: str = "cpu",
        optimizer_config: Optional[OptimizerConfig] = None,
        discount_factor: float = 1.0,
        sampling_time: float = 0.01,
        entropy_coef: float = 0.02,
        td_n: int = 1,
    ):
        Optimizable.__init__(self, optimizer_config=optimizer_config)
        self.system = system
        self.model = model
        self.device = device
        self.discount_factor = discount_factor
        self.sampling_time = sampling_time
        self.policy_model = policy_model
        self.entropy_coef = entropy_coef
        self.td_n = td_n
        self.instantiate_optimization_procedure()

    def instantiate_optimization_procedure(self):
        """Instantilize optimization procedure via Optimizable functionality."""
        self.batch_size = self.get_data_buffer_batch_size()
        (
            self.running_objective_var,
            self.observation_var,
            self.action_var,
            self.episode_id_var,
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
                name="action",
                is_constant=True,
            ),
            self.create_variable(
                name="episode_id",
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
                self.action_var,
                self.episode_id_var,
            ],
        )

    def data_buffer_objective_keys(self):
        return ["observation", "action", "running_objective", "episode_id"]

    def objective_function(
        self,
        running_objective: torch.Tensor,
        observation: torch.Tensor,
        action: torch.Tensor,
        episode_id: torch.Tensor,
    ):
        pass
