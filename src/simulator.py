from regelum.simulator import CasADi
from regelum.system import System, ComposedSystem
from typing import Union, Optional, Callable
import numpy as np

from regelum.scenario import RLScenario


def generate_state_init_for_pendulum():
    return np.array(
        [
            [
                np.random.uniform(low=np.pi - np.pi / 2, high=np.pi + np.pi / 2),
                np.random.uniform(-8, 8),
            ]
        ]
    )


class StateInitRandomSamplerSimulator(CasADi):
    def __init__(
        self,
        system: Union[System, ComposedSystem],
        state_init: Callable[[], np.ndarray] = generate_state_init_for_pendulum,
        action_init: Optional[np.ndarray] = None,
        time_final: Optional[float] = 1,
        max_step: Optional[float] = 1e-3,
        first_step: Optional[float] = 1e-6,
        atol: Optional[float] = 1e-5,
        rtol: Optional[float] = 1e-3,
    ):
        self.state_init_callable = state_init
        self.state_init = self.state_init_callable()
        super().__init__(
            system=system,
            state_init=self.state_init,
            time_final=time_final,
            action_init=action_init,
            max_step=max_step,
            first_step=first_step,
            atol=atol,
            rtol=rtol,
        )

    def reset(self):
        self.state_init = self.state_init_callable()
        if self.system.system_type == "diff_eqn":
            self.ODE_solver = self.initialize_ode_solver()
            self.time = 0.0
            self.state = self.state_init
            self.observation = self.get_observation(
                time=self.time, state=self.state_init, inputs=self.action_init
            )
        else:
            self.time = 0.0
            self.observation = self.get_observation(
                time=self.time, state=self.state_init, inputs=self.system.inputs
            )
