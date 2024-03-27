from regelum.policy import Policy
import numpy as np
from regelum.system import InvertedPendulum


class InvPendulumPolicyPD(Policy):
    def __init__(self, pd_coefs: np.ndarray, action_min: float, action_max: float):
        super().__init__()

        self.pid_coefs = np.array(pd_coefs).reshape(1, -1)
        self.action_min = action_min
        self.action_max = action_max

    def get_action(self, observation: np.ndarray):
        action = np.clip(
            (self.pid_coefs * observation).sum(),
            self.action_min,
            self.action_max,
        )
        return np.array([[action]])


class InvPendulumPolicyEnergyBased(Policy):
    def __init__(self, gain: float, action_min: float, action_max: float):
        super().__init__()

        self.gain = gain
        self.action_min = action_min
        self.action_max = action_max

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        params = InvertedPendulum._parameters
        m, g, length = params["m"], params["g"], params["l"]
        theta = observation[0, 0]
        theta_vel = observation[0, 1]

        energy_total = (
            m * g * length * (1 - np.cos(theta)) + 0.5 * m * length**2 * theta_vel**2
        )
        control_action = (
            self.gain * energy_total**2 * np.sign(theta_vel) * (theta / 100)
        )

        return np.array(
            [
                [
                    np.clip(
                        control_action,
                        self.action_min,
                        self.action_max,
                    )
                ]
            ]
        )
