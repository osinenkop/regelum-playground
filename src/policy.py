from regelum.policy import Policy
import numpy as np
from regelum.system import InvertedPendulum
from scipy.special import expit


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
    def __init__(
        self,
        gain: float,
        pd_coefs: np.ndarray,
        softswitch_loc,
        softswitch_scale,
        action_min: float,
        action_max: float,
    ):
        super().__init__()

        self.gain = gain
        self.pd_coefs = pd_coefs
        self.softswitch_loc = softswitch_loc
        self.softswitch_scale = softswitch_scale
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
        # energy_control_action = self.gain * energy_total**2 * np.sign(theta_vel)
        energy_control_action = self.gain * np.sign(theta_vel)

        pd_control_action = np.clip(
            (self.pd_coefs * observation).sum(),
            self.action_min,
            self.action_max,
        )

        alpha = expit((np.cos(theta) - self.softswitch_loc) * self.softswitch_scale)

        return np.array(
            [
                [
                    np.clip(
                        (1 - alpha) * energy_control_action + alpha * pd_control_action,
                        self.action_min,
                        self.action_max,
                    )
                ]
            ]
        )
