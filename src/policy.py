from numpy.core.multiarray import array as array
from regelum.model import Model, ModelNN
from regelum.optimizable.optimizers import OptimizerConfig
from regelum.policy import Policy
import numpy as np
from regelum.system import ComposedSystem, System
from scipy.special import expit
from src.system import (
    InvertedPendulum,
    InvertedPendulumWithFriction,
    InvertedPendulumWithMotor,
)


def soft_switch(signal1, signal2, gate, loc=np.cos(np.pi / 4), scale=10):

    # Soft switch coefficient
    switch_coeff = expit((gate - loc) * scale)

    return (1 - switch_coeff) * signal1 + switch_coeff * signal2


def pd_based_on_sin(observation, pd_coefs=[20, 10]):

    return -pd_coefs[0] * np.sin(observation[0, 0]) - pd_coefs[1] * observation[0, 1]


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


class InvertedPendulumEnergyBased(Policy):
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
            m * g * length * (np.cos(theta) - 1) + 0.5 * m * length**2 * theta_vel**2
        )
        energy_control_action = -self.gain * np.sign(theta_vel * energy_total)

        if np.cos(theta) <= np.cos(np.pi / 10):
            action = energy_control_action
        else:
            action = -0.2 * theta - 0.1 * theta_vel

        return np.array(
            [
                [
                    np.clip(
                        action,
                        # soft_switch(
                        #     signal1=energy_control_action,
                        #     signal2=pd_based_on_sin(observation, pd_coefs=[0.2, 0.1]),
                        #     gate=np.cos(theta),
                        # ),
                        # energy_control_action,
                        self.action_min,
                        self.action_max,
                    )
                ]
            ]
        )


class InvPendulumEnergyBasedFrictionCompensation(Policy):

    def __init__(self, gain: float, action_min: float, action_max: float):
        super().__init__()
        self.gain = gain
        self.action_min = action_min
        self.action_max = action_max

    def get_action(self, observation: np.ndarray) -> np.ndarray:

        params = InvertedPendulumWithFriction._parameters
        m, g, length, friction_coef = params["m"], params["g"], params["l"], params["c"]

        theta = observation[0, 0]
        theta_vel = observation[0, 1]

        energy_total = (
            m * g * length * (np.cos(theta) - 1) + 0.5 * m * length**2 * theta_vel**2
        )
        energy_control_action = -self.gain * np.sign(
            theta_vel * energy_total
        ) + friction_coef * m * length**2 * theta_vel * np.abs(theta_vel)

        if np.cos(theta) <= np.cos(np.pi / 10):
            action = energy_control_action
        else:
            action = -0.2 * theta - 0.1 * theta_vel

        return np.array(
            [
                [
                    np.clip(
                        action,
                        self.action_min,
                        self.action_max,
                    )
                ]
            ]
        )


class InvPendulumEnergyBasedFrictionAdaptive(Policy):

    def __init__(
        self,
        gain: float,
        action_min: float,
        action_max: float,
        sampling_time: float,
        gain_adaptive: float,
        friction_coef_est_init: float = 0,
    ):
        super().__init__()
        self.gain = gain
        self.action_min = action_min
        self.action_max = action_max
        self.friction_coef_est = friction_coef_est_init
        self.sampling_time = sampling_time
        self.gain_adaptive = gain_adaptive

    def get_action(self, observation: np.ndarray) -> np.ndarray:

        params = InvertedPendulumWithFriction._parameters
        m, g, length = params["m"], params["g"], params["l"]

        theta = observation[0, 0]
        theta_vel = observation[0, 1]

        energy_total = (
            m * g * length * (np.cos(theta) - 1) + 0.5 * m * length**2 * theta_vel**2
        )
        energy_control_action = -self.gain * np.sign(
            theta_vel * energy_total
        ) + self.friction_coef_est * m * length**2 * theta_vel * np.abs(theta_vel)

        # Parameter adaptation using Euler scheme
        self.friction_coef_est += (
            -self.gain_adaptive
            * energy_total
            * m
            * length**2
            * np.abs(theta_vel) ** 3
            * self.sampling_time
        )

        # print("Friction coefficient estimate: ", round(self.friction_coef_est, 2))

        return np.array(
            [
                [
                    np.clip(
                        soft_switch(
                            signal1=energy_control_action,
                            signal2=pd_based_on_sin(observation),
                            gate=np.cos(theta),
                        ),
                        self.action_min,
                        self.action_max,
                    )
                ]
            ]
        )


class InvertedPendulumBackstepping(Policy):

    def __init__(self, action_min, action_max):

        super().__init__()

        self.action_min = action_min
        self.action_max = action_max

        self.energy_gain = 0.01
        self.gain = 40
        self.pd_coefs = [150, 120, 12]

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        params = InvertedPendulumWithMotor._parameters

        m, g, length = params["m"], params["g"], params["l"]

        theta = observation[0, 0]
        theta_vel = observation[0, 1]
        torque = observation[0, 2]

        energy_total = (
            m * g * length * (np.cos(theta) - 1) + 0.5 * m * length**2 * theta_vel**2
        )
        energy_control_action = -self.energy_gain * np.sign(theta_vel * energy_total)

        # if np.cos(theta) <= np.cos(np.pi / 8):
        #     action = -self.gain * (torque - energy_control_action)
        # else:
        #     action = (
        #         -theta * self.pd_coefs[0]
        #         - theta_vel * self.pd_coefs[1]
        #         - torque * self.pd_coefs[2]
        #     )
        action = -self.gain * (torque - energy_control_action) + torque
        return np.array([[np.clip(action, self.action_min, self.action_max)]])


class InvertedPendulumWithMotorPD(Policy):

    def __init__(self, action_min, action_max):

        super().__init__()

        self.action_min = action_min
        self.action_max = action_max

        self.pd_coefs = [150, 120, 12]

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        params = InvertedPendulumWithMotor._parameters

        m, g, length = params["m"], params["g"], params["l"]

        theta = observation[0, 0]
        theta_vel = observation[0, 1]
        torque = observation[0, 2]

        action = (
            -theta * self.pd_coefs[0]
            - theta_vel * self.pd_coefs[1]
            - torque * self.pd_coefs[2]
        )
        return np.array([[np.clip(action, self.action_min, self.action_max)]])
