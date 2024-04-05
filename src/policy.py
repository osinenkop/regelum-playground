from numpy.core.multiarray import array as array
from regelum.policy import Policy
import numpy as np
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
    def __init__(
        self,
        gain: float,
        action_min: float,
        action_max: float,
        switch_loc: float,
        pd_coefs: np.ndarray,
    ):
        super().__init__()
        self.gain = gain
        self.action_min = action_min
        self.action_max = action_max
        self.switch_loc = switch_loc
        self.pd_coefs = pd_coefs

    def get_action(self, observation: np.ndarray) -> np.ndarray:

        params = InvertedPendulum._parameters
        m, g, length = params["m"], params["g"], params["l"]

        theta = observation[0, 0]
        theta_vel = observation[0, 1]

        energy_total = (
            m * g * length * (np.cos(theta) - 1) / 2
            + 0.5 * InvertedPendulum.pendulum_moment(m, length) * theta_vel**2
        )
        energy_control_action = -self.gain * np.sign(theta_vel * energy_total)

        if np.cos(theta) <= self.switch_loc:
            action = energy_control_action
        else:
            action = -self.pd_coefs[0] * np.sin(theta) - self.pd_coefs[1] * theta_vel

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


class InvPendulumEnergyBasedFrictionCompensation(Policy):

    def __init__(
        self,
        gain: float,
        action_min: float,
        action_max: float,
        switch_loc: float,
        pd_coefs: np.ndarray,
    ):
        super().__init__()
        self.gain = gain
        self.action_min = action_min
        self.action_max = action_max
        self.switch_loc = switch_loc
        self.pd_coefs = pd_coefs

    def get_action(self, observation: np.ndarray) -> np.ndarray:

        params = InvertedPendulumWithFriction._parameters
        m, g, length, friction_coef = params["m"], params["g"], params["l"], params["c"]

        theta = observation[0, 0]
        theta_vel = observation[0, 1]
        energy_total = (
            m * g * length * (np.cos(theta) - 1) / 2
            + 0.5
            * InvertedPendulumWithFriction.pendulum_moment(m, length)
            * theta_vel**2
        )
        energy_control_action = -self.gain * np.sign(
            theta_vel * energy_total
        ) + friction_coef * InvertedPendulumWithFriction.pendulum_moment(
            m, length
        ) * theta_vel * np.abs(
            theta_vel
        )

        if np.cos(theta) <= self.switch_loc:
            action = energy_control_action
        else:
            action = -self.pd_coefs[0] * np.sin(theta) - self.pd_coefs[1] * theta_vel

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
        switch_loc: float,
        pd_coefs: list,
        friction_coef_est_init: float = 0,
    ):
        super().__init__()
        self.gain = gain
        self.action_min = action_min
        self.action_max = action_max
        self.friction_coef_est = friction_coef_est_init
        self.sampling_time = sampling_time
        self.gain_adaptive = gain_adaptive
        self.switch_loc = switch_loc
        self.pd_coefs = pd_coefs

    def get_action(self, observation: np.ndarray) -> np.ndarray:

        params = InvertedPendulumWithFriction._parameters
        m, g, length = params["m"], params["g"], params["l"]

        theta = observation[0, 0]
        theta_vel = observation[0, 1]

        energy_total = (
            m * g * length * (np.cos(theta) - 1) / 2
            + 0.5
            * InvertedPendulumWithFriction.pendulum_moment(m, length)
            * theta_vel**2
        )
        energy_control_action = -self.gain * np.sign(
            theta_vel * energy_total
        ) + self.friction_coef_est * InvertedPendulumWithFriction.pendulum_moment(
            m, length
        ) * theta_vel * np.abs(
            theta_vel
        )

        # Parameter adaptation using Euler scheme
        self.friction_coef_est += (
            -self.gain_adaptive
            * energy_total
            * m
            * length**2
            * np.abs(theta_vel) ** 3
            * self.sampling_time
        )

        if np.cos(theta) <= self.switch_loc:
            action = energy_control_action
        else:
            action = -self.pd_coefs[0] * np.sin(theta) - self.pd_coefs[1] * theta_vel

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


class InvertedPendulumBackstepping(Policy):

    def __init__(self, energy_gain, gain, switch_loc, pd_coefs, action_min, action_max):

        super().__init__()

        self.action_min = action_min
        self.action_max = action_max
        self.switch_loc = switch_loc
        self.energy_gain = energy_gain
        self.gain = gain
        self.pd_coefs = pd_coefs

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        params = InvertedPendulumWithMotor._parameters

        m, g, length = params["m"], params["g"], params["l"]

        theta = observation[0, 0]
        theta_vel = observation[0, 1]
        torque = observation[0, 2]

        energy_total = (
            m * g * length * (np.cos(theta) - 1) / 2
            + 0.5 * InvertedPendulumWithMotor.pendulum_moment(m, length) * theta_vel**2
        )
        energy_control_action = -self.energy_gain * np.sign(theta_vel * energy_total)
        backstepping_action = torque - self.gain * (torque - energy_control_action)

        coef = expit((np.cos(theta) - self.switch_loc) * 5)
        action_pd = -np.sin(theta) * self.pd_coefs[0] - theta_vel * self.pd_coefs[1]

        action = (1 - coef) * backstepping_action + coef * action_pd
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


class InvertedPendulumWithMotorPD(Policy):

    def __init__(self, pd_coefs, action_min, action_max):

        super().__init__()

        self.action_min = action_min
        self.action_max = action_max

        self.pd_coefs = pd_coefs

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        theta = observation[0, 0]
        theta_vel = observation[0, 1]
        torque = observation[0, 2]

        action = -theta * self.pd_coefs[0] - theta_vel * self.pd_coefs[1]
        return np.array([[np.clip(action, self.action_min, self.action_max)]])
