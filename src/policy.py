from numpy.core.multiarray import array as array
from regelum.policy import Policy
import numpy as np
from scipy.special import expit
from src.system import (
    InvertedPendulum,
    InvertedPendulumWithFriction,
    InvertedPendulumWithMotor,
)
from typing import Union


def soft_switch(signal1, signal2, gate, loc=np.cos(np.pi / 4), scale=10):

    # Soft switch coefficient
    switch_coeff = expit((gate - loc) * scale)

    return (1 - switch_coeff) * signal1 + switch_coeff * signal2


def hard_switch(signal1: float, signal2: float, condition: bool):
    if condition:
        return signal1
    else:
        return signal2


def pd_based_on_sin(observation, pd_coeffs=[20, 10]):
    return -pd_coeffs[0] * np.sin(observation[0, 0]) - pd_coeffs[1] * observation[0, 1]


class InvPendulumPolicyPD(Policy):
    def __init__(self, pd_coeffs: np.ndarray, action_min: float, action_max: float):
        super().__init__()

        self.pid_coeffs = np.array(pd_coeffs).reshape(1, -1)
        self.action_min = action_min
        self.action_max = action_max

    def get_action(self, observation: np.ndarray):
        action = np.clip(
            (self.pid_coeffs * observation).sum(),
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
        pd_coeffs: np.ndarray,
        system: Union[InvertedPendulum, InvertedPendulumWithFriction],
    ):
        super().__init__()
        self.gain = gain
        self.action_min = action_min
        self.action_max = action_max
        self.switch_loc = switch_loc
        self.pd_coeffs = pd_coeffs
        self.system = system

    def get_action(self, observation: np.ndarray) -> np.ndarray:

        params = self.system._parameters
        mass, grav_const, length = (
            params["mass"],
            params["grav_const"],
            params["length"],
        )

        angle = observation[0, 0]
        angle_vel = observation[0, 1]

        energy_total = (
            mass * grav_const * length * (np.cos(angle) - 1) / 2
            + 0.5 * self.system.pendulum_moment_inertia() * angle_vel**2
        )
        energy_control_action = -self.gain * np.sign(angle_vel * energy_total)

        action = hard_switch(
            signal1=energy_control_action,
            signal2=-self.pd_coeffs[0] * np.sin(angle) - self.pd_coeffs[1] * angle_vel,
            condition=np.cos(angle) <= self.switch_loc,
        )

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
        pd_coeffs: np.ndarray,
        system: InvertedPendulumWithFriction,
    ):
        super().__init__()
        self.gain = gain
        self.action_min = action_min
        self.action_max = action_max
        self.switch_loc = switch_loc
        self.pd_coeffs = pd_coeffs
        self.system = system

    def get_action(self, observation: np.ndarray) -> np.ndarray:

        params = self.system._parameters
        mass, grav_const, length, friction_coeff = (
            params["mass"],
            params["grav_const"],
            params["length"],
            params["friction_coeff"],
        )

        angle = observation[0, 0]
        angle_vel = observation[0, 1]
        energy_total = (
            mass * grav_const * length * (np.cos(angle) - 1) / 2
            + 0.5 * self.system.pendulum_moment_inertia() * angle_vel**2
        )
        energy_control_action = -self.gain * np.sign(
            angle_vel * energy_total
        ) + friction_coeff * self.system.pendulum_moment_inertia() * angle_vel * np.abs(
            angle_vel
        )

        action = hard_switch(
            signal1=energy_control_action,
            signal2=-self.pd_coeffs[0] * np.sin(angle) - self.pd_coeffs[1] * angle_vel,
            condition=np.cos(angle) <= self.switch_loc,
        )

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
        pd_coeffs: list,
        system: InvertedPendulumWithFriction,
        friction_coeff_est_init: float = 0,
    ):
        super().__init__()
        self.gain = gain
        self.action_min = action_min
        self.action_max = action_max
        self.friction_coeff_est = friction_coeff_est_init
        self.sampling_time = sampling_time
        self.gain_adaptive = gain_adaptive
        self.switch_loc = switch_loc
        self.pd_coeffs = pd_coeffs
        self.system = system

    def get_action(self, observation: np.ndarray) -> np.ndarray:

        params = self.system._parameters
        mass, grav_const, length = (
            params["mass"],
            params["grav_const"],
            params["length"],
        )

        angle = observation[0, 0]
        angle_vel = observation[0, 1]

        energy_total = (
            mass * grav_const * length * (np.cos(angle) - 1) / 2
            + 0.5 * self.system.pendulum_moment_inertia() * angle_vel**2
        )
        energy_control_action = -self.gain * np.sign(
            angle_vel * energy_total
        ) + self.friction_coeff_est * self.system.pendulum_moment_inertia() * angle_vel * np.abs(
            angle_vel
        )

        # Parameter adaptation using Euler scheme
        self.friction_coeff_est += (
            -self.gain_adaptive
            * energy_total
            * mass
            * length**2
            * np.abs(angle_vel) ** 3
            * self.sampling_time
        )

        action = hard_switch(
            signal1=energy_control_action,
            signal2=-self.pd_coeffs[0] * np.sin(angle) - self.pd_coeffs[1] * angle_vel,
            condition=np.cos(angle) <= self.switch_loc,
        )
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

    def __init__(
        self,
        energy_gain: float,
        backstepping_gain: float,
        switch_loc: float,
        pd_coeffs: list[float],
        action_min: float,
        action_max: float,
        system: InvertedPendulumWithMotor,
    ):

        super().__init__()

        self.action_min = action_min
        self.action_max = action_max
        self.switch_loc = switch_loc
        self.energy_gain = energy_gain
        self.backstepping_gain = backstepping_gain
        self.pd_coeffs = pd_coeffs
        self.system = system

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        params = self.system._parameters

        mass, grav_const, length = (
            params["mass"],
            params["grav_const"],
            params["length"],
        )

        angle = observation[0, 0]
        angle_vel = observation[0, 1]
        torque = observation[0, 2]

        energy_total = (
            mass * grav_const * length * (np.cos(angle) - 1) / 2
            + 0.5 * self.system.pendulum_moment() * angle_vel**2
        )
        energy_control_action = -self.energy_gain * np.sign(angle_vel * energy_total)
        backstepping_action = torque - self.backstepping_gain * (
            torque - energy_control_action
        )
        action_pd = -self.pd_coeffs[0] * np.sin(angle) - self.pd_coeffs[1] * angle_vel

        action = hard_switch(
            signal1=backstepping_action,
            signal2=action_pd,
            condition=(np.cos(angle) - 1) ** 2 + angle_vel**2 >= self.switch_loc,
        )

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

    def __init__(self, pd_coeffs: list, action_min: float, action_max: float):

        super().__init__()

        self.action_min = action_min
        self.action_max = action_max

        self.pd_coeffs = pd_coeffs

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        angle = observation[0, 0]
        angle_vel = observation[0, 1]

        action = -self.pd_coeffs[0] * angle - self.pd_coeffs[1] * angle_vel
        return np.array([[np.clip(action, self.action_min, self.action_max)]])
