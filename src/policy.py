"""
Contains policy classes.

"""

import numpy as np
import scipy as sp

from typing import Union
from numpy.core.multiarray import array

from numpy.matlib import repmat
from numpy.linalg import norm

from scipy.optimize import minimize

from regelum.policy import Policy
from regelum.utils import rg
from regelum import CasadiOptimizerConfig

from scipy.special import expit
from src.system import (
    InvertedPendulum,
    InvertedPendulumWithFriction,
    InvertedPendulumWithMotor,
)

from .utilities import uptria2vec
from .utilities import to_row_vec
from .utilities import to_scalar
from .utilities import push_vec

# Vectors are assumed 2-dimensional rows by default.
# Some functions, like running objective or critic model, are robust to input argument dimensions and force row format internally.
# Data buffers are assumed matrices stacked off of vectors row-wise.


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
            + 1 / 2 * self.system.pendulum_moment_inertia() * angle_vel**2
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
            + 1 / 2 * self.system.pendulum_moment_inertia() * angle_vel**2
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
            + 1 / 2 * self.system.pendulum_moment_inertia() * angle_vel**2
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
            + 0.5 * self.system.pendulum_moment_inertia() * angle_vel**2
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


class ThreeWheeledRobotKinematicMinGradCLF(Policy):

    def __init__(
        self,
        optimizer_config: CasadiOptimizerConfig,
        action_bounds: list[list[float]],
        eps=0.01,
    ):
        super().__init__(optimizer_config=optimizer_config)
        self.action_bounds = action_bounds
        # An epsilon for numerical stability
        self.eps = eps
        self.instantiate_optimization_procedure()

    def derivative_of_three_wheeled_robot_kin_lyapunov_function(
        self, x_coord, y_coord, angle, vel, angle_vel
    ):
        x_derivative = vel * rg.cos(angle)
        y_derivative = vel * rg.sin(angle)

        return (
            x_coord * x_derivative
            + y_coord * y_derivative
            + (angle - np.arctan(y_coord / (rg.sign(x_coord) * self.eps + x_coord)))
            * (
                angle_vel
                - (y_derivative * x_coord - x_derivative * y_coord)
                / (x_coord**2 + y_coord**2)
            )
        )

    def instantiate_optimization_procedure(self):
        self.x_coord_var = self.create_variable(1, name="x_coord", is_constant=True)
        self.y_coord_var = self.create_variable(1, name="y_coord", is_constant=True)
        self.angle_var = self.create_variable(1, name="angle", is_constant=True)
        self.vel_var = self.create_variable(
            1, name="vel", is_constant=False, like=np.array([[0]])
        )
        self.angle_vel_var = self.create_variable(
            1, name="angle_vel", is_constant=False, like=np.array([[0]])
        )
        self.register_bounds(self.vel_var, np.array(self.action_bounds[None, 0]))
        self.register_bounds(self.angle_vel_var, np.array(self.action_bounds[None, 1]))

        self.register_objective(
            self.derivative_of_three_wheeled_robot_kin_lyapunov_function,
            variables=[
                self.x_coord_var,
                self.y_coord_var,
                self.angle_var,
                self.vel_var,
                self.angle_vel_var,
            ],
        )

    def get_action(self, observation: np.ndarray):
        x_coord = observation[0, 0]
        y_coord = observation[0, 1]
        angle = observation[0, 2]

        optimized_vel_and_angle_vel = self.optimize(
            x_coord=x_coord, y_coord=y_coord, angle=angle
        )

        # The result of optimization is a dict of casadi tensors, so we convert them to float
        angle_vel = float(optimized_vel_and_angle_vel["angle_vel"][0, 0])
        vel = float(optimized_vel_and_angle_vel["vel"][0, 0])

        return np.array([[vel, angle_vel]])


class ThreeWheeledRobotDynamicMinGradCLF(ThreeWheeledRobotKinematicMinGradCLF):

    def __init__(
        self,
        optimizer_config: CasadiOptimizerConfig,
        action_bounds: list[list[float]],
        gain: float,
        eps: float = 0.01,
    ):
        super().__init__(
            optimizer_config=optimizer_config, eps=eps, action_bounds=action_bounds
        )
        self.gain = gain

    def get_action(self, observation: np.ndarray):
        three_wheeled_robot_kin_action = super().get_action(observation)
        force_and_moment = np.array([[observation[0, 3], observation[0, 4]]])
        action = -self.gain * (force_and_moment - three_wheeled_robot_kin_action)

        return action


class InvertedPendulumRcognitaCALFQ(Policy):
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
        # Initialization of RL agent
        # 1. Common agent tuning settings
        self.run_obj_param_tensor = np.diag([1.0, 1.0, 0.0])
        self.episode_total_time = 5.0

        # 2. Actor
        self.action_change_penalty_coeff = 0.0

        # 3. Critic
        self.critic_learn_rate = 0.1
        self.critic_num_grad_steps = 20
        self.discount_factor = 1.0
        self.buffer_size = 20
        self.critic_struct = "quad-mix"
        self.critic_weight_change_penalty_coeff = 0.0

        # 4. CALFQ
        # Probability to take CALF action even when CALF constraints are not satisfied
        self.relax_probability = 0.75
        self.relax_probability_fading_factor = 0.0
        self.critic_low_kappa_coeff = 1e-2
        self.critic_up_kappa_coeff = 1e4
        # Nominal desired step-wise decay coeff of critic
        self.critic_desired_decay_coeff = 1e-4
        # Maximal desired step-wise decay coeff of critic
        self.critic_max_desired_decay_coeff = 1e-1
        self.calf_penalty_coeff = 0.5

        # Normally, all the below settings are not for agent tuning
        self.gain = gain
        self.action_min = action_min
        self.action_max = action_max
        self.switch_loc = switch_loc
        self.pd_coeffs = pd_coeffs
        self.system = system

        self.dim_state = 2
        self.dim_action = 1
        self.dim_observation = self.dim_state
        # Taken from initial_conditions config
        self.state_init = np.array([[np.pi, 1]])
        self.observation_init = self.state_init
        self.score = 0
        self.clock = 1
        self.reset_clock = 1
        self.episode_count = 0

        self.action_sampling_period = 0.01  # Taken from common/inv_pendulum config

        self.in_state_of_reset = False
        self.episode_num_steps = int(
            self.episode_total_time / self.action_sampling_period
        )
        self.reset_total_time = 4.0
        self.reset_num_steps = int(self.reset_total_time / self.action_sampling_period)

        self.action_init = self.get_safe_action(self.state_init)
        self.action_curr = self.action_init

        self.action_buffer = repmat(self.action_init, self.buffer_size, 1)
        self.observation_buffer = repmat(self.observation_init, self.buffer_size, 1)

        critic_big_number = 1e5

        if self.critic_struct == "quad-lin":
            self.dim_critic = int(
                ((self.dim_observation + self.dim_action) + 1)
                * (self.dim_observation + self.dim_action)
                / 2
                + (self.dim_observation + self.dim_action)
            )
            self.critic_weight_min = -critic_big_number
            self.critic_weight_max = critic_big_number
        elif self.critic_struct == "quadratic":
            self.dim_critic = int(
                ((self.dim_observation + self.dim_action) + 1)
                * (self.dim_observation + self.dim_action)
                / 2
            )
            self.critic_weight_min = 0
            self.critic_weight_max = critic_big_number
        elif self.critic_struct == "quad-nomix":
            self.dim_critic = self.dim_observation + self.dim_action
            self.critic_weight_min = 0
            self.critic_weight_max = critic_big_number
        elif self.critic_struct == "quad-mix":
            self.dim_critic = int(
                self.dim_observation
                + self.dim_observation * self.dim_action
                + self.dim_action
            )
            self.critic_weight_min = -critic_big_number
            self.critic_weight_max = critic_big_number

        self.critic_weight_tensor_init = to_row_vec(
            np.random.uniform(1, critic_big_number / 10, size=self.dim_critic)
        )
        self.critic_weight_tensor = self.critic_weight_tensor_init

        self.critic_weight_tensor_safe = self.critic_weight_tensor_init
        self.observation_safe = self.observation_init
        self.action_safe = self.action_init

        # self.critic_weight_tensor_init_safe = self.critic_weight_tensor_init
        # self.critic_weight_tensor_buffer_safe = []

        self.action_buffer_safe = np.zeros([self.buffer_size, self.dim_action])
        self.observation_buffer_safe = np.zeros(
            [self.buffer_size, self.dim_observation]
        )

        self.critic_desired_decay = (
            self.critic_desired_decay_coeff * self.action_sampling_period
        )
        self.critic_max_desired_decay = (
            self.critic_max_desired_decay_coeff * self.action_sampling_period
        )

        self.calf_count = 0
        self.safe_count = 0

        self.relax_probability_init = self.relax_probability

        # Debugging
        self.debug_print_counter = 0

    def run_obj(self, observation, action):

        # DEBUG
        # Override observation for pendulum

        modified_observation = to_row_vec(observation)
        angle = modified_observation[0, 0]
        angle_velocity = modified_observation[0, 1]

        modified_observation = np.array([1 - np.cos(angle), angle_velocity])

        # /DEBUG

        observation_action = np.hstack(
            [to_row_vec(modified_observation), to_row_vec(action)]
        )

        result = observation_action @ self.run_obj_param_tensor @ observation_action.T

        return to_scalar(result)

    def critic_model(self, critic_weight_tensor, observation, action):

        observation_action = np.hstack([to_row_vec(observation), to_row_vec(action)])

        if self.critic_struct == "quad-lin":
            feature_tensor = np.hstack(
                [
                    uptria2vec(
                        np.outer(observation_action, observation_action),
                        force_row_vec=True,
                    ),
                    observation_action,
                ]
            )
        elif self.critic_struct == "quadratic":
            feature_tensor = uptria2vec(
                np.outer(observation_action, observation_action), force_row_vec=True
            )
        elif self.critic_struct == "quad-nomix":
            feature_tensor = observation_action * observation_action
        elif self.critic_struct == "quad-mix":
            feature_tensor = np.hstack(
                [
                    to_row_vec(observation) ** 2,
                    np.kron(to_row_vec(observation), to_row_vec(action)),
                    to_row_vec(action) ** 2,
                ]
            )

        result = critic_weight_tensor @ feature_tensor.T

        return to_scalar(result)

    def critic_model_grad(self, critic_weight_tensor, observation, action):

        observation_action = np.hstack([to_row_vec(observation), to_row_vec(action)])

        if self.critic_struct == "quad-lin":
            feature_tensor = np.hstack(
                [
                    uptria2vec(
                        np.outer(observation_action, observation_action),
                        force_row_vec=True,
                    ),
                    observation_action,
                ]
            )
        elif self.critic_struct == "quadratic":
            feature_tensor = uptria2vec(
                np.outer(observation_action, observation_action), force_row_vec=True
            )
        elif self.critic_struct == "quad-nomix":
            feature_tensor = observation_action * observation_action
        elif self.critic_struct == "quad-mix":
            feature_tensor = np.hstack(
                [
                    to_row_vec(observation) ** 2,
                    np.kron(to_row_vec(observation), to_row_vec(action)),
                    to_row_vec(action) ** 2,
                ]
            )

        return feature_tensor

    def critic_obj(self, critic_weight_tensor_change):
        """
        Objective function for critic learning.

        Uses value iteration format where previous weights are assumed different from the ones being optimized.

        """
        critic_weight_tensor_pivot = self.critic_weight_tensor_safe
        critic_weight_tensor = (
            self.critic_weight_tensor_safe + critic_weight_tensor_change
        )

        result = 0

        for k in range(self.buffer_size - 1, 0, -1):
            # Python's array slicing may return 1D arrays, but we don't care here
            observation_prev = self.observation_buffer[k - 1, :]
            observation_next = self.observation_buffer[k, :]
            action_prev = self.action_buffer[k - 1, :]
            action_next = self.action_buffer[k, :]

            # observation_prev_safe = self.observation_buffer_safe[k-1, :]
            # observation_next_safe = self.observation_buffer_safe[k, :]
            # action_prev_safe = self.action_buffer_safe[k-1, :]
            # action_next_safe = self.action_buffer_safe[k, :]

            critic_prev = self.critic_model(
                critic_weight_tensor, observation_prev, action_prev
            )
            critic_next = self.critic_model(
                critic_weight_tensor_pivot, observation_next, action_next
            )

            temporal_error = (
                critic_prev
                - self.discount_factor * critic_next
                - self.run_obj(observation_prev, action_prev)
            )

            result += 1 / 2 * temporal_error**2

        result += (
            1
            / 2
            * self.critic_weight_change_penalty_coeff
            * norm(critic_weight_tensor_change) ** 2
        )

        return result

    def critic_obj_grad(self, critic_weight_tensor):
        """
        Gradient of the objective function for critic learning.

        Uses value iteration format where previous weights are assumed different from the ones being optimized.

        """
        critic_weight_tensor_pivot = self.critic_weight_tensor_safe
        critic_weight_tensor_change = critic_weight_tensor_pivot - critic_weight_tensor

        result = to_row_vec(np.zeros(self.dim_critic))

        for k in range(self.buffer_size - 1, 0, -1):

            observation_prev = self.observation_buffer[k - 1, :]
            observation_next = self.observation_buffer[k, :]
            action_prev = self.action_buffer[k - 1, :]
            action_next = self.action_buffer[k, :]

            # observation_prev_safe = self.observation_buffer_safe[k-1, :]
            # observation_next_safe = self.observation_buffer_safe[k, :]
            # action_prev_safe = self.action_buffer_safe[k-1, :]
            # action_next_safe = self.action_buffer_safe[k, :]

            critic_prev = self.critic_model(
                critic_weight_tensor, observation_prev, action_prev
            )
            critic_next = self.critic_model(
                critic_weight_tensor_pivot, observation_next, action_next
            )

            temporal_error = (
                critic_prev
                - self.discount_factor * critic_next
                - self.run_obj(observation_prev, action_prev)
            )

            result += temporal_error * self.critic_model_grad(
                critic_weight_tensor, observation_prev, action_prev
            )

        result += self.critic_weight_change_penalty_coeff * critic_weight_tensor_change

        return result

    def calf_diff(self, critic_weight_tensor, observation, action):
        # Q^w  (s_t, a_t)
        critic_new = self.critic_model(critic_weight_tensor, observation, action)
        # Q^w† (s†, a†)
        critic_safe = self.critic_model(
            self.critic_weight_tensor_safe, self.observation_safe, self.action_safe
        )
        # Q^w  (s_t, a_t) - Q^w† (s†, a†)
        return critic_new - critic_safe

    def calf_decay_constraint_penalty_grad(
        self, critic_weight_tensor, observation, action
    ):
        # This one is handy for explicit gradient-descent optimization.
        # We take a ReLU here

        critic_new = self.critic_model(critic_weight_tensor, observation, action)

        critic_safe = self.critic_model(
            self.critic_weight_tensor_safe, self.observation_safe, self.action_safe
        )

        if critic_new - critic_safe <= -self.critic_desired_decay:
            relu_grad = 0
        else:
            relu_grad = 1

        return (
            self.calf_penalty_coeff
            * self.critic_model_grad(critic_weight_tensor, observation, action)
            * relu_grad
        )

        # Quadratic penalty
        # return (
        #     self.calf_penalty_coeff
        #     * self.critic_model_grad(critic_weight_tensor, observation, action)
        #     * (critic_new - critic_safe + self.critic_desired_decay)
        # )

    def get_optimized_critic_weights(
        self,
        observation,
        action,
        use_grad_descent=True,
        use_calf_constraints=True,
        use_kappa_constraint=False,
        check_persistence_of_excitation=False,
    ):

        # def calf_kappa_constraint(observation, action, critic_weight_tensor):
        #     # Force critic to respect lower and upper kappa bounds

        #     # kappa_low(||s_t||) <= Q^w (s_t, a_t) <= kappa_up(||s_t||)
        #     return self.critic_model(critic_weight_tensor, observation, action)

        # Optimization method of critic. Methods that respect constraints: BFGS, L-BFGS-B, SLSQP,
        # trust-constr, Powell
        critic_opt_method = "SLSQP"
        if critic_opt_method == "trust-constr":
            # 'disp': True, 'verbose': 2}
            critic_opt_options = {"maxiter": 40, "disp": False}
        else:
            critic_opt_options = {
                "maxiter": 40,
                "maxfev": 80,
                "disp": False,
                "adaptive": True,
                "xatol": 1e-3,
                "fatol": 1e-3,
            }  # 'disp': True, 'verbose': 2}

        critic_low_kappa = self.critic_low_kappa_coeff * norm(observation) ** 2
        critic_up_kappa = self.critic_up_kappa_coeff * norm(observation) ** 2

        constraints = []
        constraints.append(
            sp.optimize.NonlinearConstraint(
                lambda critic_weight_tensor: self.calf_diff(
                    critic_weight_tensor=critic_weight_tensor,
                    observation=observation,
                    action=action,
                ),
                -self.critic_max_desired_decay,
                -self.critic_desired_decay,
            )
        )
        if use_kappa_constraint:

            constraints.append(
                sp.optimize.NonlinearConstraint(
                    lambda critic_weight_tensor: self.critic_model(
                        critic_weight_tensor=critic_weight_tensor,
                        observation=observation,
                        action=action,
                    ),
                    critic_low_kappa,
                    critic_up_kappa,
                )
            )

        bounds = sp.optimize.Bounds(
            self.critic_weight_min, self.critic_weight_max, keep_feasible=True
        )

        critic_weight_tensor_change_start_guess = to_row_vec(np.zeros(self.dim_critic))

        if use_calf_constraints:
            if use_grad_descent:

                critic_weight_tensor = self.critic_weight_tensor

                for _ in range(self.critic_num_grad_steps):

                    critic = self.critic_model(
                        critic_weight_tensor, observation, action
                    )

                    # Simple ReLU penalties for bounding kappas
                    if critic <= critic_up_kappa:
                        relu_kappa_up_grad = 0
                    else:
                        relu_kappa_up_grad = 1

                    if critic >= critic_low_kappa:
                        relu_kappa_low_grad = 0
                    else:
                        relu_kappa_low_grad = 1

                    critic_weight_tensor_change = -self.critic_learn_rate * (
                        self.critic_obj_grad(critic_weight_tensor)
                        + self.calf_decay_constraint_penalty_grad(
                            self.critic_weight_tensor, observation, action
                        )
                        + self.calf_penalty_coeff
                        * self.critic_model_grad(
                            critic_weight_tensor, observation, action
                        )
                        * relu_kappa_low_grad
                        + self.calf_penalty_coeff
                        * self.critic_model_grad(
                            critic_weight_tensor, observation, action
                        )
                        * relu_kappa_up_grad
                    )
                    critic_weight_tensor += critic_weight_tensor_change

            else:
                critic_weight_tensor_change = minimize(
                    self.critic_obj(critic_weight_tensor_change),
                    critic_weight_tensor_change_start_guess,
                    method=critic_opt_method,
                    tol=1e-3,
                    bounds=bounds,
                    constraints=constraints,
                    options=critic_opt_options,
                ).x
        else:
            if use_grad_descent:
                critic_weight_tensor_change = (
                    -self.critic_learn_rate
                    * self.critic_obj_grad(self.critic_weight_tensor)
                )
            else:
                critic_weight_tensor_change = minimize(
                    self.critic_obj(critic_weight_tensor_change),
                    critic_weight_tensor_change_start_guess,
                    method=critic_opt_method,
                    tol=1e-3,
                    bounds=bounds,
                    options=critic_opt_options,
                ).x

        if check_persistence_of_excitation:
            # Adjust the weight change by the replay condition number
            critic_weight_tensor_change *= (
                1
                / np.linalg.cond(self.observation_buffer)
                * 1
                / np.linalg.cond(self.action_buffer)
            )

        return np.clip(
            self.critic_weight_tensor + critic_weight_tensor_change,
            self.critic_weight_min,
            self.critic_weight_max,
        )

    def actor_obj(self, action_change, critic_weight_tensor, observation):
        """
        Objective function for actor learning.

        """

        result = self.critic_model(
            critic_weight_tensor, observation, self.action_curr + action_change
        )

        # Using nominal stabilizer as a pivot
        # result = self.critic_model(
        #     critic_weight_tensor,
        #     observation,
        #     self.get_safe_action(observation) + action_change,
        # )

        result += self.action_change_penalty_coeff * norm(action_change)

        return result

    def get_optimized_action(self, critic_weight_tensor, observation):

        actor_opt_method = "SLSQP"
        if actor_opt_method == "trust-constr":
            actor_opt_options = {
                "maxiter": 40,
                "disp": False,
            }  #'disp': True, 'verbose': 2}
        else:
            actor_opt_options = {
                "maxiter": 40,
                "maxfev": 60,
                "disp": False,
                "adaptive": True,
                "xatol": 1e-3,
                "fatol": 1e-3,
            }  # 'disp': True, 'verbose': 2}

        action_change_start_guess = np.zeros(self.dim_action)

        bounds = sp.optimize.Bounds(
            self.action_min, self.action_max, keep_feasible=True
        )

        action_change = minimize(
            lambda action_change: self.actor_obj(
                action_change, critic_weight_tensor, observation
            ),
            action_change_start_guess,
            method=actor_opt_method,
            tol=1e-3,
            bounds=bounds,
            options=actor_opt_options,
        ).x

        return self.action_curr + action_change

    def update_calf_state(self, critic_weight_tensor, observation, action):
        self.critic_weight_tensor_safe = critic_weight_tensor
        self.observation_safe = observation
        self.action_safe = action

        self.observation_buffer_safe = push_vec(
            self.observation_buffer_safe, observation
        )
        self.action_buffer_safe = push_vec(self.action_buffer_safe, action)

        self.calf_count += 1

    def calf_filter(self, critic_weight_tensor, observation, action):
        """
        If CALF constraints are satisfied, put the specified action through and update the CALF's state
        (safe weights, observation, action).
        Otherwise, return a safe action, do not update the CALF's state.

        """

        critic_low_kappa = self.critic_low_kappa_coeff * norm(observation) ** 2
        critic_up_kappa = self.critic_up_kappa_coeff * norm(observation) ** 2

        sample = np.random.rand()

        if (
            -self.critic_max_desired_decay
            <= self.calf_diff(critic_weight_tensor, observation, action)
            <= -self.critic_desired_decay
            and critic_low_kappa
            <= self.critic_model(
                critic_weight_tensor,
                observation,
                action,
            )
            <= critic_up_kappa
            or sample <= self.relax_probability
        ):

            self.update_calf_state(critic_weight_tensor, observation, action)

            return action

        else:

            self.safe_count += 1
            return self.get_safe_action(observation)

    def get_safe_action(self, observation: np.ndarray) -> np.ndarray:

        params = self.system._parameters
        mass, grav_const, length = (
            params["mass"],
            params["grav_const"],
            params["length"],
        )

        angle = observation[0, 0]
        angle_velocity = observation[0, 1]

        energy_total = (
            mass * grav_const * length * (np.cos(angle) - 1) / 2
            + 1 / 2 * self.system.pendulum_moment_inertia() * angle_velocity**2
        )
        energy_control_action = -self.gain * np.sign(angle_velocity * energy_total)

        action = hard_switch(
            signal1=energy_control_action,
            signal2=-self.pd_coeffs[0] * np.sin(angle)
            - self.pd_coeffs[1] * angle_velocity,
            condition=np.cos(angle) <= self.switch_loc,
        )

        return action

    def get_reset_action(self, observation):
        """
        This controller drives the system to the initial state. Currently implemented for pendulum.

        """

        p_coeff = 1.0
        d_coeff = 1.0

        angle = observation[0, 0]
        angle_velocity = observation[0, 1]

        action = -p_coeff * (-1 - np.cos(angle)) - d_coeff * angle_velocity

        return action

    def get_action(self, observation: np.ndarray) -> np.ndarray:

        # When episode ended
        if self.clock % self.episode_num_steps == 0 and not self.in_state_of_reset:
            self.in_state_of_reset = True
            self.reset_clock = 1
            self.episode_count += 1
            self.clock = 1

        if not self.in_state_of_reset:

            # Update replay buffers
            self.action_buffer = push_vec(self.action_buffer, self.action_curr)
            self.observation_buffer = push_vec(self.observation_buffer, observation)

            # Update score (cumulative objective)
            self.score += (
                self.run_obj(observation, self.action_curr)
                * self.action_sampling_period
            )

            # Update action
            new_action = self.get_optimized_action(
                self.critic_weight_tensor, observation
            )

            # Compute new critic weights
            self.critic_weight_tensor = self.get_optimized_critic_weights(
                observation,
                self.action_curr,
                use_calf_constraints=False,
                use_grad_descent=True,
            )

            # Apply relax probability annealing
            self.relax_probability = self.relax_probability * self.clock ** (
                -self.relax_probability_fading_factor
            )
            angle = observation[0, 0]
            if 1 - np.cos(angle) <= 0.2:
                self.relax_probability = 0.0

            # Apply CALF filter that checks constraint satisfaction and updates the CALF's state
            action = self.calf_filter(
                self.critic_weight_tensor, observation, new_action
            )

            # DEBUG
            # action = self.get_safe_action(observation)
            # action = self.get_reset_action(observation)
            # /DEBUG

            # DEBUG
            np.set_printoptions(precision=3)

            if self.debug_print_counter % 50 == 0:
                print(
                    "--DEBUG-- reward: %4.2f score: %4.2f"
                    % (-self.run_obj(observation, action), -self.score)
                )
                print("--DEBUG-- critic weights:", self.critic_weight_tensor)
                print("--DEBUG-- CALF counter:", self.calf_count)
                print("--DEBUG-- Safe counter:", self.safe_count)

            self.debug_print_counter += 1

            # /DEBUG

            # Update internal step counter
            self.clock += 1

        else:

            if self.reset_clock == 1:
                print(
                    "--------------------------EPISODE ",
                    self.episode_count,
                    " ENDED--------------------------",
                )
                print("--------------------------FINAL SCORE: %4.2f" % -self.score)
                self.score = 0

            if self.reset_clock < self.reset_num_steps:
                action = self.get_reset_action(observation)
            else:
                self.in_state_of_reset = False
                action = self.action_init
                self.update_calf_state(
                    self.critic_weight_tensor_safe, observation, action
                )
                self.relax_probability = self.relax_probability_init
            self.reset_clock += 1

        # Apply action bounds
        action = np.clip(
            action,
            self.action_min,
            self.action_max,
        )

        # Force proper dimensionsing according to the convention
        action = to_row_vec(action)

        # Update current action
        self.action_curr = action

        return action
