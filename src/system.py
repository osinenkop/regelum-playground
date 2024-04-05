from regelum.system import System
from regelum.utils import rg
from regelum import callback
from regelum.system import InvertedPendulum


class InvertedPendulum(InvertedPendulum):
    _parameters = {"m": 0.127, "g": 9.81, "l": 0.337}

    def _compute_state_dynamics(self, time, state, inputs):
        Dstate = rg.zeros(
            self.dim_state,
            prototype=(state, inputs),
        )

        m, g, l = (
            self._parameters["m"],
            self._parameters["g"],
            self._parameters["l"],
        )
        pendulum_moment_of_inertia = m * l**2
        Dstate[0] = state[1]
        Dstate[1] = (
            g * m * l**2 * rg.sin(state[0]) + inputs[0]
        ) / pendulum_moment_of_inertia

        return Dstate


class InvertedPendulumWithFriction(InvertedPendulum):
    _parameters = {"m": 0.127, "g": 9.81, "l": 0.337, "c": 0.08}

    def _compute_state_dynamics(self, time, state, inputs):
        Dstate = rg.zeros(
            self.dim_state,
            prototype=(state, inputs),
        )

        m, g, l, friction_coef = (
            self._parameters["m"],
            self._parameters["g"],
            self._parameters["l"],
            self._parameters["c"],
        )
        pendulum_moment_of_inertia = m * l**2 / 3

        Dstate[0] = state[1]
        Dstate[1] = (
            g * m * l**2 * rg.sin(state[0]) + inputs[0]
        ) / pendulum_moment_of_inertia - friction_coef * state[1] ** 2 * rg.sign(
            state[1]
        )

        return Dstate


class InvertedPendulumWithMotor(InvertedPendulum):
    _parameters = {
        "m": 0.127,
        "g": 9.81,
        "l": 0.337,
        "tau_motor": 0.25,
        "m_motor": 0.5,
        "r_motor": 0.05,
    }
    _dim_state = 3
    _dim_observation = 3
    _system_type = "diff_eqn"
    _dim_inputs = 1
    _observation_naming = _state_naming = [
        "angle [rad]",
        "angular velocity [rad/s]",
        "torque [kg*m**2/s**2]",
    ]
    _inputs_naming = ["motor [kg*m**2/s**2]"]
    _action_bounds = [[-10000.0, 10000.0]]

    def _compute_state_dynamics(self, time, state, inputs):
        Dstate = rg.zeros(
            self.dim_state,
            prototype=(state, inputs),
        )

        m, g, l, tau_motor = (
            self._parameters["m"],
            self._parameters["g"],
            self._parameters["l"],
            self._parameters["tau_motor"],
        )
        pendulum_moment_of_inertia = m * l**2
        motor_moment_of_inertia = (
            self._parameters["m_motor"] * self._parameters["r_motor"] ** 2 / 2
        )
        Dstate[0] = state[1]
        Dstate[1] = (m * g * l * rg.sin(state[0]) + state[2]) / (
            pendulum_moment_of_inertia  # + motor_moment_of_inertia
        )
        Dstate[2] = (inputs[0]) / tau_motor

        return Dstate
