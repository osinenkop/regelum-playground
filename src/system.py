from regelum.system import System
from regelum.utils import rg
from regelum import callback
from regelum.system import InvertedPendulum


class MYInvertedPendulum(InvertedPendulum):
    _parameters = {"m": 1, "g": 9.8, "l": 1.0}

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

        Dstate[0] = state[1]
        Dstate[1] = g / l * rg.sin(state[0]) + inputs[0] / (m * l**2)

        return Dstate


class InvertedPendulumWithFriction(InvertedPendulum):
    _parameters = {"m": 1, "g": 9.8, "l": 1.0, "c": 0.08}

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

        Dstate[0] = state[1]
        Dstate[1] = (
            g / l * rg.sin(state[0])
            + inputs[0] / (m * l**2)
            - friction_coef * state[1] ** 2 * rg.sign(state[1])
        )

        return Dstate


class InvertedPendulumWithMotor(InvertedPendulum):
    _parameters = {"m": 1, "g": 9.8, "l": 1.0, "J_motor": 10.0}
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
    _action_bounds = [[-200.0, 200.0]]

    def _compute_state_dynamics(self, time, state, inputs):
        Dstate = rg.zeros(
            self.dim_state,
            prototype=(state, inputs),
        )

        m, g, l, j_motor = (
            self._parameters["m"],
            self._parameters["g"],
            self._parameters["l"],
            self._parameters["J_motor"],
        )

        Dstate[0] = state[1]
        Dstate[1] = g / l * rg.sin(state[0]) + state[2] / (m * l**2)
        Dstate[2] = inputs[0] / j_motor

        return Dstate
