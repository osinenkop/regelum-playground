from regelum.utils import rg
from regelum.system import InvertedPendulum


class InvertedPendulum(InvertedPendulum):
    _parameters = {"mass": 0.127, "grav_const": 9.81, "length": 0.337}

    def pendulum_moment(self):
        return self._parameters["mass"] * self._parameters["length"] ** 2 / 3

    def _compute_state_dynamics(self, time, state, inputs):
        Dstate = rg.zeros(
            self.dim_state,
            prototype=(state, inputs),
        )

        mass, grav_const, length = (
            self._parameters["mass"],
            self._parameters["grav_const"],
            self._parameters["length"],
        )
        Dstate[0] = state[1]
        Dstate[1] = (
            grav_const * mass * length * rg.sin(state[0]) / 2 + inputs[0]
        ) / self.pendulum_moment()

        return Dstate


class InvertedPendulumWithFriction(InvertedPendulum):
    _parameters = {
        "mass": 0.127,
        "grav_const": 9.81,
        "length": 0.337,
        "friction_coeff": 0.08,
    }

    def _compute_state_dynamics(self, time, state, inputs):
        Dstate = rg.zeros(
            self.dim_state,
            prototype=(state, inputs),
        )

        mass, grav_const, length, friction_coeff = (
            self._parameters["mass"],
            self._parameters["grav_const"],
            self._parameters["length"],
            self._parameters["friction_coeff"],
        )

        Dstate[0] = state[1]
        Dstate[1] = (
            grav_const * mass * length * rg.sin(state[0]) / 2 + inputs[0]
        ) / self.pendulum_moment() - friction_coeff * state[1] ** 2 * rg.sign(state[1])

        return Dstate


class InvertedPendulumWithMotor(InvertedPendulum):
    _parameters = {
        "mass": 0.127,
        "grav_const": 9.81,
        "length": 0.337,
        "tau_motor": 0.05,
        "m_motor": 0.1,
        "r_motor": 0.04,
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
    _action_bounds = [[-1.0, 1.0]]

    def motor_moment(self):
        return self._parameters["m_motor"] * self._parameters["r_motor"] ** 2 / 2

    def _compute_state_dynamics(self, time, state, inputs):
        Dstate = rg.zeros(
            self.dim_state,
            prototype=(state, inputs),
        )

        mass, grav_const, length, tau_motor = (
            self._parameters["mass"],
            self._parameters["grav_const"],
            self._parameters["length"],
            self._parameters["tau_motor"],
        )

        Dstate[0] = state[1]
        Dstate[1] = (mass * grav_const * length * rg.sin(state[0]) / 2 + state[2]) / (
            self.pendulum_moment() + self.motor_moment()
        )
        Dstate[2] = (inputs[0] - state[2]) / tau_motor

        return Dstate
