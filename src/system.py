from regelum.utils import rg
from regelum.system import (
    InvertedPendulum,
    ThreeWheeledRobotKinematic,
    ThreeWheeledRobotDynamic,
)
from regelum import callback
from regelum.animation import DefaultAnimation
from .animation import MyThreeWheeledRobotAnimation


MyThreeWheeledRobotKinematic = callback.detach(ThreeWheeledRobotKinematic)
MyThreeWheeledRobotKinematic = DefaultAnimation.attach(MyThreeWheeledRobotKinematic)
MyThreeWheeledRobotKinematic = MyThreeWheeledRobotAnimation.attach(
    MyThreeWheeledRobotKinematic
)

MyThreeWheeledRobotDynamic = callback.detach(ThreeWheeledRobotDynamic)
MyThreeWheeledRobotDynamic = DefaultAnimation.attach(MyThreeWheeledRobotDynamic)
MyThreeWheeledRobotDynamic = MyThreeWheeledRobotAnimation.attach(
    MyThreeWheeledRobotDynamic
)


class MyThreeWheeledRobotDynamic(MyThreeWheeledRobotDynamic):
    _parameters = {"m": 1, "I": 0.005}


class InvertedPendulum(InvertedPendulum):
    """Parameters of this system roughly resemble those of a Quanser test stand Rotary Inverted Pendulum."""

    _parameters = {"mass": 0.127, "grav_const": 9.81, "length": 0.337}

    def pendulum_moment_inertia(self):
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
        ) / self.pendulum_moment_inertia()

        return Dstate


class InvertedPendulumWithFriction(InvertedPendulum):
    """Parameters of this system roughly resemble those of a Quanser test stand Rotary Inverted Pendulum."""

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
        ) / self.pendulum_moment_inertia() - friction_coeff * state[1] ** 2 * rg.sign(
            state[1]
        )

        return Dstate


class InvertedPendulumWithMotor(InvertedPendulum):
    """Parameters of this system roughly resemble those of a Quanser test stand Rotary Inverted Pendulum."""

    _parameters = {
        "mass": 0.127,
        "grav_const": 9.81,
        "length": 0.337,
        "motor_time_const": 0.05,
        "motor_mass": 0.1,
        "motor_radius": 0.04,
    }
    _dim_state = 3
    _dim_observation = 3
    _system_type = "diff_eqn"
    _dim_inputs = 1
    _observation_naming = _state_naming = [
        "angle [rad]",
        "angular velocity [rad/s]",
        "torque [N*m]",
    ]
    _inputs_naming = ["motor [N*m/s]"]
    _action_bounds = [[-1.0, 1.0]]

    def motor_moment(self):
        return (
            self._parameters["motor_mass"] * self._parameters["motor_radius"] ** 2 / 2
        )

    def _compute_state_dynamics(self, time, state, inputs):
        Dstate = rg.zeros(
            self.dim_state,
            prototype=(state, inputs),
        )

        mass, grav_const, length, motor_time_const = (
            self._parameters["mass"],
            self._parameters["grav_const"],
            self._parameters["length"],
            self._parameters["motor_time_const"],
        )

        Dstate[0] = state[1]
        Dstate[1] = (mass * grav_const * length * rg.sin(state[0]) / 2 + state[2]) / (
            self.pendulum_moment_inertia() + self.motor_moment()
        )
        Dstate[2] = (inputs[0] - state[2]) / motor_time_const

        return Dstate
