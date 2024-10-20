from regelum.utils import rg
from regelum.system import (
    Pendulum,
    ThreeWheeledRobotKinematic,
    ThreeWheeledRobotDynamic,
    LunarLander,
)
from regelum.animation import DefaultAnimation
from .animation import (
    ThreeWheeledRobotAnimationWithNewLims,
    ThreeWheeledRobotAnimationWithSpot,
)
from regelum.callback import detach
import numpy as np

# In the following two classes we want to alter their respective animation callbacks, so we:
# - detach the default animations
# - attach `DefaultAnimation` of the action and state plots
# - attach a new animation with new x- and y-limtis of [-1.3, 1.3]
#
# To learn more on customizing animations in regelum, go to https://regelum.aidynamic.io/tutorials/animations/


@ThreeWheeledRobotAnimationWithNewLims.attach
@DefaultAnimation.attach
@detach
class MyThreeWheeledRobotDynamic(ThreeWheeledRobotDynamic):
    """The parameters correspond roughly to those of Robotis TurtleBot3."""

    _parameters = {"m": 1, "I": 0.005}
    action_bounds = [[-1, 1], [-1, 1]]


@ThreeWheeledRobotAnimationWithNewLims.attach
@DefaultAnimation.attach
@detach
class MyThreeWheeledRobotKinematic(ThreeWheeledRobotKinematic):
    """The parameters correspond to those of Robotis TurtleBot3."""

    action_bounds = [[-0.22, 0.22], [-2.84, 2.84]]


@ThreeWheeledRobotAnimationWithSpot.attach
@DefaultAnimation.attach
@detach
class ThreeWheeledRobotKinematicWithSpot(MyThreeWheeledRobotKinematic): ...


class Pendulum(Pendulum):
    """The parameters of this system roughly resemble those of a Quanser Rotary Pendulum."""

    _parameters = {"mass": 0.127, "grav_const": 9.81, "length": 0.337}
    _action_bounds = [[-0.1, 0.1]]

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


class PendulumLooseBounds(Pendulum):
    """The parameters of this system resemble those of a Quanser Rotary Pendulum, but with action bounds large enough to stabilize the system using a PD controller."""

    _action_bounds = [[-0.3, 0.3]]


class PendulumWithGymObservation(Pendulum):
    _dim_observation = 3
    # _parameters = {"mass": 1.0, "grav_const": 9.81, "length": 1.0}
    # _action_bounds = [[-2, 2]]

    def _get_observation(self, time, state, inputs):
        observation = rg.zeros(self._dim_observation, prototype=state)
        observation[0] = rg.cos(state[0])
        observation[1] = rg.sin(state[0])
        observation[2] = state[1]
        return observation


class PendulumWithFriction(Pendulum):
    """The parameters of this system roughly resemble those of a Quanser Rotary Pendulum."""

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


class PendulumWithMotor(Pendulum):
    """The parameters of this system roughly resemble those of a Quanser Rotary Pendulum."""

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


class LunarLanderWithOffset(LunarLander):
    def _get_observation(self, time, state, inputs):
        return state - rg.array(
            np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]).reshape(*state.shape),
            prototype=state,
            _force_numeric=True,
        )
