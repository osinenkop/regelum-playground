from regelum.system import System
from regelum.utils import rg  # Essential for array computations


class MyThreeWheeledRobot(System):
    _name = "ThreeWheeledRobotKinematic"
    _system_type = "diff_eqn"
    _dim_state = 3
    _dim_inputs = 2
    _dim_observation = 3
    _observation_naming = _state_naming = ["x_rob", "y_rob", "vartheta"]
    _inputs_naming = ["v", "omega"]
    _action_bounds = [[-25.0, 25.0], [-5.0, 5.0]]

    def _compute_state_dynamics(self, time, state, inputs):
        """Calculate the robot's state dynamics."""

        # Placeholder for the right-hand side of the differential equations
        Dstate = rg.zeros(self._dim_state, prototype=state)  #

        # Element-wise calculation of the Dstate vector
        # based on the system's differential equations
        Dstate[0] = inputs[0] * rg.cos(state[2])  # v * cos(vartheta)
        Dstate[1] = inputs[0] * rg.sin(state[2])  # v * sin(vartheta)
        Dstate[2] = inputs[1]  # omega

        return Dstate
