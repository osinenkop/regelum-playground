from regelum.system import System
from regelum.utils import rg
from regelum import callback
from regelum.system import InvertedPendulum


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
