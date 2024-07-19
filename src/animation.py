from regelum.animation import ThreeWheeledRobotAnimation, LunarLanderAnimation

import matplotlib.pyplot as plt

import omegaconf
from pathlib import Path
import numpy as np


class ThreeWheeledRobotAnimationWithNewLims(ThreeWheeledRobotAnimation):
    """Animator for the 3wheel-robot with custom x- and y-plane limits."""

    def setup(self):
        super().setup()

    def lim(self, *args, **kwargs):
        self.ax.set_xlim(-1.5, 1.5)
        self.ax.set_ylim(-1.5, 1.5)


class ThreeWheeledRobotAnimationWithSpot(ThreeWheeledRobotAnimation):
    """Animator for the 3wheel-robot with custom x-, y-plane limits and gaussian cost spot."""

    def setup(self):
        super().setup()
        config_running_objective = omegaconf.OmegaConf.load(
            Path(__file__).parent.parent / "presets" / "scenario" / "mpc_scenario.yaml"
        )["running_objective"]
        center = (
            config_running_objective["spot_x_center"],
            config_running_objective["spot_y_center"],
        )
        self.ax.add_patch(
            plt.Circle(
                center,
                config_running_objective["spot_std"],
                color="r",
                alpha=0.66,
            )
        )
        self.ax.add_patch(
            plt.Circle(
                center,
                2 * config_running_objective["spot_std"],
                color="r",
                alpha=0.29,
            )
        )
        self.ax.add_patch(
            plt.Circle(
                center,
                3 * config_running_objective["spot_std"],
                color="r",
                alpha=0.15,
            )
        )

    def lim(self, *args, **kwargs):
        self.ax.set_xlim(-1.5, 1.5)
        self.ax.set_ylim(-1.5, 1.5)


class LunarLanderAnimationWithAdjustedAngle(LunarLanderAnimation):
    def setup(self):
        super().setup()

    def on_trigger(self, _):
        self.add_frame(
            x=self.system_state[0],
            y=self.system_state[1],
            theta=self.system_state[2] + np.pi,
        )
