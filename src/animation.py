from regelum.animation import ThreeWheeledRobotAnimation, AnimationCallback

import matplotlib.pyplot as plt

from .objective import SpotObjective
import omegaconf
from pathlib import Path


class ThreeWheeledRobotAnimationWithNewLims(ThreeWheeledRobotAnimation):
    """Animator for the 3wheel-robot with custom x- and y-plane limits."""

    def setup(self):
        super().setup()

    def lim(self, *args, **kwargs):
        self.ax.set_xlim(-1.3, 1.3)
        self.ax.set_ylim(-1.3, 1.3)


class ThreeWheeledRobotAnimationWithSpot(ThreeWheeledRobotAnimation):
    def setup(self):
        super().setup()
        config_running_objective = omegaconf.OmegaConf.load(
            Path(__file__).parent.parent / "presets" / "scenario" / "mpc_scenario.yaml"
        )
        self.ax.add_patch(
            plt.Circle(
                (
                    config_running_objective[r"spot_mean_center_x%%"],
                    config_running_objective[r"spot_mean_center_y%%"],
                ),
                config_running_objective[r"spot_std%%"],
                color="r",
                alpha=0.66,
            )
        )
        self.ax.add_patch(
            plt.Circle(
                (
                    config_running_objective[r"spot_mean_center_x%%"],
                    config_running_objective[r"spot_mean_center_y%%"],
                ),
                2 * config_running_objective[r"spot_std%%"],
                color="r",
                alpha=0.29,
            )
        )
        self.ax.add_patch(
            plt.Circle(
                (
                    config_running_objective[r"spot_mean_center_x%%"],
                    config_running_objective[r"spot_mean_center_y%%"],
                ),
                3 * config_running_objective[r"spot_std%%"],
                color="r",
                alpha=0.15,
            )
        )

    def lim(self, *args, **kwargs):
        self.ax.set_xlim(-1.3, 1.3)
        self.ax.set_ylim(-1.3, 1.3)


class SpotObjectiveAnimation(AnimationCallback):  # inherit from this
    def setup(self):
        super().setup()
        self.circle = plt.Circle((0, 0), 0.5, color="r")
        self.ax.add_patch(self.circle)

    def is_target(self, obj, method, output, triggers):
        return isinstance(obj, SpotObjective) and method == "__call__"

    def on_function_call(self, obj: SpotObjective, method, output):
        self.circle.set(center=(obj.x_center, obj.y_center), radius=obj.r)
