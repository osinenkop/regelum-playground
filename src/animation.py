from regelum.animation import ThreeWheeledRobotAnimation, Animation
import matplotlib.pyplot as plt

import abc

from src.objective import SpotObjective


class ThreeWheeledRobotAnimationWithNewLims(ThreeWheeledRobotAnimation):
    """Animator for the 3wheel-robot with custom x- and y-plane limits."""

    def lim(self, *args, **kwargs):
        self.ax.set_xlim(-1.3, 1.3)
        self.ax.set_ylim(-1.3, 1.3)


class SpotObjectiveAnimation(Animation): # inherit from this
    def setup(self):
        super().setup()
        self.circle = plt.Circle((0, 0), 0, color='r')
        self.ax.add_patch(self.circle)

    def is_target(self, obj, method, output, triggers):
        return isinstance(obj, SpotObjective) and method == '__init__'

    def on_function_call(self, obj, method, output):
        self.circle.set(center=(obj.x, obj.y), radius=obj.r)
