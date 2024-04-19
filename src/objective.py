from regelum import objective
import numpy as np
from typing import Union
from regelum.utils import rg


class SpotObjective(objective.RunningObjective):
    def __init__(self, x, y, r, magnitude):
        self.x_center = x
        self.y_center = y
        self.r = r
        self.magnitude = magnitude

    @apply_callbacks()  # noqa
    def __call__(self, observation, action):
        x, y = observation[:2]
        return (
            (x - self.x_center) ** 2 + (y - self.y_center) ** 2 <= self.r**2
        ) * self.magnitude


class ThreeWheeledRobotCostWithSpot(objective.RunningObjective):
    def __init__(
        self,
        quadratic_model,
        gaussian_gain: float,
        gaussian_mean: Union[list[float], np.ndarray],
        gaussian_std: float,
    ):
        self.quadratic_model = quadratic_model
        self.gaussian_gain = gaussian_gain
        self.gaussian_mean = np.array(gaussian_mean)
        self.gaussian_std = gaussian_std

    def __call__(
        self,
        observation,
        action,
        is_save_batch_format: bool = False,
    ):
        spot_cost = (
            self.gaussian_gain
            * rg.exp(
                -(
                    (observation[:, 0] - self.gaussian_mean[0]) ** 2
                    + (observation[:, 1] - self.gaussian_mean[1]) ** 2
                )
                / (2 * self.gaussian_std**2)
            )
            / (2 * np.pi * self.gaussian_std**2)
        )

        quadratic_cost = self.quadratic_model(observation, action)
        cost = quadratic_cost + spot_cost

        if is_save_batch_format:
            return cost
        else:
            return cost[0, 0]

        # if is_save_batch_format:
        #     return rg.array(
        #         rg.array(quadratic_cost, prototype=observation),
        #         prototype=observation,
        #     )
        # else:
        #     return quadratic_cost
