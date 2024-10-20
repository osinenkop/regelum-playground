from regelum import objective
import numpy as np
from typing import Union
from regelum.utils import rg
from regelum.model import ModelQuadLin


class ThreeWheeledRobotCostWithSpot(objective.RunningObjective):
    def __init__(
        self,
        quadratic_model: ModelQuadLin,
        spot_gain: float,
        spot_x_center: float,
        spot_y_center: float,
        spot_std: float,
    ):
        self.quadratic_model = quadratic_model
        self.spot_gain = spot_gain
        self.spot_x_center = spot_x_center
        self.spot_y_center = spot_y_center
        self.spot_std = spot_std

    def __call__(
        self,
        observation,
        action,
        is_save_batch_format: bool = False,
    ):
        spot_cost = (
            self.spot_gain
            * rg.exp(
                -(
                    (observation[:, 0] - self.spot_x_center) ** 2
                    + (observation[:, 1] - self.spot_y_center) ** 2
                )
                / (2 * self.spot_std**2)
            )
            / (2 * np.pi * self.spot_std**2)
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


def angle_normalize(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


class GymPendulumRunningObjective:
    def __call__(self, observation, action):
        if observation.shape[1] == 3:
            cos_angle = observation[0, 0]
            sin_angle = observation[0, 1]
            angle_vel = observation[0, 2]
            torque = action[0, 0]
            angle = np.arctan2(sin_angle, cos_angle)
            return angle_normalize(angle) ** 2 + 0.1 * angle_vel**2 + 0.001 * torque**2
        elif observation.shape[1] == 2:
            angle = observation[0, 0]
            angle_vel = observation[0, 1]
            torque = action[0, 0]
            return angle_normalize(angle) ** 2 + 0.1 * angle_vel**2 + 0.001 * torque**2
        else:
            raise ValueError("Invalid observation shape")
