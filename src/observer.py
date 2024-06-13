from regelum.observer import Observer
import numpy as np


class PendulumObserver(Observer):
    def __init__(self):
        pass

    def get_state_estimation(self, t, observation: np.ndarray, action):
        # sin_theta, one_minus_cos_theta, x, theta_dot, x_dot = observation.reshape(-1)
        one_minus_cos_theta, sin_theta, angle_vel = observation.reshape(-1)

        return np.array([[np.arctan2(sin_theta, 1 - one_minus_cos_theta), angle_vel]])
