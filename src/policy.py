from regelum.policy import Policy
import numpy as np


class MyPolicy(Policy):
    def __init__(self):
        super().__init__()

    def get_action(self, observation):
        return np.array([[6, 2]])
