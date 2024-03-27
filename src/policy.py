from regelum.policy import Policy
import numpy as np


class InvPendulumPolicyPID(Policy):
    def __init__(self, pd_coefs, action_min, action_max):
        super().__init__()

        self.pid_coefs = np.array(pd_coefs).reshape(1, -1)
        self.action_min = action_min
        self.action_max = action_max

    def get_action(self, observation: np.ndarray):
        action = np.clip(
            (self.pid_coefs * observation).sum(),
            self.action_min,
            self.action_max,
        )
        return np.array([[action]])


class InvPendulumPolicyEnergyBased(Policy):
    def __inti__(self, action_min, action_max):
        super().__init__()

        # saving action bounds
        self.action_min = action_min
        self.action_max = action_max

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        action = np.clip(
            (np.array([-10, -10]) * observation).sum(),
            self.action_min,
            self.action_max,
        )
