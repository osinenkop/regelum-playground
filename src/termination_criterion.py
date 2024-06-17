import numpy as np


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


class SuccessTerminationCriterionInvPendulum:
    def __init__(
        self,
        required_hold_time: float,
        angle_abs_bound: float,
        angle_vel_abs_bound: float,
    ) -> None:
        self.required_hold_time = required_hold_time
        self.buffer_states = []
        self.buffer_times = []
        self.angle_abs_bound = angle_abs_bound
        self.angle_vel_abs_bound = angle_vel_abs_bound

    def __call__(self, time: float, state: np.ndarray, action: np.ndarray) -> bool:
        if len(self.buffer_times) > 0 and time < self.buffer_times[-1]:
            self.buffer_states = []
            self.buffer_times = []

        self.buffer_times.append(np.copy(time))
        self.buffer_states.append(np.copy(state))

        while (
            len(self.buffer_times) > 0
            and time - self.buffer_times[0] > self.required_hold_time
        ):
            self.buffer_states.pop(0)
            self.buffer_times.pop(0)

        buffer_states_array = np.array(self.buffer_states)
        buffer_states_array[:, 0] = angle_normalize(buffer_states_array[:, 0])

        return np.all(
            np.mean(np.abs(buffer_states_array), axis=0)
            < np.array([[self.angle_abs_bound, self.angle_vel_abs_bound]])
        )
