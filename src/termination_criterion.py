import numpy as np


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

        if (
            len(self.buffer_times) == 0
            or time - self.buffer_times[0] < self.required_hold_time
        ):
            self.buffer_times.append(np.copy(time))
            self.buffer_states.append(np.copy(state))
        else:
            self.buffer_states.append(np.copy(state))
            self.buffer_states.pop(0)

            self.buffer_times.append(np.copy(time))
            self.buffer_times.pop(0)

        return np.all(
            np.mean(np.abs(self.buffer_states), axis=0)
            < np.array([[self.angle_abs_bound, self.angle_vel_abs_bound]])
        )
