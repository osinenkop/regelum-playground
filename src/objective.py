import regelum
from regelum import objective


class SpotObjective(objective.RunningObjective):
    @apply_callbacks()
    def __init__(self, x, y, r, magnitude):
        self.x_center = x
        self.y_center = y
        self.r = r
        self.magnitude = magnitude

    def __call__(self, observation, action):
        x, y = observation[:2]
        return ((x - self.x_center) ** 2 + (y - self.y_center) ** 2 <= self.r ** 2) * self.magnitude

