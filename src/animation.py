from regelum.animation import ThreeWheeledRobotAnimation


class MyThreeWheeledRobotAnimation(ThreeWheeledRobotAnimation):

    def lim(self, *args, **kwargs):
        self.ax.set_xlim(-1.3, 1.3)
        self.ax.set_ylim(-1.3, 1.3)
