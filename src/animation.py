###############################################################################
# Here we adjust the animation pane by detaching, altering and then attaching it back.
# For this, we redefine the system classes

from regelum.animation import ThreeWheeledRobotAnimation

class MyThreeWheeledRobotAnimation(ThreeWheeledRobotAnimation):

    def lim(self, *args, **kwargs):
        self.ax.set_xlim(-1.3, 1.3)
        self.ax.set_ylim(-1.3, 1.3)

from regelum import callback
from regelum.animation import DefaultAnimation
from .animation import MyThreeWheeledRobotAnimation

MyThreeWheeledRobotKinematic = callback.detach(ThreeWheeledRobotKinematic)
MyThreeWheeledRobotKinematic = DefaultAnimation.attach(MyThreeWheeledRobotKinematic)
MyThreeWheeledRobotKinematic = MyThreeWheeledRobotAnimation.attach(
    MyThreeWheeledRobotKinematic
)

MyThreeWheeledRobotDynamic = callback.detach(ThreeWheeledRobotDynamic)
MyThreeWheeledRobotDynamic = DefaultAnimation.attach(MyThreeWheeledRobotDynamic)
MyThreeWheeledRobotDynamic = MyThreeWheeledRobotAnimation.attach(
    MyThreeWheeledRobotDynamic
)

###############################################################################