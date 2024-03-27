from src.policy import MyPolicy
import numpy as np
from regelum.simulator import CasADi
import regelum as rg
from regelum.scenario import Scenario
from regelum.system import ThreeWheeledRobotKinematic


@rg.main(config_name="main")
def main(cfg):

    # Define the initial state (initial position of the kinematic point).
    initial_state = np.array([[2.0, 2.0, 2.0]])  # Start at position (2, 2)

    # Initialize the kinematic point system.
    system = ThreeWheeledRobotKinematic()

    # Instantiate a simulator for the kinematic point system.
    simulator = CasADi(
        system=system, state_init=initial_state, time_final=4, max_step=0.01
    )

    policy = MyPolicy()

    Scenario(policy=policy, simulator=simulator, sampling_time=0.01).run()


if __name__ == "__main__":
    main()
