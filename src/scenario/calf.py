from .base import CleanRLScenario
from regelum.simulator import Simulator
from regelum.objective import RunningObjective
from .calf_agent.calfv import AgentCALFV
from .calf_agent.calfq import AgentCALFQ
from typing import Union


# Note: The CleanRLScenario class is used for CALF due to its convenience.
# CALF is not natively implemented in CleanRL.
class CALFScenario(CleanRLScenario):
    """CALFScenario class is used to run the CALF algorithm."""

    def __init__(
        self,
        simulator: Simulator,
        running_objective: RunningObjective,
        total_timesteps: int,
        agent_calf: Union[AgentCALFV, AgentCALFQ],
    ):
        """Initialize the CALFScenario.

        This method sets up the scenario with the given simulator, running objective,
        total timesteps, and agent_calf. It initializes the environment, episode management,
        and other necessary attributes for running the scenario.

        Args:
            simulator: The simulator object used for the environment.
            running_objective: The running objective function for reward calculation.
            total_timesteps: The total number of timesteps to run the scenario.
            agent_calf: The CALF agent used in the scenario, either AgentCALFV or AgentCALFQ.
        """
        super().__init__(
            simulator, running_objective, total_timesteps=total_timesteps, device="cpu"
        )
        self.agent_calf = agent_calf

    def run(self):
        obs, _ = self.envs.reset()
        self.agent_calf.reset(obs_init=obs, global_step=0)

        for global_step in range(self.total_timesteps):
            action = self.agent_calf.get_action(obs)

            self.state = self.envs.envs[0].env.state.reshape(1, -1)
            self.time = self.envs.envs[0].env.simulator.time
            obs, rewards, terminations, truncations, infos = self.envs.step(action)
            self.post_compute_action(
                self.state,
                obs,
                action,
                float(rewards.reshape(-1)),
                self.time,
                global_step,
            )
            if "final_info" in infos:
                self.agent_calf.reset(obs_init=obs, global_step=global_step)
                self.save_episodic_return(
                    global_step=global_step, episodic_return=self.value
                )
                self.reload_scenario()
                self.reset_episode()
                self.reset_iteration()
