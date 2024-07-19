<!-- TOC start (generated with https://github.com/derlin/bitdowntoc) -->

- [About ](#about)
- [Getting started](#getting-started)
   * [Installation requirements](#installation-requirements)
- [Ready examples](#ready-examples)
   * [(EXPERIMENTAL) CALF-Q controller for pendulum](#calfq-pendulum) 
   * [Proportional-derivative (PD) controller for pendulum](#proportional-derivative-pd-controller-for-pendulum)
   * [Energy-based controller for pendulum](#energy-based-controller-for-pendulum)
   * [Energy-based controller for pendulum with joint friction](#energy-based-controller-for-pendulum-with-joint-friction)
   * [Energy-based controller with friction compenstation for pendulum with joint friction](#energy-based-controller-with-friction-compenstation-for-pendulum-with-joint-friction)
   * [Adaptive energ-based controller for pendulum with joint friction](#adaptive-energ-based-controller-for-pendulum-with-joint-friction)
   * [Backstepping controller for pendulum with motor dynamics](#backstepping-controller-for-pendulum-with-motor-dynamics)
   * [PD controller for pendulum with motor dynamics](#pd-controller-for-pendulum-with-motor-dynamics)
   * [Lyapunov-based controller for kinematic three-wheeled robot](#lyapunov-based-controller-for-kinematic-three-wheeled-robot)
   * [Backstepping controller for dynamic three-wheeled robot](#backstepping-controller-for-dynamic-three-wheeled-robot)
   * [Model-predictive controller for three-wheeled robot](#model-predictive-controller-for-three-wheeled-robot)
      + [On a plane with quadratic cost](#on-a-plane-with-quadratic-cost)
      + [On a plane with a Guassian spot of high cost](#on-a-plane-with-a-guassian-spot-of-high-cost)
- [Repo structure](#repo-structure)

<!-- TOC end -->

<!-- TOC --><a name="about"></a>
## About 

This is a playground based on [regelum-control](https://regelum.aidynamic.io), a framework for control and reinforcement learning.
It showcases various dynamical systems and controllers (also called policies).

<!-- TOC --><a name="getting-started"></a>
## Getting started

<!-- TOC --><a name="installation-requirements"></a>
### Installation requirements

If you are working in Windows, it is recommended to use WSL and, possibly, a display server like Xming to properly output graphics from WSL.
Before installing the [`requirements.txt`](./requirements.txt), it is recommended to create a virtual environment for your project, say, `pyenv` or `virtualenv`. The instructions on these are standard and may be found on the web.
For instance, in case of `virtualenv`, install, create a virtual environment, e.g., `rgenv` and then activate it, being in the folder into which you cloned the repo:

```shell
pip install virtualenv
python -m venv rgenv
source rgenv/bin/activate
```

Then, install necessary packages:

```shell
pip install -r requirements.txt
```
<!-- TOC --><a name="ready-examples"></a>
## Ready examples

Below are examples with respective terminal run commands. 

<!-- TOC --><a name="calfq-lunar"></a>
### (EXPERIMENTAL) Nominal controller for lunar lander
```shell
python run.py policy=pid_lunar_lander system=lunar_lander --interactive --fps=10 scenario=calfq_scenario scenario.N_iterations=3
```

Create config for calf-q on lunar lander and paste the name of the config after policy=...

<!-- TOC --><a name="calfq-pendulum"></a>
### (EXPERIMENTAL) CALF-Q controller for pendulum

```shell
python run.py policy=rc_calfq system=inv_pendulum --interactive --fps=10
```

See `get_action`, there is a `# DEBUG` section where you can force nominal stabilizer for testing.
Most CALF-Q settings are in the `__init__` of `InvertedPendulumRcognitaCALFQ` class.

<!-- TOC --><a name="proportional-derivative-pd-controller-for-pendulum"></a>
### Proportional-derivative (PD) controller for pendulum

```shell
python run.py policy=pd system=inv_pendulum --interactive --fps=10
```    

Observe it doesn't work with the default action bounds.
Further, we will see how the energy-based controller does the job.

To run the PD controller with custom action bounds:

```shell
python run.py policy=pd system=inv_pendulum policy.action_min=-2 policy.action_max=2 --interactive --fps=10
```  

Making the bounds sufficiently large will help the PD controller to upswing and upright hold the pendulum.

<!-- TOC --><a name="energy-based-controller-for-pendulum"></a>
### Energy-based controller for pendulum

```shell
python run.py policy=energy_based system=inv_pendulum --interactive --fps=10
```  

<!-- TOC --><a name="energy-based-controller-for-pendulum-with-joint-friction"></a>
### Energy-based controller for pendulum with joint friction

```shell
python run.py policy=energy_based system=inv_pendulum_with_friction --interactive --fps=10
```  

Observe it doesn't work due to ignorance of friction.

<!-- TOC --><a name="energy-based-controller-with-friction-compenstation-for-pendulum-with-joint-friction"></a>
### Energy-based controller with friction compenstation for pendulum with joint friction

```shell
python run.py policy=energy_based_friction_compensation system=inv_pendulum_with_friction --interactive --fps=10
```

This works better than the last one.

<!-- TOC --><a name="adaptive-energ-based-controller-for-pendulum-with-joint-friction"></a>
### Adaptive energ-based controller for pendulum with joint friction

```shell
python run.py policy=energy_based_friction_adaptive system=inv_pendulum_with_friction --interactive --fps=10
```  

This should also work.

<!-- TOC --><a name="backstepping-controller-for-pendulum-with-motor-dynamics"></a>
### Backstepping controller for pendulum with motor dynamics

This showcases the use of backstepping.

```shell
python run.py policy=backstepping system=inv_pendulum_with_motor --interactive --fps=10 
``` 

<!-- TOC --><a name="pd-controller-for-pendulum-with-motor-dynamics"></a>
### PD controller for pendulum with motor dynamics

```shell
python run.py policy=motor_pd system=inv_pendulum_with_motor --interactive --fps=10 
```  
Notice for it fails to do the job.

<!-- TOC --><a name="lyapunov-based-controller-for-kinematic-three-wheeled-robot"></a>
### Lyapunov-based controller for kinematic three-wheeled robot

```shell
python run.py policy=3wrobot_kin_min_grad_clf initial_conditions=3wrobot_kin system=3wrobot_kin common.sampling_time=0.01 --interactive --fps=10 
```  

<!-- TOC --><a name="backstepping-controller-for-dynamic-three-wheeled-robot"></a>
### Backstepping controller for dynamic three-wheeled robot

```shell
python run.py policy=3wrobot_dyn_min_grad_clf initial_conditions=3wrobot_dyn system=3wrobot_dyn common.sampling_time=0.01 --interactive --fps=10 
```

<!-- TOC --><a name="model-predictive-controller-for-three-wheeled-robot"></a>
### Model-predictive controller for three-wheeled robot

<!-- TOC --><a name="on-a-plane-with-quadratic-cost"></a>
#### On a plane with quadratic cost

```shell
python run.py \
  initial_conditions=3wrobot_kin_with_spot \
  system=3wrobot_kin \
  scenario=mpc_scenario \
  scenario.running_objective.spot_gain=0 \
  scenario.prediction_horizon=3 \
  --interactive \
  --fps=10
```

<!-- TOC --><a name="on-a-plane-with-a-guassian-spot-of-high-cost"></a>
#### On a plane with a Guassian spot of high cost

```shell
python run.py \
  initial_conditions=3wrobot_kin_with_spot \
  system=3wrobot_kin_with_spot \
  scenario=mpc_scenario \
  scenario.running_objective.spot_gain=100 \
  scenario.prediction_horizon=3 \
  --interactive \
  --fps=10
```
Notice how the robot avoids the spot with high cost

### Proximal Policy Optimizaion on Pendulum

```
python run.py \
    scenario=ppo_scenario \
    system=inv_pendulum \
    common.time_final=5 \
    scenario.N_episodes=1 
```


> **Note:**
>
> For the `--fps` parameter, you can select any suitable value to ensure a smooth experience (e.g., `--fps=2`, `--fps=10`, `--fps=20`, etc.).

<!-- TOC --><a name="repo-structure"></a>
## Repo structure

- [`run.py`](./run.py): The main executable script.
- [`src/`](./src/): Contains the source code of the repo.
    - [`policy.py`](./src/policy.py): Implements the PD and energy-based controllers.
    - [`system.py`](./src/system.py): Implements the pendulum system and pendulum system with friction.
- [`presets/`](./presets/): Houses configuration files.
    - [`common/`](./presets/common): General configurations.
        - [`common.yaml`](./presets/common/common.yaml): Settings for common variables (like sampling time).
    - [`policy/`](./presets/policy/): Controller-specific configurations.
        - [`pd.yaml`](./presets/policy/pd.yaml): Settings for the proportional-derivative (PD) controller.
        - [`energy_based.yaml`](./presets/policy/energy_based.yaml): Settings for the energy-based controller.
        - [`energy_based_friction_compensation.yaml`](./presets/policy/energy_based_friction_compensation.yaml): Settings for the energy-based controller with friction compensation.
        - [`energy_based_friction_adaptive.yaml`](./presets/policy/energy_based_friction_adaptive.yaml): Settings for the adaptive energy-based controller with adaptive friction.

    - [`scenario/`](./presets/scenario/): Scenario configuration folder. Scenario is a top-level module in regelum.
        - [`scenario.yaml`](./presets/scenario/scenario.yaml): Scenario settings.
    - [`simulator`](./presets/simulator/): Simulator configuration folder.
        - [`casadi.yaml`](./presets/simulator/casadi.yaml): Configurations for the [CasADi](https://web.casadi.org/) [RK](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods) simulator.


