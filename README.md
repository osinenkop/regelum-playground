## Overview 

This is a playground based on [regelum-control](https://regelum.aidynamic.io/systems/inv_pendulum/), a framework for control and reinforcement learning.
It showcases various systems and controllers.
If you are working in Windows, it is recommended to use WSL and, possibly, a display server like Xming to properly output graphics from WSL.
Note, in [regelum-control](https://regelum.aidynamic.io/systems/inv_pendulum/), we refer to a controller by the term "policy", whence those two are interchangeable.

## Getting started

### Install requirements

Before installing the [`requirements.txt`](./requirements.txt), it is recommended to create a virtual environment for your project, say, `pyenv` or `virtualenv`. The instructions on these are standard and may be found on the web.

```shell
pip install -r requirements.txt
```

Below are examples with respective terminal run commands. 

### Proportional-derivative (PD) controller for inverted pendulum

```shell
python run.py policy=pd system=inv_pendulum --interactive --fps=10
```    

Observe it doesn't work with default action bounds.

To run the PD controller with custom action bounds:

```shell
python run.py policy=pd system=inv_pendulum policy.action_min=-2 policy.action_max=2 --interactive --fps=10
```  

### Energy-based controller for inverted pendulum

```shell
python run.py policy=energy_based system=inv_pendulum --interactive --fps=10
```  

### Energy-based controller for inverted pendulum with joint friction

```shell
python run.py policy=energy_based system=inv_pendulum_with_friction --interactive --fps=10
```  

Observe doesn't work.

### Energy-based controller with friction compenstation for inverted pendulum with joint friction

```shell
python run.py policy=energy_based_friction_compensation system=inv_pendulum_with_friction --interactive --fps=10
```

This works better than the last one.

### Adaptive energ-based controller for inverted pendulum with joint friction

```shell
python run.py policy=energy_based_friction_adaptive system=inv_pendulum_with_friction --interactive --fps=10
```  

Should also work.

### Backstepping controller for inverted pendulum with motor dynamics

```shell
python run.py policy=backstepping system=inv_pendulum_with_motor --interactive --fps=10 
``` 

### PD controller for inverted pendulum with motor dynamics

```shell
python run.py policy=motor_pd system=inv_pendulum_with_motor --interactive --fps=10 
```  

### Lyapunov-based controller for kinematic three-wheeled robot

```shell
python run.py policy=3wrobot_kin_dissasembled system=3wrobot_kin --interactive --fps=10
```  

### Backstepping controller for dynamic three-wheeled robot

```shell
python run.py policy=3wrobot_kin_dissasembled system=3wrobot_kin --interactive --fps=10
```  

> **Note:**
>
> For the `--fps` parameter, you can select any suitable value to ensure a smooth experience (e.g., `--fps=2`, `--fps=10`, `--fps=20`, etc.).

## Repo structure

- [`run.py`](./run.py): The main executable script.
- [`src/`](./src/): Contains the source code of the repo.
    - [`policy.py`](./src/policy.py): Implements the PD and energy-based controllers.
    - [`system.py`](./src/system.py): Implements the inverted pendulum system and inverted pendulum system with friction.
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


