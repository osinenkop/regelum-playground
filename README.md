## Overview 

This repository showcases the operation of PD regulator, energy-based controller and adaptive controller on an [inverted pendulum system](https://regelum.aidynamic.io/systems/inv_pendulum/), utilizing the [regelum-control](https://regelum.aidynamic.io/systems/inv_pendulum/) Python package. This package is designed for researchers and practitioners in reinforcement learning and optimal control.

## Getting started


### Step 1: install requirements

Before installing the [`requirements.txt`](./requirements.txt), it is recommended to create a virtual environment for your project.

```shell
pip install -r requirements.txt
```

### Step 2: Run PD Regulator

Execute the PD regulator using the following command in your terminal:

```shell
python run.py policy=pd --interactive --fps=10
```    


To run the PD regulator with custom action bounds (default is `[-20, 20]`):


```shell
python run.py policy=pd policy.action_min=-3 policy.action_max=3 --interactive --fps=10
```  

### Step 3: Run Energy-Based Controller

To run the energy-based controller:

```shell
python run.py policy=energy_based --interactive --fps=10
```  

For custom action bounds:
```shell
python run.py policy=energy_based policy.action_min=-4 policy.action_max=4 --interactive --fps=10
```  

### Step 3: Run Adaptive Controller

To run the adaptive controller:

```shell
python run.py policy=adaptive --interactive --fps=10
```  

For custom action bounds:
```shell
python run.py policy=adaptive policy.action_min=-4 policy.action_max=4 --interactive --fps=10
```  



> **Note:**
>
> For the `--fps` parameter, you can select any suitable value to ensure a smooth experience (e.g., `--fps=2`, `--fps=10`, `--fps=20`, etc.).

## Repo structure

- [`run.py`](./run.py): The main executable script.
- [`src/`](./src/): Contains the source code of the repo.
    - [`policy.py`](./src/policy.py): Implements the PD and energy-based policies.
    - [`system.py`](./src/system.py): Implements the inverted pendulum system with friction.
- [`presets/`](./presets/): Houses configuration files.
    - [`common/`](./presets/common): General configurations.
        - [`common.yaml`](./presets/common/common.yaml): Settings for common variables (like sampling time)
    - [`policy/`](./presets/policy/): Policy-specific configurations.
        - [`adaptive.yaml`](./presets/policy/adaptive.yaml): Settings for the adaptive controller.
        - [`energy_based.yaml`](./presets/policy/energy_based.yaml): Settings for the energy-based policy.
        - [`pd.yaml`](./presets/policy/pd.yaml): Settings for the Proportional-Derivative (PD) regulator.
    - [`scenario/`](./presets/scenario/): Scenario configurations.
        - [`scenario.yaml`](./presets/scenario/scenario.yaml): Main orchestrator settings.
    - [`simulator`](./presets/simulator/): Simulator configurations.
        - [`casadi.yaml`](./presets/simulator/casadi.yaml): Configurations for the [CasADi](https://web.casadi.org/) [RK](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods) simulator.


