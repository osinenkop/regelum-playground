## Overview 

This repository showcases the operation of PD regulator and energy-based controller on an [inverted pendulum system](https://regelum.aidynamic.io/systems/inv_pendulum/), utilizing the [regelum-control](https://regelum.aidynamic.io/systems/inv_pendulum/) Python package. This package is designed for researchers and practitioners in reinforcement learning and optimal control.

## Getting started


### Step 1: install requirements

Before installing the [`requirements.txt`](./requirements.txt), it is recommended to create a virtual environment for your project.

```shell
pip install -r requirements.txt
```

### Step 2: Run PD Regulator

Execute the PD regulator using the following command in your terminal:

```shell
python run.py policy=pd --single-thread --interative --fps=10
```    


To run the PD regulator with custom action bounds (default is `[-20, 20]`):


```shell
python run.py policy=pd policy.action_min=-1 policy.action_max=1 --single-thread --interative --fps=10
```  

### Step 3: Run Energy-Based Controller

To run the energy-based controller:

```shell
python run.py policy=energy_based --single-thread --interative --fps=10
```  

For custom action bounds:
```shell
python run.py policy=energy_based policy.action_min=-1 policy.action_max=1 --single-thread --interative --fps=10
```  


> **Note:**
>
> For the `--fps` parameter, you can select any suitable value to ensure a smooth experience (e.g., `--fps=2`, `--fps=10`, `--fps=20`, etc.).

## Repo structure

- [`run.py`](./run.py): The main executable script.
- [`src/`](./src/): Contains the source code of the repo.
    - [`policy.py`](./src/policy.py): Implements the PD and energy-based policies.
- [`presets/`](./presets/): Houses configuration files.
    - [`common/`](./presets/common): General configurations.
        - [`common.yaml`](./presets/common/common.yaml): Settings for common variables (like sampling time)
    - [`policy/`](./presets/policy/): Policy-specific configurations.
        - [`energy_based.yaml`](./presets/policy/energy_based.yaml): Settings for the energy-based policy.
        - [`pd.yaml`](./presets/policy/pd.yaml): Settings for the Proportional-Derivative (PD) regulator.
    - [`scenario/`](./presets/scenario/): Scenario configurations.
        - [`scenario.yaml`](./presets/scenario/scenario.yaml): Main orchestrator settings.
    - [`simulator`](./presets/simulator/): Simulator configurations.
        - [`casadi.yaml`](./presets/simulator/casadi.yaml): Configurations for the [CasADi](https://web.casadi.org/) [RK](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods) simulator.


