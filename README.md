<!-- TOC start (generated with https://github.com/derlin/bitdowntoc) -->

- [About ](#about)
- [Getting started](#getting-started)
   * [Installation requirements](#installation-requirements)
- [Ready examples](#ready-examples)
   * [Proportional-derivative (PD) controller for Pendulum](#proportional-derivative-pd-controller-for-inverted-pendulum)
   * [Energy-based controller for Pendulum](#energy-based-controller-for-inverted-pendulum)
   * [Energy-based controller for Pendulum with joint friction](#energy-based-controller-for-inverted-pendulum-with-joint-friction)
   * [Energy-based controller with friction compenstation for Pendulum with joint friction](#energy-based-controller-with-friction-compenstation-for-inverted-pendulum-with-joint-friction)
   * [Adaptive energ-based controller for Pendulum with joint friction](#adaptive-energ-based-controller-for-inverted-pendulum-with-joint-friction)
   * [Backstepping controller for Pendulum with motor dynamics](#backstepping-controller-for-inverted-pendulum-with-motor-dynamics)
   * [PD controller for Pendulum with motor dynamics](#pd-controller-for-inverted-pendulum-with-motor-dynamics)
   * [Lyapunov-based controller for kinematic three-wheeled robot](#lyapunov-based-controller-for-kinematic-three-wheeled-robot)
   * [Backstepping controller for dynamic three-wheeled robot](#backstepping-controller-for-dynamic-three-wheeled-robot)
   * [Model-predictive controller for three-wheeled robot](#model-predictive-controller-for-three-wheeled-robot)
      + [On a plane with quadratic cost](#on-a-plane-with-quadratic-cost)
      + [On a plane with a Guassian spot of high cost](#on-a-plane-with-a-guassian-spot-of-high-cost)
   * [Proximal Policy Optimizaion on Pendulum](#proximal-policy-optimizaion-on-inverted-pendulum)
   * [Soft Actor-Critic (SAC) on Pendulum with Gym-like Observation](#soft-actor-critic-sac-on-inverted-pendulum-with-gym-like-observation)
   * [Twin Delayed Deep Deterministic Policy Gradient (TD3) on Pendulum with Gym-like Observation](#twin-delayed-deep-deterministic-policy-gradient-td3-on-inverted-pendulum-with-gym-like-observation)
   * [CALF algorithm](#calf-algorithm)

<!-- TOC end -->

<!-- TOC --><a name="about"></a>
## About 

This is a playground based on [regelum-control](https://regelum.aidynamic.group), a framework for control and reinforcement learning.
It showcases various dynamical systems and controllers (also called policies).

<!-- TOC --><a name="getting-started"></a>
## Getting started

<!-- TOC --><a name="installation-requirements"></a>
### Installation requirements

If you are working in Windows, it is recommended to use WSL and, possibly, a display server like Xming to properly output graphics from WSL.
Before installing the [`requirements.txt`](./requirements.txt), it is recommended to create a virtual environment for your project, say, `pyenv` or `virtualenv`. The instructions on these are standard and may be found on the web.

First of all, make sure you are working with Python<=3.11 to avoid packge building problems.
Under Ubuntu 24, do this, for instance:

```shell
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.11
```

Install `virtualenv`:

```shell
sudo apt-get install python3.11-venv
```

Create a virtual environment (here called `rgenv`):

```shell
python3.11 -m venv rgenv
```

Activate it:

```shell
source rgenv/bin/activate
```

When the virtual environment is created, install necessary packages:

```shell
pip install -r requirements.txt
```

<!-- TOC --><a name="ready-examples"></a>
## Ready examples

Below are examples with respective terminal run commands. 

<!-- TOC --><a name="proportional-derivative-pd-controller-for-inverted-pendulum"></a>
### Proportional-derivative (PD) controller for Pendulum

```shell
python run.py policy=pd system=pendulum_loose_bounds --interactive --fps=10
```    

Observe it doesn't work with the default action bounds.
Further, we will see how the energy-based controller does the job.

To run the PD controller with custom action bounds:

```shell
python run.py policy=pd system=pendulum_loose_bounds policy.action_min=-2 policy.action_max=2 --interactive --fps=10
```  

Making the bounds sufficiently large will help the PD controller to upswing and upright hold the pendulum.

<!-- TOC --><a name="energy-based-controller-for-inverted-pendulum"></a>
### Energy-based controller for Pendulum

```shell
python run.py policy=energy_based system=pendulum_loose_bounds --interactive --fps=10
```  

<!-- TOC --><a name="energy-based-controller-for-inverted-pendulum-with-joint-friction"></a>
### Energy-based controller for Pendulum with joint friction

```shell
python run.py policy=energy_based system=pendulum_with_friction --interactive --fps=10
```  

Observe it doesn't work due to ignorance of friction.

<!-- TOC --><a name="energy-based-controller-with-friction-compenstation-for-inverted-pendulum-with-joint-friction"></a>
### Energy-based controller with friction compenstation for Pendulum with joint friction

```shell
python run.py policy=energy_based_friction_compensation system=pendulum_with_friction --interactive --fps=10
```

This works better than the last one.

<!-- TOC --><a name="adaptive-energ-based-controller-for-inverted-pendulum-with-joint-friction"></a>
### Adaptive energ-based controller for Pendulum with joint friction

```shell
python run.py policy=energy_based_friction_adaptive system=pendulum_with_friction --interactive --fps=10
```  

This should also work.

<!-- TOC --><a name="backstepping-controller-for-inverted-pendulum-with-motor-dynamics"></a>
### Backstepping controller for Pendulum with motor dynamics

This showcases the use of backstepping.

```shell
python run.py policy=backstepping system=pendulum_with_motor --interactive --fps=10 
``` 

<!-- TOC --><a name="pd-controller-for-inverted-pendulum-with-motor-dynamics"></a>
### PD controller for Pendulum with motor dynamics

```shell
python run.py policy=motor_pd system=pendulum_with_motor --interactive --fps=10 
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

<!-- TOC --><a name="proximal-policy-optimizaion-on-inverted-pendulum"></a>
### Proximal Policy Optimizaion on Pendulum

```
python run.py \
    scenario=ppo_scenario scenario.discount_factor=0.7 \
    system=pendulum_loose_bounds \
    common.time_final=10 \
    scenario.N_episodes=2 \
    --interactive \
    --fps=10 \
    scenario.policy_model.std=0.01 \
    scenario.policy_model.normalize_output_coef=0.0001
```


<!-- TOC --><a name="soft-actor-critic-sac-on-inverted-pendulum-with-gym-like-observation"></a>
### Soft Actor-Critic (SAC) on Pendulum with Gym-like Observation

```bash
python run.py \
    scenario=sac \
    system=pendulum_with_gym_observation \
    simulator=casadi_random_state_init \
    scenario.autotune=False \
    scenario.policy_lr=0.00079 \
    scenario.q_lr=0.00025 \
    scenario.alpha=0.0085 \
    +seed=4 \
    --interactive \
    --fps=10
```
> **Note:**
>
> You can set any seed you want by modifying the `+seed` parameter. For example, you can use `+seed=42` or any other integer value. Different seeds may lead to different training outcomes due to the stochastic nature of the algorithm.

<!-- TOC --><a name="twin-delayed-deep-deterministic-policy-gradient-td3-on-inverted-pendulum-with-gym-like-observation"></a>
### Twin Delayed Deep Deterministic Policy Gradient (TD3) on Pendulum with Gym-like Observation

To run the TD3 algorithm on the Pendulum system with a gym-like observation space, use the following command:

```bash
python run.py \
    scenario=td3 \
    system=pendulum_with_gym_observation \
    simulator=casadi_random_state_init  \
    +seed=4  \
    --interactive \
    --fps=10
```

> **Note:**
>
> You can set any seed you want by modifying the `+seed` parameter. For example, you can use `+seed=42` or any other integer value. Different seeds may lead to different training outcomes due to the stochastic nature of the algorithm.


> **Note:**
>
> The implementations of SAC and TD3 in regelum-playground are adapted from the [CleanRL](https://docs.cleanrl.dev). CleanRL provides single-file implementations of reinforcement learning algorithms, which have been modified to work within the regelum framework. For detailed information on the original implementations, please refer to the [CleanRL documentation](https://docs.cleanrl.dev)
>
> The adaptations made for regelum-playground include integration with regelum's Simulator and RunningObjective classes, use of regelum's callback system for logging, and modifications to work with regelum's configuration system. While the core algorithm logic remains similar to CleanRL's implementation, these changes allow for seamless integration with the regelum ecosystem. For detailed information on the implementation of SAC and TD3 algorithms in regelum-playground, please refer to the comprehensive tutorial in [notes/sac_td3_regelum_tutorial.md](./notes/sac_td3_regelum_tutorial.md). This tutorial provides in-depth explanations of the algorithm structures, key features, and integration with the regelum framework.

> **Note:**
>
> For the `--fps` parameter, you can select any suitable value to ensure a smooth experience (e.g., `--fps=2`, `--fps=10`, `--fps=20`, etc.).

Here is the new section for the README:



<!-- TOC --><a name="calf-algorithm"></a>
### CALF algorithm

To run the CALF algorithm on the pendulum system, use the following command:

```shell
python run.py scenario=calf system=pendulum --interactive --fps=10 common.time_final=10
```

The CALF algorithm also works with other systems such as `lunar_lander`, `3wrobot_kin_rg` and `cartpole_pg`. Note that `3wrobot_kin_rg` is the system that is natively imported from regelum without overriding the system parameters. For these systems, use the following command (please note that it is not necessary to add `common.time_final=...` in the command, as everything is already preconfigured):

```shell
python run.py scenario=calf system=<system_name> --interactive --fps=10
```

Replace `<system_name>` with the desired system, e.g., `lunar_lander`, `3wrobot_kin_rg` or `cartpole_pg`.


