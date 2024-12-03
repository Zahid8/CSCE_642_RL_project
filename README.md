# CSCE_642_RL_project

This project implements various reinforcement learning algorithms to train agents for performing soccer free kicks in a simulated environment. The algorithms include DDPG, SAC, A2C, A3C, and PPO. The project is structured into two main directories: `ddpg/` and `ppo/`, each containing implementations related to their respective algorithms.

## Table of Contents

- [Project Structure](#project-structure)
- [Introduction](#introduction)
- [DDPG Directory](#ddpg-directory)
  - [Agents and Models](#agents-and-models)
  - [Training Scripts](#training-scripts)
  - [Environment](#environment)
  - [Utilities](#utilities)
  - [Training Results](#training-results)
- [PPO Directory](#ppo-directory)
  - [Agent and Model](#agent-and-model)
  - [Training Script](#training-script)
  - [Environment](#environment-1)
  - [Training Results](#training-results-1)
- [Getting Started](#getting-started)
- [Dependencies](#dependencies)
- [Usage](#usage)
  - [Training the DDPG Agent](#training-the-ddpg-agent)
  - [Training the SAC Agent](#training-the-sac-agent)
  - [Training the A2C Agent](#training-the-a2c-agent)
  - [Training the A3C Agent](#training-the-a3c-agent)
  - [Training the PPO Agent](#training-the-ppo-agent)
- [Results](#results)
- [License](#license)


## Introduction

The goal of this project is to compare the performance of different reinforcement learning algorithms on the task of executing free kicks in a soccer simulation. By implementing and training agents using various algorithms, we aim to analyze their effectiveness in learning complex control policies.

## DDPG Directory

The `ddpg/` directory contains implementations of several actor-critic algorithms, including DDPG, SAC, A2C, and A3C.

### Agents and Models

- **ddpg_agent.py**: Implements the DDPG (Deep Deterministic Policy Gradient) agent. It defines the agent's behavior, including action selection using a deterministic policy and updating the actor and critic networks based on the DDPG algorithm.

- **ddpg_model.py**: Defines the neural network architectures for the DDPG agent's actor and critic networks. The actor network outputs continuous actions, and the critic network estimates the Q-values for state-action pairs.

- **sac_agent.py**: Implements the SAC (Soft Actor-Critic) agent. It uses a stochastic policy and optimizes a maximum entropy objective to encourage exploration.

- **sac_model.py**: Defines the neural network architectures for the SAC agent's policy and two Q-value networks to mitigate overestimation bias.

- **a2c_agent.py**: Implements the A2C (Advantage Actor-Critic) agent, which synchronously updates policy and value networks based on the advantage function.

- **a2c_model.py**: Defines the neural network architectures for the A2C agent's policy and value networks.

- **a3c_agent.py**: Implements the A3C (Asynchronous Advantage Actor-Critic) agent, running multiple instances of the environment in parallel threads for asynchronous updates.

- **a3c_model.py**: Defines the shared neural network model used by the A3C agent for both policy and value estimation.

### Training Scripts

- **train_ddpg.py**: Script to train the DDPG agent in the `SoccerFreeKickEnv`. Initializes the environment and agent, runs the training loop, and logs performance metrics.

- **train_sac.py**: Script to train the SAC agent, handling its specific training parameters and logging.

- **train_a2c.py**: Script for training the A2C agent, managing synchronous updates and logging progress.

- **train_a3c.py**: Script to train the A3C agent, creating multiple threads for asynchronous training.

### Environment

- **SoccerFreeKickEnv.py**: Defines the custom OpenAI Gym environment for simulating soccer free kicks, including state and action spaces, reward functions, and dynamics.

### Utilities

- **replay_buffer.py**: Implements the replay buffer for experience replay, essential for off-policy algorithms like DDPG and SAC.

### Training Results

- **ddpg_training_results.csv**: Contains logged training results for the DDPG agent, including rewards and losses.

- **sac_training_results.csv**: Contains training results for the SAC agent.

- **a2c_training_results.csv**: Contains training logs for the A2C agent.

- **ppo_training_results.csv**: (Note: This file appears here but is related to PPO and may be misplaced.)

## PPO Directory

The `ppo/` directory contains the implementation of the PPO algorithm.

### Agent and Model

- **ppo_agent.py**: Implements the PPO (Proximal Policy Optimization) agent, using a clipped surrogate objective for stable policy updates.

- **model.py**: Defines the neural network architecture used by the PPO agent for policy and value estimation.

### Training Script

- **train_ppo.py**: Script to train the PPO agent in the `SoccerFreeKickEnv`, handling trajectory collection and policy updates.

### Environment

- **SoccerFreeKickEnv.py**: Defines the soccer free kick environment used for training the PPO agent.

### Training Results

- **ppo_training_results.csv**: Contains logged training results for the PPO agent.

## Getting Started

Clone the repository and install the required dependencies to get started.

```bash
git clone https://github.com/your_username/free_kick.git
cd free_kick
pip install -r requirements.txt
```

## Execution

```cd``` into either ddpg or ppo folder and run the train script : ```python3 train_ppo.py```, ```python3 train_ddpg.py```, ```python3 train_sac.py```, ```python3 train_a2c.py```, ```python3 train_a3c.py```.


