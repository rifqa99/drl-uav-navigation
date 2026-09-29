# DDQN-Based UAV Navigation with LiDAR

## Overview

This project implements a Dueling Double Deep Q-Network (DDQN) agent for autonomous UAV navigation in obstacle-filled environments using LiDAR observations.

The UAV operates in a 2D environment with realistic physics-based dynamics, including:

* Linear and rotational motion
* Wind disturbances
* Dynamic obstacles
* 64-beam LiDAR sensing
* Curriculum learning with increasing obstacle density

The agent receives stacked observations consisting of LiDAR measurements and UAV state information and learns collision-free navigation toward a target location.

---

## Features

* Dueling Double DQN architecture
* Physics-based UAV dynamics
* Dynamic obstacle environments
* 64-beam LiDAR perception
* Curriculum learning
* Evaluation framework with unseen test seeds
* Ground Control Station (GCS) visualizer

---

## Project Structure

```text
ddqn-uav-navigation/
│
├── agents/
│   ├── dqn_agent.py
│   ├── dueling_dqn.py
│   └── replay_buffer.py
│
├── env/
│   ├── dynamics.py
│   ├── rewards_standard.py
│   ├── uav_env.py
│   └── uav_env_dynamic.py
│
├── outputs/
│   ├── checkpoints/
│   └── test_results/
│
├── train_dqn_static.py
├── train_dqn_dynamic.py
├── evaluate_models.py
├── gcs_station.py
├── requirements.txt
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Training

### Static Environment

```bash
python train_dqn_static.py
```

### Dynamic Environment

```bash
python train_dqn_dynamic.py
```

---

## Evaluation

Evaluate the trained model on 100 unseen random seeds:

```bash
python evaluate_models.py
```

Results are saved to:

```text
outputs/test_results/
```

---

## Visualization

Launch the Ground Control Station interface:

```bash
python gcs_station.py
```

The visualizer allows:

* Dynamic and static environment selection
* Obstacle count adjustment
* Random map generation
* Real-time trajectory visualization
* Live telemetry monitoring

---

## Observation Space

The observation vector consists of:

* 64 LiDAR beam distances
* UAV velocity (vx, vy)
* UAV heading angle
* Angular velocity
* Relative target direction

Three consecutive observations are stacked before being fed to the DDQN agent.

---

## Action Space

| Action | Description                |
| ------ | -------------------------- |
| 0      | Hover                      |
| 1      | Forward Thrust             |
| 2      | Reverse Thrust             |
| 3      | Clockwise Rotation         |
| 4      | Counter-Clockwise Rotation |

---

## Reward Function

The reward function includes:

* Positive reward for progress toward the goal
* Goal completion reward
* Collision penalty
* Small time penalty
* Rotation penalty
* Angular velocity penalty

---

## Author

Refga Mobarak Awadelkarim Mohamed

Master's Degree Project – Deep Reinforcement Learning for UAV Navigation
