import numpy as np


class UAVRewardShaping:
    def __init__(self, world_size):
        self.world_size = world_size

    def compute_reward(
        self,
        progress,
        current_action,
        prev_action,
        lidar_readings,
        distance,
        speed,
        max_speed,
        collision,
        reached_goal,
    ):

        # Terminal conditions
        if collision:
            return -300.0

        if reached_goal:
            return 1000.0

        # Progress reward
        reward = 5.0 * progress

        # Small time penalty
        reward -= 0.005

        # Energy penalty
        action_energy = {
            0: 0.0,   # Hover
            1: 1.0,   # Forward thrust
            2: 1.0,   # Reverse thrust
            3: 0.3,   # Clockwise turn
            4: 0.3,   # Counter-clockwise turn
        }

        reward -= 0.01 * action_energy.get(current_action, 0.0)

        # Obstacle proximity penalty
        min_lidar = float(np.min(lidar_readings))

        if min_lidar < 0.25:
            reward -= 2.0 * (0.25 - min_lidar)

        return float(reward)
