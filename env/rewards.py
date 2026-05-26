import numpy as np


class UAVRewardShaping:
    def __init__(self, world_size):
        self.world_size = world_size

    def compute_reward(self, progress, current_action, prev_action, lidar_readings, distance, collision, reached_goal):
        """
        Calculates the finalized step-wise potential-based reward.
        Eliminates loitering behavior by optimizing path progress efficiency.
        """
        reward = 0.0

        # 1. Potential-Based Progress Reward (Primary Motivation)
        reward += 20.0 * progress

        # 2. Time-Step Decay Penalty (Anti-Loitering)
        reward -= 0.05

        # 3. Energy Component Penalty (Action-Aware Expense)
        # 1 and 2 represent Forward/Reverse Thrust; 3 and 4 represent Clockwise/Counter-Clockwise Torque
        energy_use = 1.0 if current_action in [1, 2] else (
            0.2 if current_action in [3, 4] else 0.0)
        reward -= 0.05 * energy_use

        # 4. Flight Smoothness Penalty (Jitter Mitigation)
        smoothness_penalty = 1.0 if current_action != prev_action else 0.0
        reward -= 0.15 * smoothness_penalty

        # 5. Proximity Safety Soft Buffer
        min_lidar = np.min(lidar_readings)
        if min_lidar < 0.25:
            reward -= 1.0 * (1.0 - min_lidar)

        # 6. Hard Boundary Terminal Conditions
        if collision:
            reward -= 100.0

        if reached_goal:
            reward += 500.0

        return reward
