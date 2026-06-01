import numpy as np


class UAVRewardShaping:
    def __init__(self, world_size):
        self.world_size = world_size

    def compute_reward(self, progress, current_action, prev_action, lidar_readings,
                       distance, speed, max_speed, collision, reached_goal):
        """
        Calculates the finalized step-wise potential-based reward.
        Optimized for path progress efficiency and safe, low-velocity landing.
        """
        reward = 0.0

        # 1. Potential-Based Progress Reward (Primary Motivation)
        # Scaled to 15.0 so terminal landing bonus remains the dominant driver
        reward += 15.0 * progress

        # 2. Time-Step Decay Penalty (Anti-Loitering clock)
        reward -= 0.05

        # 3. Energy Component Penalty (Action-Aware Expense)
        energy_use = 1.0 if current_action in [1, 2] else (
            0.2 if current_action in [3, 4] else 0.0)
        reward -= 0.05 * energy_use

        # 4. Flight Smoothness Penalty (Jitter Mitigation)
        smoothness_penalty = 1.0 if current_action != prev_action else 0.0
        reward -= 0.15 * smoothness_penalty

        # 5. Proximity Safety Soft Buffer (Invisible barrier force-field)
        min_lidar = np.min(lidar_readings)
        if min_lidar < 0.25:
            reward -= 1.0 * (1.0 - min_lidar)

        # 6. Hard Boundary Terminal Conditions
        if collision:
            reward -= 100.0

        if reached_goal:
            # FIXED: Variables are now passed explicitly into the scope
            speed_penalty = 50.0 * (speed / max_speed)
            reward += (500.0 - speed_penalty)

        return reward
