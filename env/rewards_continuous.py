import numpy as np


class UAVRewardShapingContinuous:
    def __init__(self, world_size=10.0):
        self.world_size = world_size

    def compute_reward(self, progress, current_action, prev_action, lidar_readings,
                       distance, speed, max_speed, collision, reached_goal):

        # 1. Terminal Penalties & Rewards (Identical to DQN values)
        if collision:
            return -100.0

        if reached_goal:
            # Safe low-velocity arrival landing reward tracking
            speed_penalty = 50.0 * (speed / max_speed)
            return 500.0 - speed_penalty

        # 2. Potential-Based Progress Reward
        reward = progress * 100.0

        # 3. Time-Step Efficiency Penalty
        reward -= 0.05

        # 4. Continuous Actuation & Energy Scaling
        # current_action is [thrust, torque]
        thrust_cmd = float(current_action[0])
        torque_cmd = float(current_action[1])

        # Energy scales naturally with throttle intensity
        energy_use = thrust_cmd * 1.0
        reward -= (0.05 * energy_use)

        # Smoothness penalty based on vector distance delta
        smoothness_penalty = float(
            np.linalg.norm(current_action - prev_action))
        reward -= (0.02 * smoothness_penalty)

        # 5. Proximity Avoidance Safety Margin
        min_lidar = float(np.min(lidar_readings))
        if min_lidar < 0.25:  # Critical danger zone radius threshold
            reward -= 0.2 * (1.0 - min_lidar)

        return float(reward)
