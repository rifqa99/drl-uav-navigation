import numpy as np


class UAVRewardShapingContinuous:
    def __init__(self, world_size=10.0):
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
        if reached_goal:
            return 1000.0

        if collision:
            return -150.0

        reward = 10.0 * progress

        reward -= 0.01

        current_action = np.asarray(current_action, dtype=np.float32)
        action_energy = float(np.sum(current_action ** 2))
        reward -= 0.005 * action_energy

        min_lidar = float(np.min(lidar_readings))

        if min_lidar < 0.30:
            reward -= 1.0 * (0.30 - min_lidar)

        if distance < 1.0:
            reward += 1.0 * (1.0 - distance)

        return float(reward)
