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
        # 1. Terminal objectives
        if collision:
            return -300.0

        if reached_goal:
            return 1000.0

        # 2. Progress toward goal
        reward = 5.0 * progress

        # 3. Small time penalty
        reward -= 0.005

        # 4. Energy penalty for continuous control
        current_action = np.asarray(current_action, dtype=np.float32)
        action_energy = float(np.sum(current_action ** 2))
        reward -= 0.01 * action_energy

        # 5. Obstacle proximity penalty
        min_lidar = float(np.min(lidar_readings))

        if min_lidar < 0.25:
            reward -= 2.0 * (0.25 - min_lidar)

        return float(reward)
