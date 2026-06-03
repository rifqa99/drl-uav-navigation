import numpy as np

class UAVRewardShaping:
    def __init__(self, world_size=10.0):
        self.world_size = world_size

    def compute_reward(
        self,
        progress,
        action,
        lidar_readings,
        collision,
        reached_goal,
        speed=0.0,
        omega=0.0
    ):
        if collision:
            return -300.0

        if reached_goal:
            return 1000.0

        reward = 0.0

        reward += 5.0 * float(progress)
        reward -= 0.005

        action_energy = {
            0: 0.0,
            1: 1.0,
            2: 1.0,
            3: 0.3,
            4: 0.3,
        }

        reward -= 0.01 * action_energy.get(action, 0.0)

        min_lidar_m = float(np.min(lidar_readings)) * self.world_size

        if min_lidar_m < 0.25:
            reward -= 2.0 * (0.25 - min_lidar_m)

        return float(reward)