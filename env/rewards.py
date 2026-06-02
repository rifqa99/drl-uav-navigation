# import numpy as np

# class UAVRewardShaping:
#     def __init__(self, world_size):
#         self.world_size = world_size


#     def compute_reward(
#         self,
#         progress,
#         current_action,
#         prev_action,
#         lidar_readings,
#         distance,
#         speed,
#         max_speed,
#         collision,
#         reached_goal,
#     ):

#         # Terminal conditions
#         if collision:
#             return -300.0

#         if reached_goal:
#             return 1000.0

#         # Progress reward
#         reward = 5.0 * progress

#         # Small time penalty
#         reward -= 0.005

#         # Energy penalty
#         action_energy = {
#             0: 0.0,   # Hover
#             1: 1.0,   # Forward thrust
#             2: 1.0,   # Reverse thrust
#             3: 0.3,   # Clockwise turn
#             4: 0.3,   # Counter-clockwise turn
#         }

#         reward -= 0.01 * action_energy.get(current_action, 0.0)

#         # Obstacle proximity penalty
#         min_lidar = float(np.min(lidar_readings))

#         if min_lidar < 0.25:
#             reward -= 2.0 * (0.25 - min_lidar)

#         return float(reward)
################## above ıs the old reward shaping code, which is now replaced by the new one below ##################
# The new reward shaping function is designed to fix the issue of spinning ın addition to risk-awareness to decrease the chances of collision. The reward is calculated based on the following components:
import numpy as np


class UAVRewardShaping:
    def __init__(self, world_size=10.0, safety_distance=0.25):
        self.world_size = world_size
        self.safety_distance = safety_distance

    def compute_reward(
        self,
        progress,
        action,
        lidar_readings,
        collision,
        reached_goal
    ):

        if collision:
            return -1000.0

        if reached_goal:
            return 1000.0

        reward = 0.0

        # Goal progress
        reward += 5.0 * float(progress)

        # Time penalty
        reward -= 0.005

        # Anti-spinning
        if action in [3, 4]:
            reward -= 0.05

        # Safety envelope in REAL METERS
        if lidar_readings is not None and len(lidar_readings) > 0:

            min_lidar_m = (
                float(np.min(lidar_readings))
                * self.world_size
            )
            if min_lidar_m < 1.0:
                reward -= 0.5

            if min_lidar_m < 0.5:
                reward -= 2.0

            if min_lidar_m < 0.25:
                reward -= 10.0

        return float(reward)