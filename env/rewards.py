import numpy as np


 class UAVRewardShaping:
    def __init__(self, world_size):
        self.world_size = world_size

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


    def compute_reward(self, info, action_idx):
        """
        Advanced Multi-Objective Reward Function
        Incorporates:
        1. Distance Progress to Target
        2. Decisiveness Time Penalty
        3. Angular Velocity Dampener (Anti-Spinning Fix)
        4. Risk-Aware LiDAR Safety Envelope (Anti-Collision Fix)
        """
        reward = 0.0
        
        # ---------------------------------------------------------
        # 1. Standard Performance Metrics
        # ---------------------------------------------------------
        # Distance progress step
        reward += 5.0 * info.get('progress', 0.0)
        
        # Decisiveness Time Penalty (Anti-Stalling)
        reward -= 0.005
        
        # ---------------------------------------------------------
        # 2. Angular Velocity Dampener (Fixes the Spinning)
        # ---------------------------------------------------------
        # Assuming action space mapping: 3 = Turn Left, 4 = Turn Right
        # Adjust action indices if your mapping differs!
        if action_idx in [3, 4]:
            reward -= 0.05  # Slight energy penalty for spinning on the spot
            
        # ---------------------------------------------------------
        # 3. Risk-Aware LiDAR Safety Envelope (Fixes the Crashing)
        # ---------------------------------------------------------
        # Extract structural raw array readings from your lidar observation state
        # Assumes self.env.lidar_sensor_data or passed via observation matrix
        lidar_readings = info.get('raw_lidar', None) 
        if lidar_readings is not None and len(lidar_readings) > 0:
            min_lidar = np.min(lidar_readings)
            
            # Danger-zone breach threshold set at 0.25 meters spatial boundary
            if min_lidar < 0.25:
                # Linear scaling penalty: gets exponentially harsher the closer it gets
                risk_penalty = 2.0 * (0.25 - min_lidar)
                reward -= risk_penalty
                
        # ---------------------------------------------------------
        # 4. Terminal State Modifiers
        # ---------------------------------------------------------
        if info.get('reached_goal', False):
            reward += 1000.0
        elif info.get('collision', False):
            reward -= 1000.0
            
        return float(reward)