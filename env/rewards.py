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
    def __init__(self, world_size):
        self.world_size = world_size

    def compute_reward(self, *args, **kwargs):
        """
        Robust, type-checked reward hook.
        Properly differentiates between old environment positional steps
        and new explicit training loop dictionary calls.
        """
        info = {}
        action_idx = 0

        # Check if the first positional argument is an explicit dictionary (Our New Loop)
        if len(args) > 0 and isinstance(args[0], dict):
            info = args[0]
            action_idx = kwargs.get('action_idx', args[1] if len(args) > 1 else 0)
            
        # Check if called via explicit old keywords (Our Old Env)
        elif 'progress' in kwargs:
            info = {
                'progress': kwargs.get('progress', 0.0),
                'raw_lidar': kwargs.get('lidar_readings', None),
                'collision': kwargs.get('collision', False),
                'reached_goal': kwargs.get('reached_goal', False)
            }
            action_idx = kwargs.get('current_action', 0)
            
        # Fallback: Environment is passing old raw positional variables (progress, current_action, ...)
        elif len(args) > 1:
            info = {
                'progress': args[0],
                'raw_lidar': args[3] if len(args) > 3 else None,
                'collision': args[6] if len(args) > 6 else False,
                'reached_goal': args[7] if len(args) > 7 else False
            }
            action_idx = args[1]

        # ---------------------------------------------------------
        # NEW CORE LOGIC: Risk-Aware + Anti-Spinning
        # ---------------------------------------------------------
        reward = 0.0
        
        # 1. Standard Performance Metrics
        reward += 5.0 * float(info.get('progress', 0.0))
        reward -= 0.005  # Decisiveness Time Penalty
        
        # 2. Angular Velocity Dampener (Anti-Spinning Fix)
        if action_idx in [3, 4]:
            reward -= 0.05  # Slight energy penalty for turning on the spot
            # ---------------------------------------------------------
        # 3. Risk-Aware LiDAR Safety Envelope (Fixes the Crashing)
        # ---------------------------------------------------------
        lidar_readings = info.get('raw_lidar', None) 
        if lidar_readings is not None and len(lidar_readings) > 0:
            # FORCE ABSOLUTE VALUES to eliminate negative coordinate noise
            clean_distances = np.abs(lidar_readings)
            min_lidar = np.min(clean_distances)
            
            # Danger-zone boundary breach threshold set at 0.25 meters
            if min_lidar < 0.25:
                risk_penalty = 2.0 * (0.25 - min_lidar)
                reward -= risk_penalty
                
        # 4. Terminal State Modifiers
        if info.get('reached_goal', False):
            reward += 1000.0
        elif info.get('collision', False):
            reward -= 1000.0
            
        return float(reward)