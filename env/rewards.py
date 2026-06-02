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


    def compute_reward(self, *args, **kwargs):
        """
        Dual-compatible reward hook. 
        Handles old environment keyword calls and reroutes them to our new 
        Risk-Aware + Anti-Spinning optimization layout.
        """
        # If the environment calls it using the old keyword arguments, bundle them into an 'info' dict
        if 'progress' in kwargs or len(args) > 0:
            # Fallbacks to extract values based on your old signature
            progress = kwargs.get('progress', args[0] if len(args) > 0 else 0.0)
            current_action = kwargs.get('current_action', args[1] if len(args) > 1 else 0)
            lidar_readings = kwargs.get('lidar_readings', args[3] if len(args) > 3 else None)
            collision = kwargs.get('collision', args[6] if len(args) > 6 else False)
            reached_goal = kwargs.get('reached_goal', args[7] if len(args) > 7 else False)
            
            # Reconstruct the expected 'info' packet dynamically
            info = {
                'progress': progress,
                'raw_lidar': lidar_readings,
                'collision': collision,
                'reached_goal': reached_goal
            }
            action_idx = current_action
        else:
            # If called directly from the new training loop using (info, action_idx)
            info = args[0] if len(args) > 0 else kwargs.get('info', {})
            action_idx = args[1] if len(args) > 1 else kwargs.get('action_idx', 0)

        # ---------------------------------------------------------
        # NEW CORE LOGIC: Risk-Aware + Anti-Spinning
        # ---------------------------------------------------------
        reward = 0.0
        
        # 1. Standard Performance Metrics
        reward += 5.0 * info.get('progress', 0.0)
        reward -= 0.005  # Decisiveness Time Penalty
        
        # 2. Angular Velocity Dampener (Fixes the Spinning)
        # Action space mapping: 3 = Turn Left, 4 = Turn Right
        if action_idx in [3, 4]:
            reward -= 0.05  # Slight energy penalty to prevent continuous spinning
            
        # 3. Risk-Aware LiDAR Safety Envelope (Fixes the Crashing)
        lidar_readings = info.get('raw_lidar', None) 
        if lidar_readings is not None and len(lidar_readings) > 0:
            min_lidar = np.min(lidar_readings)
            
            # Danger-zone breach threshold set at 0.25 meters
            if min_lidar < 0.25:
                risk_penalty = 2.0 * (0.25 - min_lidar)
                reward -= risk_penalty
                
        # 4. Terminal State Modifiers
        if info.get('reached_goal', False):
            reward += 1000.0
        elif info.get('collision', False):
            reward -= 1000.0
            
        return float(reward)