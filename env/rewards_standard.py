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
        
        # 1. Catastrophic Terminal Triggers
        if collision:
            return -1000.0

        if reached_goal:
            return 1000.0
        
        reward = 0.0

        # 2. Potential-Based Progress & Decisiveness Constraints
        reward += 5.0 * float(progress)
        reward -= 0.005  # Standard frame time penalty

        # 3. Dual Rotational Smoothness Constraints (Anti-Spinning Fix)
        if action in [3, 4]:
            reward -= 0.20  # Actuator selection penalty
            

        # 4. Proximity Risk Envelope
        if lidar_readings is not None and len(lidar_readings) > 0:
            min_lidar_m = float(np.min(lidar_readings)) * self.world_size
            if min_lidar_m < self.safety_distance:
                risk_penalty = 4.0 * (self.safety_distance - min_lidar_m) # 2  ıs too gentle, 8 is too harsh
                reward -= risk_penalty

        return float(reward)
    
