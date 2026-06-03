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
        reached_goal,
        speed=0.0,
        omega=0.0
    ):
        # 1. Catastrophic Terminal Triggers
        if collision:
            return -1000.0

        if reached_goal:
            terminal_reward = 1000.0
            safe_landing_speed = 0.4
            
            # Kinematic Braking Constraint
            if speed > safe_landing_speed:
                terminal_reward -= 200.0 * (speed - safe_landing_speed)
            return float(terminal_reward)

        reward = 0.0

        # 2. Potential-Based Progress & Decisiveness Constraints
        reward += 5.0 * float(progress)
        reward -= 0.005  # Standard frame time penalty

        # 3. Dual Rotational Smoothness Constraints (Anti-Spinning Fix)
        if action in [3, 4]:
            reward -= 0.20  # Actuator selection penalty
            
        reward -= 0.10 * abs(float(omega))  # Kinetic angular velocity penalty

        # 4. Proximity Risk Envelope
        if lidar_readings is not None and len(lidar_readings) > 0:
            min_lidar_m = float(np.min(lidar_readings)) * self.world_size
            if min_lidar_m < self.safety_distance:
                risk_penalty = 2.0 * (self.safety_distance - min_lidar_m) # 2  ıs too gentle, 8 is too harsh
                reward -= risk_penalty

        return float(reward)