import numpy as np
import math


class AdvancedUAVDynamicsContinuous:

    def __init__(self):
        # Kinematic state parameters
        self.dt = 0.1
        self.mass = 1.5          # Total mass of the quadcopter (kg)
        self.I_z = 0.05          # Rotational moment of inertia (kg*m^2)
        self.drag_linear = 0.05  # Linear aerodynamic drag coefficient (lambda)
        self.drag_angular = 0.1  # Angular drag coefficient
        self.arm_length = 0.25   # Distance from center to rotors (m)

        # Ambient continuous wind vector
        self.wind_speed = np.array([0.5, -0.2])

    def update_physics(self, pos, vel, theta, omega, action):
        """Updates the rigid body state using variable thrust and torque inputs.

        action: np.array([thrust_input, torque_input]) 
                thrust_input is expected in range [0.0, 1.0] 
                torque_input is expected in range [-1.0, 1.0]
        """
        # --- FIXED: These lines MUST be indented inside this method ---
        # Linear forward force scale maxes out at 3.0 N
        thrust = action[0] * 3.0
        # Rotational torque scale maps precisely to [-0.5, 0.5]
        torque = action[1] * 0.5

        # 1. Stochastic Wind Noise Addition
        wind_noise = np.random.normal(0, 0.1, size=(2,))
        total_wind = self.wind_speed + wind_noise

        # 2. Linear Second-Order Dynamics Equations
        f_x = thrust * math.cos(theta) + total_wind[0]
        f_y = thrust * math.sin(theta) + total_wind[1]

        accel_x = (f_x / self.mass) - self.drag_linear * vel[0]
        accel_y = (f_y / self.mass) - self.drag_linear * vel[1]

        vel[0] += accel_x * self.dt
        vel[1] += accel_y * self.dt
        pos[0] += vel[0] * self.dt
        pos[1] += vel[1] * self.dt

        # 3. Rotational Dynamics Equations (Inertia)
        alpha = (torque / self.I_z) - self.drag_angular * omega
        omega += alpha * self.dt
        theta += omega * self.dt

        # Normalize heading angle between [-pi, pi]
        theta = math.atan2(math.sin(theta), math.cos(theta))

        return pos, vel, theta, omega, np.array([accel_x, accel_y])
