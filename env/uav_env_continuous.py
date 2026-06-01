import numpy as np
import matplotlib.pyplot as plt
from env.dynamics_continuous import AdvancedUAVDynamicsContinuous
from env.rewards import UAVRewardShapingContinuous

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    gym = None
    spaces = None


class UAVLiDARContinuousEnv(gym.Env if gym is not None else object):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        world_size=10.0,
        n_lidar=64,
        max_steps=3000,
        dt=0.1,
        drag=0.2,
        thrust=1.0,
        max_speed=1.2,
        goal_radius=0.6,
        collision_radius=0.25,
        n_obstacles=5,
        obstacle_radius_range=(0.3, 0.8),
        seed=None,
    ):
        if gym is None:
            raise ImportError("Install gymnasium first: pip install gymnasium")

        super().__init__()

        self.world_size = world_size
        self.n_lidar = n_lidar
        self.max_steps = max_steps
        self.dt = dt
        self.drag = drag
        self.thrust = thrust
        self.max_speed = max_speed
        self.goal_radius = goal_radius
        self.collision_radius = collision_radius
        self.n_obstacles = n_obstacles
        self.obstacle_radius_range = obstacle_radius_range

        # Physics & Reward Modules
        self.dynamics = AdvancedUAVDynamicsContinuous()
        self.reward_shaper = UAVRewardShapingContinuous(
            world_size=self.world_size)

        self.theta = 0.0
        self.omega = 0.0
        self.rng = np.random.default_rng(seed)

        # --- CHANGED FOR PPO: CONTINUOUS ACTION SPACE ---
        # Action space shape: (2,) -> [Thrust command, Torque command]
        # Thrust maps to [0.0, 1.0], Torque maps to [-1.0, 1.0]
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # Dimension stays exactly 69 (64 LiDAR + vx, vy, theta, omega, target_angle)
        obs_dim = self.n_lidar + 5
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.pos = None
        self.vel = None
        self.goal = None
        self.obstacles = None
        self.steps = 0
        self.prev_distance = None

        # Stored as an array of zeros to match continuous tracking variables
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.trajectory = []

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.steps = 0
        self.theta = 0.0
        self.omega = 0.0

        self.vel = np.zeros(2, dtype=np.float32)
        self.prev_action = np.zeros(2, dtype=np.float32)

        self.pos = np.random.uniform(1.0, 3.0, size=2).astype(np.float32)
        self.goal = np.random.uniform(
            self.world_size - 3.0, self.world_size - 1.0, size=2).astype(np.float32)

        self.obstacles = self._generate_obstacles()
        self.prev_distance = self._distance_to_goal()
        self.trajectory = [self.pos.copy()]

        return self._get_obs(), {}

    def step(self, action):
        self.steps += 1

        # --- CONTINUOUS INTERACTION HOOK ---
        # Clean continuous clips to prevent value leakage outside valid physical bounds
        thrust_cmd = float(np.clip(action[0], 0.0, 1.0))
        torque_cmd = float(np.clip(action[1], -1.0, 1.0))
        continuous_action_vector = np.array(
            [thrust_cmd, torque_cmd], dtype=np.float32)

        # Update physics via your dynamics engine module
        # (Note: If update_physics expects an index ID, we handle the mapping below)
        self.pos, self.vel, self.theta, self.omega, accel = self.dynamics.update_physics(
            self.pos, self.vel, self.theta, self.omega, continuous_action_vector
        )

        # Speed Limiter
        speed = np.linalg.norm(self.vel)
        if speed > self.max_speed:
            self.vel = self.vel / (speed + 1e-8) * self.max_speed

        self.pos = np.clip(self.pos, 0.0, self.world_size)
        self.trajectory.append(self.pos.copy())

        # Calculations
        distance = self._distance_to_goal()
        progress = self.prev_distance - distance
        self.prev_distance = distance

        collision = self._check_collision()
        reached_goal = distance <= self.goal_radius
        timeout = self.steps >= self.max_steps

        # --- MODULAR REWARD SHAPING CALL ---
        # Note: We pass the continuous vector or a placeholder scalar index to reward shaper
        reward = self.reward_shaper.compute_reward(
            progress=progress,
            current_action=continuous_action_vector,
            prev_action=self.prev_action,
            lidar_readings=self._lidar_scan(),
            distance=distance,
            speed=float(speed),
            max_speed=self.max_speed,
            collision=collision,
            reached_goal=reached_goal
        )

        # Metric logging placeholders
        energy_use = thrust_cmd
        smoothness_penalty = float(np.linalg.norm(
            continuous_action_vector - self.prev_action))

        self.prev_action = continuous_action_vector.copy()
        terminated = collision or reached_goal
        truncated = timeout

        info = {
            "distance_to_goal": distance,
            "collision": collision,
            "reached_goal": reached_goal,
            "speed": float(speed),
            "energy_use": float(energy_use),
            "smoothness_violation": smoothness_penalty
        }

        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        lidar = self._lidar_scan()
        vx = self.vel[0] / self.max_speed
        vy = self.vel[1] / self.max_speed
        norm_theta = self.theta / np.pi
        norm_omega = self.omega / np.pi

        target_vec = self.goal - self.pos
        target_angle = np.arctan2(target_vec[1], target_vec[0]) / np.pi

        obs = np.concatenate([
            lidar,
            np.array([vx, vy, norm_theta, norm_omega,
                     target_angle], dtype=np.float32)
        ])
        return obs.astype(np.float32)

    def _lidar_scan(self):
        angles = np.linspace(0, 2 * np.pi, self.n_lidar, endpoint=False)
        max_range = self.world_size
        readings = np.ones(self.n_lidar, dtype=np.float32)

        for i, angle in enumerate(angles):
            direction = np.array(
                [np.cos(angle), np.sin(angle)], dtype=np.float32)
            min_dist = max_range

            for center, radius in self.obstacles:
                dist = self._ray_circle_distance(
                    self.pos, direction, center, radius)
                if dist is not None:
                    min_dist = min(min_dist, dist)

            wall_dist = self._ray_wall_distance(self.pos, direction)
            min_dist = min(min_dist, wall_dist)
            readings[i] = np.clip(min_dist / max_range, 0.0, 1.0)

        return readings

    def _ray_circle_distance(self, origin, direction, center, radius):
        oc = origin - center
        b = 2.0 * np.dot(oc, direction)
        c = np.dot(oc, oc) - radius**2
        discriminant = b**2 - 4 * c

        if discriminant < 0:
            return None

        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / 2.0
        t2 = (-b + sqrt_disc) / 2.0
        valid = [t for t in [t1, t2] if t >= 0]

        return min(valid) if valid else None

    def _ray_wall_distance(self, origin, direction):
        distances = []
        if direction[0] > 1e-6:
            distances.append((self.world_size - origin[0]) / direction[0])
        elif direction[0] < -1e-6:
            distances.append((0.0 - origin[0]) / direction[0])

        if direction[1] > 1e-6:
            distances.append((self.world_size - origin[1]) / direction[1])
        elif direction[1] < -1e-6:
            distances.append((0.0 - origin[1]) / direction[1])

        distances = [d for d in distances if d >= 0]
        return min(distances) if distances else self.world_size

    def _generate_obstacles(self):
        obstacles = []
        start = np.array([1.0, 1.0])
        goal = np.array([self.world_size - 1.0, self.world_size - 1.0])

        for _ in range(self.n_obstacles):
            for _attempt in range(100):
                radius = self.rng.uniform(*self.obstacle_radius_range)
                center = self.rng.uniform(
                    radius, self.world_size - radius, size=2)

                too_close_start = np.linalg.norm(center - start) < 1.5
                too_close_goal = np.linalg.norm(center - goal) < 1.5

                if not too_close_start and not too_close_goal:
                    obstacles.append(
                        (center.astype(np.float32), float(radius)))
                    break
        return obstacles

    def _distance_to_goal(self):
        return float(np.linalg.norm(self.goal - self.pos))

    def _check_collision(self):
        if np.any(self.pos <= 0.0) or np.any(self.pos >= self.world_size):
            return True
        for center, radius in self.obstacles:
            if np.linalg.norm(self.pos - center) <= radius + self.collision_radius:
                return True
        return False

    def render(self):
        plt.figure(figsize=(6, 6))
        ax = plt.gca()
        ax.set_xlim(0, self.world_size)
        ax.set_ylim(0, self.world_size)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        for center, radius in self.obstacles:
            circle = plt.Circle(center, radius, alpha=0.4)
            ax.add_patch(circle)

        goal_circle = plt.Circle(self.goal, self.goal_radius, alpha=0.6)
        ax.add_patch(goal_circle)
        ax.text(self.goal[0], self.goal[1] + 0.4, "Goal", ha="center")

        uav_circle = plt.Circle(self.pos, self.collision_radius, alpha=0.9)
        ax.add_patch(uav_circle)
        ax.text(self.pos[0], self.pos[1] + 0.35, "UAV", ha="center")

        trajectory = np.array(self.trajectory)
        if len(trajectory) > 1:
            ax.plot(trajectory[:, 0], trajectory[:, 1], linewidth=2, alpha=0.7)

        plt.show()
