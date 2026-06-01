import numpy as np
from env.dynamics import AdvancedUAVDynamics
from env.rewards import UAVRewardShaping

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    gym = None
    spaces = None


class UAVLiDARDynamicEnv(gym.Env if gym is not None else object):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        world_size=10.0,
        n_lidar=64,
        max_steps=2000,
        dt=0.1,
        drag=0.2,
        thrust=1.0,
        max_speed=1.2,
        goal_radius=0.6,
        collision_radius=0.25,
        n_obstacles=8,  # Evaluated directly against your dense obstacle benchmark
        obstacle_radius_range=(0.3, 0.8),
        seed=None,
    ):
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

        self.dynamics = AdvancedUAVDynamics()
        self.reward_shaper = UAVRewardShaping(world_size=self.world_size)

        self.theta = 0.0
        self.omega = 0.0
        self.rng = np.random.default_rng(seed)

        self.action_space = spaces.Discrete(5)

        obs_dim = self.n_lidar + 5
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        self.pos = None
        self.vel = None
        self.goal = None
        self.obstacles = None
        self.obstacle_vels = None  # Tracks moving velocity vectors for each circle
        self.steps = 0
        self.prev_distance = None
        self.prev_action = 0
        self.trajectory = []

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.steps = 0
        self.theta = 0.0
        self.omega = 0.0
        self.vel = np.zeros(2, dtype=np.float32)
        self.prev_action = 0

        self.pos = np.random.uniform(1.0, 3.0, size=2).astype(np.float32)
        self.goal = np.random.uniform(
            self.world_size - 3.0, self.world_size - 1.0, size=2
        ).astype(np.float32)

        self._generate_dynamic_obstacles()
        self.prev_distance = self._distance_to_goal()
        self.trajectory = [self.pos.copy()]

        return self._get_obs(), {}

    def _generate_dynamic_obstacles(self):
        """Generates random obstacle locations and assigns them constant kinetic velocities."""
        self.obstacles = []
        self.obstacle_vels = []
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
                    self.obstacles.append(
                        [center.astype(np.float32), float(radius)])

                    # Assign a slow velocity vector components between [-0.25, 0.25] m/s
                    v_x = self.rng.uniform(-0.25, 0.25)
                    v_y = self.rng.uniform(-0.25, 0.25)
                    self.obstacle_vels.append(
                        np.array([v_x, v_y], dtype=np.float32))
                    break

    def step(self, action):
        self.steps += 1

        # 1. Update UAV Rigid Body State via Discrete Action Mapping
        self.pos, self.vel, self.theta, self.omega, accel = self.dynamics.update_physics(
            self.pos, self.vel, self.theta, self.omega, action
        )

        speed = np.linalg.norm(self.vel)
        if speed > self.max_speed:
            self.vel = self.vel / (speed + 1e-8) * self.max_speed

        self.pos = np.clip(self.pos, 0.0, self.world_size)
        self.trajectory.append(self.pos.copy())

        # 2. Update Obstacle Spatial Centers Frame-by-Frame
        for i in range(len(self.obstacles)):
            center, radius = self.obstacles[i]
            vel = self.obstacle_vels[i]

            new_center = center + vel * self.dt

            # Boundary Reflection Check: If the sphere hits a wall, invert its velocity component
            if new_center[0] <= radius or new_center[0] >= self.world_size - radius:
                vel[0] *= -1.0
            if new_center[1] <= radius or new_center[1] >= self.world_size - radius:
                vel[1] *= -1.0

            new_center = np.clip(new_center, radius, self.world_size - radius)
            self.obstacles[i] = [new_center.astype(np.float32), radius]

        # 3. Structural Condition Evaluators
        distance = self._distance_to_goal()
        progress = self.prev_distance - distance
        self.prev_distance = distance

        collision = self._check_collision()
        reached_goal = distance <= self.goal_radius
        timeout = self.steps >= self.max_steps

        # Relying perfectly on your original discrete reward shaping setup
        reward = self.reward_shaper.compute_reward(
            progress=progress, current_action=action, prev_action=self.prev_action,
            lidar_readings=self._lidar_scan(), distance=distance, speed=float(speed),
            max_speed=self.max_speed, collision=collision, reached_goal=reached_goal
        )

        energy_use = 1.0 if action in [1, 2] else (
            0.2 if action in [3, 4] else 0.0)
        smoothness_penalty = 1.0 if action != self.prev_action else 0.0

        self.prev_action = action
        terminated = collision or reached_goal
        truncated = timeout

        info = {
            "distance_to_goal": distance, "collision": collision, "reached_goal": reached_goal,
            "speed": float(speed), "energy_use": float(energy_use), "smoothness_violation": float(smoothness_penalty)
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

        return np.concatenate([lidar, np.array([vx, vy, norm_theta, norm_omega, target_angle], dtype=np.float32)])

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
            readings[i] = np.clip(
                min(min_dist, wall_dist) / max_range, 0.0, 1.0)

        return readings

    def _ray_circle_distance(self, origin, direction, center, radius):
        oc = origin - center
        b = 2.0 * np.dot(oc, direction)
        c = np.dot(oc, oc) - radius**2
        discriminant = b**2 - 4 * c
        if discriminant < 0:
            return None
        sqrt_disc = np.sqrt(discriminant)
        t1, t2 = (-b - sqrt_disc) / 2.0, (-b + sqrt_disc) / 2.0
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
        return min(distances) if distances else self.world_size

    def _distance_to_goal(self):
        return float(np.linalg.norm(self.goal - self.pos))

    def _check_collision(self):
        if np.any(self.pos <= 0.0) or np.any(self.pos >= self.world_size):
            return True
        for center, radius in self.obstacles:
            if np.linalg.norm(self.pos - center) <= radius + self.collision_radius:
                return True
        return False
