import numpy as np
import matplotlib.pyplot as plt

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    gym = None
    spaces = None


class UAVLiDAREnv(gym.Env if gym is not None else object):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        world_size=10.0,
        n_lidar=64,
        max_steps=3000,
        dt=0.1,
        drag=0.2,  # 0.1 hıgh ınterıa -> 0.2 stability
        thrust=1.0,
        max_speed=1.2,
        goal_radius=0.4,
        collision_radius=0.25,
        n_obstacles=8,
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

        self.rng = np.random.default_rng(seed)

        self.action_vectors = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [-1.0, 0.0],
                [0.0, 1.0],
                [0.0, -1.0],
                [1.0, 1.0],
                [1.0, -1.0],
                [-1.0, 1.0],
                [-1.0, -1.0],
            ],
            dtype=np.float32,
        )

        for i in range(len(self.action_vectors)):
            norm = np.linalg.norm(self.action_vectors[i])
            if norm > 0:
                self.action_vectors[i] /= norm

        self.action_space = spaces.Discrete(len(self.action_vectors))

        obs_dim = self.n_lidar + 3
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
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.trajectory = []

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.steps = 0
        self.vel = np.zeros(2, dtype=np.float32)
        self.prev_action = np.zeros(2, dtype=np.float32)

        self.pos = np.array([1.0, 1.0], dtype=np.float32)
        self.goal = np.array(
            [self.world_size - 1.0, self.world_size - 1.0],
            dtype=np.float32,
        )

        self.obstacles = self._generate_obstacles()
        self.prev_distance = self._distance_to_goal()
        self.trajectory = [self.pos.copy()]

        return self._get_obs(), {}


def step(self, action):
        self.steps += 1

        # 1. Action to Thrust Vector
        action_vec = self.action_vectors[action] * self.thrust

        # 2. Physics Update: Velocity = Previous Velocity + (Thrust * dt) - (Drag * Velocity)
        # This implements the second-order dynamics from your proposal
        self.vel = self.vel + action_vec * self.dt - self.drag * self.vel

        # Speed Limiter
        speed = np.linalg.norm(self.vel)
        if speed > self.max_speed:
            self.vel = self.vel / (speed + 1e-8) * self.max_speed

        # Position Update
        self.pos = self.pos + self.vel * self.dt
        self.pos = np.clip(self.pos, 0.0, self.world_size)

        # Update Trajectory for visualization
        self.trajectory.append(self.pos.copy())

        # 3. Distance and Progress Calculations
        distance = self._distance_to_goal()
        progress = self.prev_distance - distance
        self.prev_distance = distance

        # 4. Status Checks
        collision = self._check_collision()
        reached_goal = distance <= self.goal_radius
        timeout = self.steps >= self.max_steps

        # 5. Advanced Reward Shaping
        reward = 0.0

        # Progress Reward: High weight for moving toward goal
        reward += 15.0 * progress

        # Step Penalty: Discourage loitering/unnecessary movement
        reward -= 0.02

        # Energy Penalty: Discourage high thrust usage (thrust magnitude squared)
        # Directly supports your "energy-aware" navigation objective
        energy_use = np.linalg.norm(action_vec)**2
        reward -= 0.05 * energy_use

        # Smoothness Penalty: Penalize sudden changes in thrust (acceleration variance)
        # Helps prevent jittery flight paths
        smoothness_penalty = np.linalg.norm(action_vec - self.prev_action)**2
        reward -= 0.15 * smoothness_penalty

        # Danger Zone / Safety Penalty: Non-physical soft constraint
        # Simulates restricted area avoidance in GPS-denied scenarios
        min_lidar = np.min(self._lidar_scan())
        if min_lidar < 0.25:  # "Buffer Zone"
            # Reward decreases linearly as the agent gets closer to an obstacle
            reward -= 1.0 * (1.0 - min_lidar)

        # Collision Penalty: Severe penalty to discourage contact
        if collision:
            reward -= 100.0

        # Success Reward: Massive reward for completing the mission
        if reached_goal:
            reward += 200.0

        # 6. Housekeeping for next step
        self.prev_action = action_vec.copy()
        terminated = collision or reached_goal
        truncated = timeout

        info = {
            "distance_to_goal": distance,
            "collision": collision,
            "reached_goal": reached_goal,
            "speed": float(speed),
            "energy_use": float(energy_use),
            "smoothness_violation": float(smoothness_penalty)
        }

        return self._get_obs(), reward, terminated, truncated, info


    def _get_obs(self):
        lidar = self._lidar_scan()

        vx = self.vel[0] / self.max_speed
        vy = self.vel[1] / self.max_speed

        target_vec = self.goal - self.pos
        target_angle = np.arctan2(target_vec[1], target_vec[0]) / np.pi

        obs = np.concatenate(
            [
                lidar,
                np.array([vx, vy, target_angle], dtype=np.float32),
            ]
        )

        return obs.astype(np.float32)

    def _lidar_scan(self):
        angles = np.linspace(0, 2 * np.pi, self.n_lidar, endpoint=False)
        max_range = self.world_size
        readings = np.ones(self.n_lidar, dtype=np.float32)

        for i, angle in enumerate(angles):
            direction = np.array(
                [np.cos(angle), np.sin(angle)],
                dtype=np.float32,
            )

            min_dist = max_range

            for center, radius in self.obstacles:
                dist = self._ray_circle_distance(
                    self.pos,
                    direction,
                    center,
                    radius,
                )

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

        if not valid:
            return None

        return min(valid)

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
                    radius,
                    self.world_size - radius,
                    size=2,
                )

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
            ax.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                linewidth=2,
                alpha=0.7,
            )

        angles = np.linspace(0, 2 * np.pi, self.n_lidar, endpoint=False)
        lidar = self._lidar_scan()
        max_range = self.world_size

        for angle, dist_norm in zip(angles, lidar):
            dist = dist_norm * max_range
            end = self.pos + dist * np.array([np.cos(angle), np.sin(angle)])

            ax.plot(
                [self.pos[0], end[0]],
                [self.pos[1], end[1]],
                linewidth=0.8,
                alpha=0.5,
            )

        ax.arrow(
            self.pos[0],
            self.pos[1],
            self.vel[0],
            self.vel[1],
            head_width=0.15,
            length_includes_head=True,
        )

        ax.set_title(f"Step: {self.steps}")
        plt.show()


if __name__ == "__main__":
    env = UAVLiDAREnv(seed=42)
    obs, info = env.reset()

    print("Observation shape:", obs.shape)

    total_reward = 0
    for _ in range(500):

        target_vec = env.goal - env.pos
        target_vec = target_vec / (np.linalg.norm(target_vec) + 1e-8)

        lidar = env._lidar_scan()
        action_dirs = env.action_vectors

        scores = action_dirs @ target_vec

        for i, action_dir in enumerate(action_dirs):
            if i == 0:
                continue

            action_angle = np.arctan2(action_dir[1], action_dir[0])
            lidar_angles = np.linspace(
                0, 2 * np.pi, env.n_lidar, endpoint=False)

            angle_diffs = np.abs(
                np.angle(np.exp(1j * (lidar_angles - action_angle)))
            )

            nearest_lidar_idx = np.argmin(angle_diffs)

            obstacle_penalty = 4.0 * (1.0 - lidar[nearest_lidar_idx])

            scores[i] -= obstacle_penalty

        # scores[0] -= 0.3
        action = int(np.argmax(scores[1:]) + 1)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        print(reward, info)

        if terminated or truncated:
            break
    print("Total reward:", total_reward)

    env.render()
