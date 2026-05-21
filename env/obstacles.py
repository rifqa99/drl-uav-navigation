self.obstacles = []

num_obstacles = np.random.randint(4, 8)

for _ in range(num_obstacles):
    radius = np.random.uniform(0.8, 1.8)

    center = np.random.uniform(
        low=2.0,
        high=self.world_size - 2.0,
        size=2,
    )

    self.obstacles.append((center, radius))

self.pos = np.random.uniform(1.0, 3.0, size=2)

self.goal = np.random.uniform(
    self.world_size - 3.0,
    self.world_size - 1.0,
    size=2,
)
