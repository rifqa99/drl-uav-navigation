import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from env.uav_env import UAVLiDAREnv
from agents.dueling_dqn import DuelingDQN


def animate():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = UAVLiDAREnv(
        seed=123,
        drag=0.15,
        thrust=0.6,
        max_speed=1.2,
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DuelingDQN(state_dim, action_dim)

    checkpoint = torch.load(
        "outputs/checkpoints/dqn_episode_1300.pth",
        map_location=device,
        weights_only=False,
    )

    agent.q_network.load_state_dict(checkpoint["q_network"])
    agent.epsilon = 0.0

    state, _ = env.reset()

    positions = []
    lidar_history = []

    for _ in range(env.max_steps):
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)

        positions.append(env.pos.copy())
        lidar_history.append(env._lidar_scan())

        state = next_state

        if terminated or truncated:
            break

    fig, ax = plt.subplots(figsize=(6, 6))

    def update(frame):
        ax.clear()

        ax.set_xlim(0, env.world_size)
        ax.set_ylim(0, env.world_size)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Step: {frame}")

        for center, radius in env.obstacles:
            circle = plt.Circle(center, radius, alpha=0.4)
            ax.add_patch(circle)

        goal = plt.Circle(env.goal, env.goal_radius, alpha=0.6)
        ax.add_patch(goal)
        ax.text(env.goal[0], env.goal[1] + 0.4, "Goal", ha="center")

        pos = positions[frame]

        uav = plt.Circle(pos, env.collision_radius, alpha=0.9)
        ax.add_patch(uav)
        ax.text(pos[0], pos[1] + 0.35, "UAV", ha="center")

        path = positions[: frame + 1]
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        ax.plot(xs, ys, linewidth=2)

        angles = env.n_lidar
        import numpy as np
        lidar_angles = np.linspace(0, 2 * np.pi, env.n_lidar, endpoint=False)

        lidar = lidar_history[frame]

        for angle, dist_norm in zip(lidar_angles, lidar):
            dist = dist_norm * env.world_size
            end = pos + dist * np.array([np.cos(angle), np.sin(angle)])

            ax.plot(
                [pos[0], end[0]],
                [pos[1], end[1]],
                linewidth=0.6,
                alpha=0.4,
            )

    ani = FuncAnimation(
        fig,
        update,
        frames=len(positions),
        interval=40,
        repeat=False,
    )

    plt.show()


if __name__ == "__main__":
    animate()
