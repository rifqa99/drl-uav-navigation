import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque

# Import the HTML5 display components for Google Colab rendering
from IPython.display import HTML

from env.uav_env import UAVLiDAREnv
from agents.dqn_agent import DQNAgent  # FIXED: Import the agent wrapper


def animate():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Match your Phase 2 hard-map evaluation constraints dynamically
    env = UAVLiDAREnv(
        seed=123,  # Fresh validation seed to prove generalizability
        drag=0.15,
        thrust=0.6,
        max_speed=1.2,
        n_obstacles=8  # Set to 8 to test your crowded-map parameters
    )

    stack_size = 3
    original_obs_dim = env.observation_space.shape[0]
    stacked_state_dim = original_obs_dim * stack_size

    # FIXED: Initialize the full DQNAgent so select_action() is fully available
    agent = DQNAgent(
        state_dim=stacked_state_dim,
        action_dim=env.action_space.n,
        device=device
    )

    checkpoint_path = "/content/drive/MyDrive/drl-uav-navigation/outputs/checkpoints/dqn_episode_3000.pth"

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Could not locate your checkpoint file at: {checkpoint_path}")

    print(
        f"Loading weights from final convergence snapshot: {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False)
    agent.q_network.load_state_dict(checkpoint["q_network"])
    agent.epsilon = 0.0  # Force pure deterministic optimal evaluation path execution

    # Simulate and record flight path
    obs, _ = env.reset()
    frame_stack = deque([obs] * stack_size, maxlen=stack_size)
    state = np.concatenate(list(frame_stack), axis=0)

    positions = []
    lidar_history = []

    # Store initial baseline positions before loop execution
    positions.append(env.pos.copy())
    lidar_history.append(env._lidar_scan())

    # Generate flight matrices
    for _ in range(env.max_steps):
        action = agent.select_action(state)
        next_obs, reward, terminated, truncated, info = env.step(action)

        frame_stack.append(next_obs)
        next_state = np.concatenate(list(frame_stack), axis=0)

        positions.append(env.pos.copy())
        lidar_history.append(env._lidar_scan())

        state = next_state
        if terminated or truncated:
            print(
                f"Flight path recording finished. Steps: {len(positions)}, Terminated: {terminated}, Reached Goal: {info['reached_goal']}")
            break

    # --- ANIMATION RENDERING GENERATION SYSTEM ---
    fig, ax = plt.subplots(figsize=(6, 6))
    lidar_angles = np.linspace(0, 2 * np.pi, env.n_lidar, endpoint=False)

    def update(frame):
        ax.clear()
        ax.set_xlim(0, env.world_size)
        ax.set_ylim(0, env.world_size)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Autonomous Flight Evaluation | Frame Step: {frame}")

        # Render static obstacle distributions
        for center, radius in env.obstacles:
            circle = plt.Circle(center, radius, color='darkred', alpha=0.35)
            ax.add_patch(circle)

        # Render objective destination node
        goal = plt.Circle(env.goal, env.goal_radius,
                          color='forestgreen', alpha=0.5)
        ax.add_patch(goal)
        ax.text(env.goal[0], env.goal[1] + 0.4, "Goal Target",
                ha="center", weight='bold', color='darkgreen')

        # Extract targeted step tracking metrics
        pos = positions[frame]

        # Render dynamic UAV state node
        uav = plt.Circle(pos, env.collision_radius,
                         color='royalblue', alpha=0.9)
        ax.add_patch(uav)
        ax.text(pos[0], pos[1] + 0.35, "UAV", ha="center",
                weight='bold', color='darkblue')

        # Trace historical trajectory line array updates
        path = positions[: frame + 1]
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        ax.plot(xs, ys, color='mediumblue', linewidth=2.0, linestyle='-')

        # Render active LiDAR sensor beam arrays
        lidar = lidar_history[frame]
        for angle, dist_norm in zip(lidar_angles, lidar):
            dist = dist_norm * env.world_size
            end = pos + dist * np.array([np.cos(angle), np.sin(angle)])

            # Draw lines changing color to crimson if danger barriers trigger near range detection limits
            line_color = 'crimson' if dist_norm < 0.25 else 'lightslateregray'
            ax.plot(
                [pos[0], end[0]], [pos[1], end[1]],
                color=line_color, linewidth=0.5, alpha=0.3
            )

    # Compile animation parameters safely
    ani = FuncAnimation(
        fig,
        update,
        frames=len(positions),
        interval=60,  # Adjusted delay step rate intervals slightly for visible tracking display
        repeat=False,
    )
    plt.close()  # Prevents extra hanging empty frame canvas window generation plots

    # Return the HTML5 compatible object layout structure back out to cell framework
    return HTML(ani.to_jshtml())


if __name__ == "__main__":
    # In Google Colab, you need to save the function return directly to a cell execution line:
    # simply type: `animate()` directly into your active runtime notebook cell workspace block
    animate()
