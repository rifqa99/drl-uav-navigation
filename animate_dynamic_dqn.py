import os
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from env.uav_env_dynamic import UAVLiDARDynamicEnv
from agents.dqn_agent import DQNAgent


def generate_dynamic_dqn_animation(checkpoint_episode=2000, n_obstacles=2):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stack_size = 3

    # Initialize dynamic environment matching your training configuration
    env = UAVLiDARDynamicEnv(n_obstacles=n_obstacles)
    state_dim = env.observation_space.shape[0] * stack_size
    action_dim = env.action_space.n

    # Load your trained model weights safely
    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, device=device)
    checkpoint_path = f"/content/drive/MyDrive/drl-uav-navigation/outputs_dynamic/checkpoints/{checkpoint_episode}.pth"

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return None

    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False)
    agent.q_network.load_state_dict(checkpoint["model_state_dict"])
    # Force deterministic action selection (greedy policy evaluation)
    agent.epsilon = 0.0

    # Run one full flight trajectory to collect data frames
    # Set seed for reproducible evaluation tracking
    obs, _ = env.reset(seed=12)
    frame_stack = deque([obs] * stack_size, maxlen=stack_size)
    state = np.concatenate(list(frame_stack), axis=0)

    positions = []
    headings = []
    obstacle_history = []  # Tracks moving center coordinates for rendering

    max_eval_steps = 1000
    step_count = 0
    reached = False

    print("Simulating evaluation flight trajectory against moving targets...")
    while step_count < max_eval_steps:
        action = agent.select_action(state)
        next_obs, reward, terminated, truncated, info = env.step(action)

        # Save positions of the drone and ALL moving obstacles for this frame
        positions.append(env.pos.copy())
        headings.append(env.theta)

        current_obs_frames = [[center.copy(), radius]
                              for center, radius in env.obstacles]
        obstacle_history.append(current_obs_frames)

        frame_stack.append(next_obs)
        state = np.concatenate(list(frame_stack), axis=0)
        step_count += 1

        if terminated or truncated:
            reached = info.get('reached_goal', False)
            break

    print(f"Flight recorded. Steps: {step_count} | Reached Goal: {reached}")

    # --- Build Matplotlib Animation Plot ---
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, env.world_size)
    ax.set_ylim(0, env.world_size)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_title(
        f"Dynamic DQN Flight Path Evaluation (Obstacles: {n_obstacles})")

    # Render Goal Location
    goal_circle = plt.Circle(env.goal, env.goal_radius,
                             color='g', alpha=0.3, label="Goal Zone")
    ax.add_patch(goal_circle)
    ax.plot(env.goal[0], env.goal[1], 'gx', markersize=10)

    # Dynamic render elements
    trail_line, = ax.plot([], [], 'b--', alpha=0.6, label="UAV Flight Path")
    uav_dot, = ax.plot([], [], 'bo', markersize=8)
    heading_arrow = ax.quiver(
        0, 0, 0, 0, color='darkblue', scale=15, width=0.01)

    # Initialize list to hold moving obstacle circle patches
    obs_patches = []
    initial_obstacles = obstacle_history[0]
    for center, radius in initial_obstacles:
        circle = plt.Circle(center, radius, color='r', alpha=0.4)
        ax.add_patch(circle)
        obs_patches.append(circle)

    ax.legend(loc="upper left")
    pos_history_x = []
    pos_history_y = []

    def init():
        trail_line.set_data([], [])
        uav_dot.set_data([], [])
        return [trail_line, uav_dot, heading_arrow] + obs_patches

    def update(frame):
        # Update drone history line and location marker
        current_pos = positions[frame]
        pos_history_x.append(current_pos[0])
        pos_history_y.append(current_pos[1])

        trail_line.set_data(pos_history_x, pos_history_y)
        uav_dot.set_data([current_pos[0]], [current_pos[1]])

        # Adjust heading pointer vector
        current_theta = headings[frame]
        heading_arrow.set_offsets(current_pos)
        heading_arrow.set_UVC(np.cos(current_theta), np.sin(current_theta))

        # Extract and update positions for all moving obstacle shapes
        current_obstacles = obstacle_history[frame]
        for idx, (center, radius) in enumerate(current_obstacles):
            obs_patches[idx].set_center(center)

        return [trail_line, uav_dot, heading_arrow] + obs_patches

    ani = FuncAnimation(fig, update, frames=step_count,
                        init_func=init, blit=False, interval=60)
    plt.close()
    return ani
