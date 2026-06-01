import os
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

from env.uav_env_continuous import UAVLiDARContinuousEnv
from agents.ppo_agent import PPOAgent


def generate_ppo_flight_animation(checkpoint_episode=3000, n_obstacles=0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stack_size = 3

    # Initialize the matching continuous environment
    env = UAVLiDARContinuousEnv(n_obstacles=n_obstacles)
    stacked_state_dim = env.observation_space.shape[0] * stack_size
    action_dim = env.action_space.shape[0]

    # Load PPO Agent weights
    agent = PPOAgent(state_dim=stacked_state_dim,
                     action_dim=action_dim, device=device)
    checkpoint_path = f"/content/drive/MyDrive/drl-uav-navigation/outputs_ppo/checkpoints/ppo_episode_{checkpoint_episode}.pth"

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return None

    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False)
    agent.policy.load_state_dict(checkpoint["policy_state_dict"])
    agent.policy_old.load_state_dict(checkpoint["policy_state_dict"])
    agent.policy.eval()
    agent.policy_old.eval()

    # Run one full flight trajectory to collect data frames
    obs, _ = env.reset(seed=42)  # Fixed seed for a reliable diagnostic flight
    frame_stack = deque([obs] * stack_size, maxlen=stack_size)
    state = np.concatenate(list(frame_stack), axis=0)

    positions = []
    headings = []
    thrust_commands = []
    torque_commands = []

    max_eval_steps = 500
    step_count = 0
    reached = False

    print("Simulating evaluation flight trajectory...")
    while step_count < max_eval_steps:
        action, _, _ = agent.select_action(state, evaluate=True)
        next_obs, reward, terminated, truncated, info = env.step(action)

        # Track parameters for visualization
        positions.append(env.pos.copy())
        headings.append(env.theta)
        thrust_commands.append(action[0])
        torque_commands.append(action[1])

        frame_stack.append(next_obs)
        state = np.concatenate(list(frame_stack), axis=0)
        step_count += 1

        if terminated or truncated:
            reached = info.get('reached_goal', False)
            break

    print(f"Flight recorded. Steps: {step_count} | Reached Goal: {reached}")

    # --- Build Matplotlib Animation Plot ---
    fig, (ax_map, ax_control) = plt.subplots(1, 2, figsize=(12, 6))

    # Left subplot: The Spatial Map
    ax_map.set_xlim(0, env.world_size)
    ax_map.set_ylim(0, env.world_size)
    ax_map.grid(True, linestyle='--', alpha=0.5)
    ax_map.set_title(f"PPO Flight Path Evaluation (Obstacles: {n_obstacles})")

    # Draw Goal Zone
    goal_circle = plt.Circle(env.goal, env.goal_radius,
                             color='g', alpha=0.3, label="Goal")
    ax_map.add_patch(goal_circle)
    ax_map.plot(env.goal[0], env.goal[1], 'gx', markersize=10)

    # Draw Static Obstacles if present
    for center, radius in env.obstacles:
        obs_circle = plt.Circle(center, radius, color='r', alpha=0.4)
        ax_map.add_patch(obs_circle)

    # Dynamic plot elements
    trail_line, = ax_map.plot([], [], 'b--', alpha=0.6, label="UAV Trail")
    uav_dot, = ax_map.plot([], [], 'bo', markersize=8)
    heading_arrow = ax_map.quiver(
        0, 0, 0, 0, color='darkblue', scale=15, width=0.01)

    ax_map.legend(loc="upper left")

    # Right subplot: Continuous Controls Monitoring
    steps_x = np.arange(step_count)
    ax_control.set_xlim(0, max(50, step_count))
    ax_control.set_ylim(-1.1, 1.1)
    ax_control.set_title("Real-time Continuous Action Outputs")
    ax_control.set_xlabel("Environment Steps")
    ax_control.set_ylabel("Normalized Control Scales")

    thrust_line, = ax_control.plot(
        [], [], 'darkorange', label="Thrust Input (0 to 1)")
    torque_line, = ax_control.plot(
        [], [], 'teal', label="Torque Input (-1 to 1)")
    ax_control.axhline(0, color='black', linewidth=0.8, linestyle=':')
    ax_control.legend(loc="lower left")

    pos_history_x = []
    pos_history_y = []

    def init():
        trail_line.set_data([], [])
        uav_dot.set_data([], [])
        thrust_line.set_data([], [])
        torque_line.set_data([], [])
        return trail_line, uav_dot, heading_arrow, thrust_line, torque_line

    def update(frame):
        # Update map visualization
        current_pos = positions[frame]
        pos_history_x.append(current_pos[0])
        pos_history_y.append(current_pos[1])

        trail_line.set_data(pos_history_x, pos_history_y)
        uav_dot.set_data([current_pos[0]], [current_pos[1]])

        # Redraw the orientation arrow based on theta heading
        current_theta = headings[frame]
        heading_arrow.set_offsets(current_pos)
        heading_arrow.set_UVC(np.cos(current_theta), np.sin(current_theta))

        # Update control history visualization
        thrust_line.set_data(steps_x[:frame+1], thrust_commands[:frame+1])
        torque_line.set_data(steps_x[:frame+1], torque_commands[:frame+1])

        return trail_line, uav_dot, heading_arrow, thrust_line, torque_line

    ani = FuncAnimation(fig, update, frames=step_count,
                        init_func=init, blit=False, interval=50)
    plt.close()
    return ani
