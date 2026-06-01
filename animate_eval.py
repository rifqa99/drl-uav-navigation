import os
import torch
import numpy as np
from collections import deque

from env.uav_env import UAVLiDAREnv
from agents.dqn_agent import DQNAgent


def evaluate_policy():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    stack_size = 3
    total_eval_episodes = 10
    successful_goals = 0
    total_steps_taken = 0

    # Initialize basic network dimensions
    # We will rebuild the env dynamically inside the loop with changing seeds
    temp_env = UAVLiDAREnv(n_obstacles=8)
    stacked_state_dim = temp_env.observation_space.shape[0] * stack_size
    action_dim = temp_env.action_space.n

    agent = DQNAgent(state_dim=stacked_state_dim,
                     action_dim=action_dim, device=device)

    checkpoint_path = "/content/drive/MyDrive/drl-uav-navigation/outputs/checkpoints/dqn_episode_3000.pth"
    print(f"Loading weights for evaluation: {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False)
    agent.q_network.load_state_dict(checkpoint["q_network"])
    agent.epsilon = 0.0  # Turn off exploration for strict evaluation

    print("\n--- Starting 10-Episode Validation Run ---")

    for ep in range(total_eval_episodes):
        # Use a different seed for each episode, starting at 500 to ensure fresh unseen maps
        current_seed = 500 + ep

        env = UAVLiDAREnv(
            seed=current_seed,
            drag=0.15,
            thrust=0.6,
            max_speed=1.2,
            n_obstacles=8
        )

        obs, _ = env.reset()

        # Double check if the seed immediately spawned the drone in a collision state
        if env._check_collision():
            # Skip this seed if it's an unfair generation bug where a wall sits on the spawn point
            continue

        frame_stack = deque([obs] * stack_size, maxlen=stack_size)
        state = np.concatenate(list(frame_stack), axis=0)

        ep_steps = 0

        while True:
            action = agent.select_action(state)
            next_obs, reward, terminated, truncated, info = env.step(action)

            frame_stack.append(next_obs)
            state = np.concatenate(list(frame_stack), axis=0)

            ep_steps += 1

            if terminated or truncated:
                total_steps_taken += ep_steps
                if info['reached_goal']:
                    successful_goals += 1

                print(
                    f"Test Ep {ep+1:02d} (Seed {current_seed}) | Steps: {ep_steps:3d} | Reached Goal: {info['reached_goal']}")
                break

    success_rate = (successful_goals / total_eval_episodes) * 100
    avg_steps = total_steps_taken / total_eval_episodes if total_eval_episodes > 0 else 0

    print("\n--- Final Evaluation Metrics ---")
    print(f"Total Test Episodes: {total_eval_episodes}")
    print(f"Successful Navigations: {successful_goals}")
    print(f"Evaluation Success Rate: {success_rate:.2f}%")
    print(f"Average Steps to Target: {avg_steps:.1f}")


if __name__ == "__main__":
    evaluate_policy()
