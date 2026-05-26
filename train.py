import os
import torch
import numpy as np
from tqdm import tqdm
from collections import deque

import env
from env.uav_env import UAVLiDAREnv
from agents.replay_buffer import ReplayBuffer
from agents.dqn_agent import DQNAgent


def train(checkpoint_to_load=None):  # Add this parameter option
    os.makedirs("outputs/checkpoints", exist_ok=True)
    os.makedirs("outputs/logs", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    stack_size = 3
    episodes = 1500
    batch_size = 64
    target_update_freq = 10

    # Phase 1 Initial Setup: Start training with fewer obstacles!
    num_obstacles_current = 2 if checkpoint_to_load is None else 8

    env = UAVLiDAREnv(
        seed=42,
        drag=0.15,
        thrust=0.6,
        max_speed=1.2,
        n_obstacles=num_obstacles_current  # Controlled dynamically
    )

    original_obs_dim = env.observation_space.shape[0]
    stacked_state_dim = original_obs_dim * stack_size

    agent = DQNAgent(
        state_dim=stacked_state_dim,
        action_dim=env.action_space.n,
        lr=1e-3,
        gamma=0.99,
        # Start with low exploration if reloading
        epsilon=1.0 if checkpoint_to_load is None else 0.2,
        epsilon_decay=0.998,  # Slowed down decay as discussed
        epsilon_min=0.05,
        device=device,
    )

    # --- CRITICAL: WEIGHT RE-LOADING LOGIC ---

    # Debug print for obstacle count
    print(f"\n number of obstacles: {env.n_obstacles}")

    start_episode = checkpoint.get("episode", 1) + 1
    if checkpoint_to_load is not None:
        print(f"Loading weights from checkpoint: {checkpoint_to_load}")
        checkpoint = torch.load(
            checkpoint_to_load, map_location=device, weights_only=False)
        agent.q_network.load_state_dict(checkpoint["q_network"])
        agent.target_network.load_state_dict(checkpoint["q_network"])
        # Optional: pull previous epsilon if you don't want to force 0.2
        agent.epsilon = checkpoint.get("epsilon", 0.2)
        start_episode = checkpoint.get("episode", 1) + 1

    replay_buffer = ReplayBuffer(capacity=100000)
    rewards_history = []

    # Update loop index to account for start_episode
    for episode in tqdm(range(start_episode, episodes + 1)):
        # 3. Initialize Stack
        obs, _ = env.reset()
        # Create a queue that holds the last 3 observations
        # Initially, fill it with the first observation
        frame_stack = deque([obs] * stack_size, maxlen=stack_size)

        # Flatten the stack into a single 1D vector for the network
        state = np.concatenate(list(frame_stack), axis=0)

        total_reward = 0
        total_loss = 0
        loss_count = 0

        for step in range(env.max_steps):
            action = agent.select_action(state)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # 4. Update Stack and Create Next State
            frame_stack.append(next_obs)
            next_state = np.concatenate(list(frame_stack), axis=0)

            # Store the stacked vectors in the buffer
            replay_buffer.push(state, action, reward, next_state, done)

            loss = agent.train_step(replay_buffer, batch_size)

            if loss is not None:
                total_loss += loss
                loss_count += 1

            state = next_state
            total_reward += reward

            if done:
                break

        agent.decay_epsilon()

        if episode % target_update_freq == 0:
            agent.update_target_network()

        avg_loss = total_loss / loss_count if loss_count > 0 else 0.0
        rewards_history.append(total_reward)

        if episode % 10 == 0:  # Print every 10 episodes to keep logs clean
            print(
                f"Ep {episode:03d} | Reward: {total_reward:.2f} | "
                f"Loss: {avg_loss:.4f} | Eps: {agent.epsilon:.2f} | "
                f"Goal: {info['reached_goal']}"
            )

        if episode % 100 == 0:
            checkpoint_path = f"/content/drive/MyDrive/dqn_episode_{episode}.pth"
            torch.save({
                "episode": episode,
                "q_network": agent.q_network.state_dict(),
                "epsilon": agent.epsilon,
            }, checkpoint_path)

    np.save("outputs/logs/rewards_history.npy", np.array(rewards_history))


if __name__ == "__main__":
    # To start Phase 1 (Easy Map), leave it empty:
    # train(checkpoint_to_load=None)

    # To start Phase 2 (Hard Map), comment out above and use:
    train(checkpoint_to_load="outputs/checkpoints/dqn_episode_500.pth")
