import os
import torch
import numpy as np
from tqdm import tqdm
from collections import deque

import env
from env.uav_env import UAVLiDAREnv
from agents.replay_buffer import ReplayBuffer
from agents.dqn_agent import DQNAgent


def train(checkpoint_to_load=None):
    os.makedirs("outputs/checkpoints", exist_ok=True)
    os.makedirs("outputs/logs", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    stack_size = 3

    # 1. FIXED: Set your ultimate final total target here (e.g., 3000 episodes)
    episodes = 3000
    batch_size = 64
    target_update_freq = 10

    num_obstacles_current = 2 if checkpoint_to_load is None else 8

    env = UAVLiDAREnv(
        seed=42,
        drag=0.15,
        thrust=0.6,
        max_speed=1.2,
        n_obstacles=num_obstacles_current
    )

    original_obs_dim = env.observation_space.shape[0]
    stacked_state_dim = original_obs_dim * stack_size

    agent = DQNAgent(
        state_dim=stacked_state_dim,
        action_dim=env.action_space.n,
        lr=1e-3,
        gamma=0.99,
        epsilon=1.0 if checkpoint_to_load is None else 0.2,
        epsilon_decay=0.998,
        epsilon_min=0.05,
        device=device,
    )

    # 2. FIXED: Core variables initialization sequence
    start_episode = 1
    rewards_history = []  # Will be preserved or loaded from Drive below

    print(f"\n Number of obstacles initialized: {env.n_obstacles}")

    if checkpoint_to_load is not None:
        print(f"Loading weights from checkpoint: {checkpoint_to_load}")
        checkpoint = torch.load(
            checkpoint_to_load, map_location=device, weights_only=False)
        agent.q_network.load_state_dict(checkpoint["q_network"])
        agent.target_network.load_state_dict(checkpoint["q_network"])

        agent.epsilon = checkpoint.get("epsilon", 0.2)
        start_episode = checkpoint.get("episode", 1) + 1
        print(f"Resuming training directly from Episode: {start_episode}")

        # --- NEW: LOAD PAST HISTORY ARRAY FROM DRIVE ---
        drive_history_path = "/content/drive/MyDrive/drl-uav-navigation/outputs/rewards_history.npy"
        if os.path.exists(drive_history_path):
            rewards_history = list(np.load(drive_history_path))
            print(
                f"Successfully loaded {len(rewards_history)} steps of past reward history from Drive!")
        else:
            print("Warning: Checkpoint loaded, but no previous rewards_history.npy found on Drive. Starting history fresh.")

    replay_buffer = ReplayBuffer(capacity=100000)

    # Update loop index dynamically to target maximum range
    for episode in tqdm(range(start_episode, episodes + 1)):
        obs, _ = env.reset()
        frame_stack = deque([obs] * stack_size, maxlen=stack_size)
        state = np.concatenate(list(frame_stack), axis=0)

        total_reward = 0
        total_loss = 0
        loss_count = 0

        for step in range(env.max_steps):
            action = agent.select_action(state)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            frame_stack.append(next_obs)
            next_state = np.concatenate(list(frame_stack), axis=0)

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

        # Append latest metrics to your tracking list
        rewards_history.append(total_reward)

        if episode % 10 == 0:
            print(
                f"Ep {episode:03d} | Reward: {total_reward:.2f} | "
                f"Loss: {avg_loss:.4f} | Eps: {agent.epsilon:.2f} | "
                f"Goal: {info['reached_goal']}"
            )

        # --- FIXED: SAVE BOTH WEIGHTS AND ACCUMULATED LOGS TO DRIVE EVERY 100 EPISODES ---
        if episode % 100 == 0:
            checkpoint_path = f"/content/drive/MyDrive/drl-uav-navigation/outputs/checkpoints/dqn_episode_{episode}.pth"
            torch.save({
                "episode": episode,
                "q_network": agent.q_network.state_dict(),
                "epsilon": agent.epsilon,
            }, checkpoint_path)

            # Save history incrementally so it's always safe on your Drive
            np.save("/content/drive/MyDrive/drl-uav-navigation/outputs/rewards_history.npy",
                    np.array(rewards_history))
            print(
                f"-> Saved Checkpoint {episode} and complete reward history up to Drive.")

    # Final backup save when entire pipeline wraps up completely
    np.save("/content/drive/MyDrive/drl-uav-navigation/outputs/rewards_history.npy",
            np.array(rewards_history))
    print("Training finished successfully. Final history file secured on Google Drive!")


if __name__ == "__main__":
    # CRITICAL: Since you are changing environment dimensions (adding theta/omega),
    # remember that you cannot load your old pre-fix episode 500 weights.
    # Start fresh here to build a completely stable, fully compatible (207, 5) shape baseline!
    train(checkpoint_to_load=None)

    # Future Use (Once your new run saves fresh compatible checkpoints):
    train(checkpoint_to_load="/content/drive/MyDrive/drl-uav-navigation/outputs/checkpoints/dqn_episode_1500.pth")
