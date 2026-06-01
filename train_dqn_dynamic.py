import os
import torch
import numpy as np
from collections import deque
from tqdm import tqdm

from env.uav_env_dynamic import UAVLiDARDynamicEnv
from agents.dqn_agent import DQNAgent
# Import your verified buffer module
from agents.replay_buffer import ReplayBuffer


def train_dqn_dynamic():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- Hyperparameters ---
    episodes = 2000
    batch_size = 64
    gamma = 0.99
    lr = 1e-4
    stack_size = 3
    buffer_capacity = 50000
    target_update_frequency = 10  # Synchronize target network weights every 10 episodes

    # Set up dedicated output directories on Google Drive
    save_dir = "/content/drive/MyDrive/drl-uav-navigation/outputs_dynamic"
    checkpoint_dir = os.path.join(save_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize environment with 8 moving hazards
    env = UAVLiDARDynamicEnv(n_obstacles=8)

    state_dim = env.observation_space.shape[0] * stack_size
    action_dim = env.action_space.n

    # Instantiate Buffer and Agent matching your core class structures
    replay_buffer = ReplayBuffer(capacity=buffer_capacity)
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=lr,
        gamma=gamma,
        device=device
    )

    rewards_history = []
    success_window = deque(maxlen=100)

    print(
        f"Starting Dynamic Obstacle DQN Training across {episodes} episodes...")

    for episode in tqdm(range(1, episodes + 1)):
        obs, _ = env.reset()
        frame_stack = deque([obs] * stack_size, maxlen=stack_size)
        state = np.concatenate(list(frame_stack), axis=0)

        episode_reward = 0

        while True:
            # FIXED: Your agent handles its own epsilon selection inside select_action!
            action = agent.select_action(state)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            frame_stack.append(next_obs)
            next_state = np.concatenate(list(frame_stack), axis=0)

            # FIXED: Push data directly into your external replay_buffer instance
            replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            # FIXED: Run optimization steps using your train_step function
            agent.train_step(replay_buffer, batch_size)

            if done:
                rewards_history.append(episode_reward)
                success_window.append(1 if info['reached_goal'] else 0)
                break

        # FIXED: Call your agent's built-in parameter decay handler
        agent.decay_epsilon()

        # FIXED: Call target network updates at set episode milestones
        if episode % target_update_frequency == 0:
            agent.update_target_network()

        # Progress Monitoring Logs
        if episode % 50 == 0:
            rolling_sr = (sum(success_window) / len(success_window)
                          ) * 100 if success_window else 0.0
            print(
                f"| Ep {episode:04d} | Reward: {episode_reward:7.2f} | SR: {rolling_sr:5.1f}% | Eps: {agent.epsilon:.3f} | Goal: {info['reached_goal']}")

        # Save Structural State Checkpoints
        if episode % 200 == 0:
            checkpoint_path = os.path.join(
                checkpoint_dir, f"dqn_dynamic_{episode}.pth")
            torch.save({
                'episode': episode,
                'model_state_dict': agent.q_network.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
            }, checkpoint_path)
            np.save(os.path.join(save_dir, "rewards_history_dynamic.npy"),
                    np.array(rewards_history))

    print("\nDynamic Training Completed Successfully.")


if __name__ == "__main__":
    train_dqn_dynamic()
