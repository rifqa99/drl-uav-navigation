import os
import torch
import numpy as np
import random
from collections import deque
from tqdm import tqdm

from env.uav_env_dynamic import UAVLiDARDynamicEnv
# Pulling directly from your verified codebase
from agents.dqn_agent import DQNAgent


def train_dqn_dynamic():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- Hyperparameters ---
    episodes = 2000
    batch_size = 64
    gamma = 0.99
    lr = 1e-4
    stack_size = 3
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay = 0.996

    # Clean separate outputs directory for your thesis plots
    save_dir = "/content/drive/MyDrive/drl-uav-navigation/outputs_dynamic"
    checkpoint_dir = os.path.join(save_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize Environment directly with 8 dynamic hazards
    env = UAVLiDARDynamicEnv(n_obstacles=8)

    state_dim = env.observation_space.shape[0] * stack_size
    action_dim = env.action_space.n

    # Instantiate your stable DQN Agent core
    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim,
                     lr=lr, gamma=gamma, device=device)

    rewards_history = []
    success_window = deque(maxlen=100)
    epsilon = epsilon_start

    print(
        f"Starting Dynamic Obstacle DQN Training across {episodes} episodes...")

    for episode in tqdm(range(1, episodes + 1)):
        obs, _ = env.reset()
        frame_stack = deque([obs] * stack_size, maxlen=stack_size)
        state = np.concatenate(list(frame_stack), axis=0)

        episode_reward = 0

        while True:
            # Epsilon-greedy exploration block
            if random.random() < epsilon:
                action = random.randint(0, action_dim - 1)
            else:
                action = agent.act(state)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            frame_stack.append(next_obs)
            next_state = np.concatenate(list(frame_stack), axis=0)

            # Push transition into experience buffer memory
            agent.remember(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            # Train network once memory layout initializes cleanly
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

            if done:
                rewards_history.append(episode_reward)
                success_window.append(1 if info['reached_goal'] else 0)
                break

        # Decay exploration factor
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # Print detailed diagnostics metrics block
        if episode % 50 == 0:
            rolling_sr = (sum(success_window) / len(success_window)
                          ) * 100 if success_window else 0.0
            print(
                f"| Ep {episode:04d} | Reward: {episode_reward:7.2f} | SR: {rolling_sr:5.1f}% | Eps: {epsilon:.3f} | Goal: {info['reached_goal']}")

        # Save weights history
        if episode % 200 == 0:
            checkpoint_path = os.path.join(
                checkpoint_dir, f"dqn_dynamic_{episode}.pth")
            torch.save({
                'episode': episode,
                'model_state_dict': agent.model.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
            }, checkpoint_path)
            np.save(os.path.join(save_dir, "rewards_history_dynamic.npy"),
                    np.array(rewards_history))

    print("\nDynamic Training Completed Successfully.")


if __name__ == "__main__":
    train_dqn_dynamic()
