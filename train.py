import os
import torch
import numpy as np

from env.uav_env import UAVLiDAREnv
from agents.replay_buffer import ReplayBuffer
from agents.dqn_agent import DQNAgent


def train():
    os.makedirs("outputs/checkpoints", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    env = UAVLiDAREnv(
        seed=42,
        drag=0.15,
        thrust=0.6,
        max_speed=1.2,
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=1e-3,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.05,
        device=device,
    )

    replay_buffer = ReplayBuffer(capacity=100000)

    episodes = 500
    batch_size = 64
    target_update_freq = 10

    rewards_history = []

    for episode in range(1, episodes + 1):
        state, _ = env.reset()
        total_reward = 0
        total_loss = 0
        loss_count = 0

        for step in range(env.max_steps):
            action = agent.select_action(state)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

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

        print(
            f"Episode {episode:04d} | "
            f"Reward: {total_reward:.2f} | "
            f"Loss: {avg_loss:.4f} | "
            f"Epsilon: {agent.epsilon:.3f} | "
            f"Success: {info['reached_goal']} | "
            f"Collision: {info['collision']}"
        )

        if episode % 100 == 0:
            checkpoint_path = f"outputs/checkpoints/dqn_episode_{episode}.pth"

            torch.save(
                {
                    "episode": episode,
                    "q_network": agent.q_network.state_dict(),
                    "target_network": agent.target_network.state_dict(),
                    "optimizer": agent.optimizer.state_dict(),
                    "epsilon": agent.epsilon,
                    "rewards_history": rewards_history,
                },
                checkpoint_path,
            )

            print("Saved:", checkpoint_path)

    np.save("outputs/logs/rewards_history.npy", np.array(rewards_history))


if __name__ == "__main__":
    train()
