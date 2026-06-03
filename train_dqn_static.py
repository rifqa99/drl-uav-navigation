import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from collections import deque

from env.uav_env import UAVLiDAREnv
from agents.replay_buffer import ReplayBuffer
from agents.dqn_agent import DQNAgent


def train_static(
    reward_mode="risk_aware",
    checkpoint_to_load=None
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print("Reward mode:", reward_mode)

    episodes = 3000
    batch_size = 64
    gamma = 0.99
    lr = 1e-4
    stack_size = 3
    buffer_capacity = 100000
    target_update_freq = 10

    n_obstacles = 5

    save_dir = f"/content/drive/MyDrive/drl-uav-navigation/outputs_static_{reward_mode}"
    checkpoint_dir = os.path.join(save_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    env = UAVLiDAREnv(
        seed=42,
        drag=0.15,
        thrust=0.6,
        max_speed=1.2,
        n_obstacles=n_obstacles,
        reward_mode=reward_mode
    )

    state_dim = env.observation_space.shape[0] * stack_size
    action_dim = env.action_space.n

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=lr,
        gamma=gamma,
        epsilon=1.0,
        epsilon_decay=0.998,
        epsilon_min=0.05,
        device=device,
    )

    replay_buffer = ReplayBuffer(capacity=buffer_capacity)

    start_episode = 1

    rewards_history = []
    loss_history = []
    success_history = []
    min_proximity_history = []
    rotation_history = []
    speed_history = []
    omega_history = []
    steps_history = []

    if checkpoint_to_load and os.path.exists(checkpoint_to_load):
        checkpoint = torch.load(
            checkpoint_to_load,
            map_location=device,
            weights_only=False
        )

        agent.q_network.load_state_dict(checkpoint["model_state_dict"])
        agent.target_network.load_state_dict(checkpoint["target_model_state_dict"])

        if "optimizer_state_dict" in checkpoint:
            agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        agent.epsilon = checkpoint.get("epsilon", agent.epsilon)
        start_episode = checkpoint.get("episode", 0) + 1

        print(f"Resuming from episode {start_episode}")

    for episode in tqdm(range(start_episode, episodes + 1)):
        obs, _ = env.reset(seed=42 + episode)

        frame_stack = deque([obs] * stack_size, maxlen=stack_size)
        state = np.concatenate(list(frame_stack), axis=0)

        episode_reward = 0.0
        episode_losses = []
        episode_rotations = 0
        episode_min_proximity = float("inf")
        episode_speeds = []
        episode_omegas = []
        episode_steps = 0

        while True:
            action = agent.select_action(state)

            if action in [3, 4]:
                episode_rotations += 1

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            frame_stack.append(next_obs)
            next_state = np.concatenate(list(frame_stack), axis=0)

            replay_buffer.push(
                state,
                action,
                reward,
                next_state,
                done
            )

            state = next_state
            episode_reward += reward
            episode_steps += 1

            episode_speeds.append(info.get("speed", 0.0))
            episode_omegas.append(abs(info.get("omega", 0.0)))

            if "min_lidar_distance" in info:
                episode_min_proximity = min(
                    episode_min_proximity,
                    info["min_lidar_distance"]
                )

            if len(replay_buffer) >= batch_size:
                loss = agent.train_step(replay_buffer, batch_size)
                if loss is not None:
                    episode_losses.append(loss)

            if done:
                break

        agent.decay_epsilon()

        if episode % target_update_freq == 0:
            agent.update_target_network()

        is_success = 1 if info.get("reached_goal", False) else 0

        rewards_history.append(float(episode_reward))
        loss_history.append(float(np.mean(episode_losses)) if episode_losses else 0.0)
        success_history.append(is_success)
        rotation_history.append(int(episode_rotations))
        steps_history.append(int(episode_steps))

        if episode_min_proximity == float("inf"):
            episode_min_proximity = 0.0

        min_proximity_history.append(float(episode_min_proximity))
        speed_history.append(float(np.mean(episode_speeds)) if episode_speeds else 0.0)
        omega_history.append(float(np.mean(episode_omegas)) if episode_omegas else 0.0)

        if episode % 20 == 0:
            recent_sr = np.mean(success_history[-100:]) * 100

            print(
                f"Ep {episode:04d} | "
                f"Mode: {reward_mode} | "
                f"Reward: {episode_reward:8.2f} | "
                f"SR100: {recent_sr:5.1f}% | "
                f"Loss: {loss_history[-1]:.4f} | "
                f"Steps: {steps_history[-1]} | "
                f"d_min: {min_proximity_history[-1]:.3f}m | "
                f"Rot: {rotation_history[-1]} | "
                f"Speed: {speed_history[-1]:.3f} | "
                f"Omega: {omega_history[-1]:.3f} | "
                f"Goal: {bool(info.get('reached_goal', False))} | "
                f"Eps: {agent.epsilon:.3f}"
            )

        if episode % 100 == 0:
            checkpoint_path = os.path.join(
                checkpoint_dir,
                f"dqn_static_{reward_mode}_ep_{episode}.pth"
            )

            torch.save(
                {
                    "episode": episode,
                    "reward_mode": reward_mode,
                    "model_state_dict": agent.q_network.state_dict(),
                    "target_model_state_dict": agent.target_network.state_dict(),
                    "optimizer_state_dict": agent.optimizer.state_dict(),
                    "epsilon": agent.epsilon,
                    "n_obstacles": n_obstacles,
                },
                checkpoint_path
            )

            np.save(os.path.join(save_dir, "rewards_history.npy"), np.array(rewards_history))
            np.save(os.path.join(save_dir, "loss_history.npy"), np.array(loss_history))
            np.save(os.path.join(save_dir, "success_history.npy"), np.array(success_history))
            np.save(os.path.join(save_dir, "min_proximity_history.npy"), np.array(min_proximity_history))
            np.save(os.path.join(save_dir, "rotation_history.npy"), np.array(rotation_history))
            np.save(os.path.join(save_dir, "speed_history.npy"), np.array(speed_history))
            np.save(os.path.join(save_dir, "omega_history.npy"), np.array(omega_history))
            np.save(os.path.join(save_dir, "steps_history.npy"), np.array(steps_history))

            print(f"Saved checkpoint: {checkpoint_path}")

    print(f"\nStatic training complete for reward_mode={reward_mode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--reward_mode",
        type=str,
        default="risk_aware",
        choices=["standard", "risk_aware"]
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None
    )

    args = parser.parse_args()

    train_static(
        reward_mode=args.reward_mode,
        checkpoint_to_load=args.checkpoint
    )