import os
import torch
import numpy as np
from collections import deque
from tqdm import tqdm

from env.uav_env_dynamic import UAVLiDARDynamicEnv
from agents.dqn_agent import DQNAgent
from agents.replay_buffer import ReplayBuffer


def train_dqn_dynamic_adaptive_colab(checkpoint_file=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    episodes = 6000
    batch_size = 64
    gamma = 0.99
    lr = 1e-4
    stack_size = 3
    buffer_capacity = 50000
    target_update_frequency = 10

    save_dir = "/content/drive/MyDrive/drl-uav-navigation/outputs_dynamic_risk_aware"
    checkpoint_dir = os.path.join(save_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    current_obstacles = 2
    start_episode = 1

    rewards_history = []
    loss_history = []
    success_history = []
    obstacle_history = []
    min_proximity_history = []
    total_rotation_history = []

    success_window = deque(maxlen=100)

    env = UAVLiDARDynamicEnv(n_obstacles=current_obstacles)

    state_dim = env.observation_space.shape[0] * stack_size
    action_dim = env.action_space.n

    replay_buffer = ReplayBuffer(capacity=buffer_capacity)

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=lr,
        gamma=gamma,
        device=device,
    )

    if checkpoint_file and os.path.exists(checkpoint_file):
        print(f"Loading checkpoint: {checkpoint_file}")

        checkpoint = torch.load(
            checkpoint_file,
            map_location=device,
            weights_only=False
        )

        agent.q_network.load_state_dict(checkpoint["model_state_dict"])
        agent.target_network.load_state_dict(checkpoint["model_state_dict"])

        if "optimizer_state_dict" in checkpoint:
            agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        start_episode = checkpoint.get("episode", 0) + 1
        current_obstacles = checkpoint.get("obstacles", 2)

        env = UAVLiDARDynamicEnv(n_obstacles=current_obstacles)

        history_files = {
            "rewards_history_dynamic.npy": rewards_history,
            "loss_history_dynamic.npy": loss_history,
            "success_history_dynamic.npy": success_history,
            "obstacle_history_dynamic.npy": obstacle_history,
            "min_proximity_history.npy": min_proximity_history,
            "total_rotation_history.npy": total_rotation_history,
        }

        for filename, target_list in history_files.items():
            path = os.path.join(save_dir, filename)
            if os.path.exists(path):
                target_list.extend(list(np.load(path)))

        agent.epsilon = agent.epsilon_min
        print(f"Resumed from episode {start_episode}, obstacles={current_obstacles}")

    else:
        print("Starting training from scratch.")

    print(
        f"\nStarting adaptive dynamic DQN training | "
        f"Obstacles: {current_obstacles} | Start episode: {start_episode}\n"
    )

    for episode in tqdm(range(start_episode, episodes + 1)):

        # Curriculum update
        if len(success_window) >= 50:
            rolling_sr = sum(success_window) / len(success_window)

            if rolling_sr >= 0.70 and current_obstacles < 8:
                current_obstacles += 2

                print("\n" + "=" * 60)
                print(f"Stage cleared: rolling success rate = {rolling_sr * 100:.1f}%")
                print(f"Increasing dynamic obstacles to {current_obstacles}")
                print("=" * 60 + "\n")

                env = UAVLiDARDynamicEnv(n_obstacles=current_obstacles)
                success_window.clear()

        obs, _ = env.reset()

        frame_stack = deque([obs] * stack_size, maxlen=stack_size)
        state = np.concatenate(list(frame_stack), axis=0)

        episode_reward = 0.0
        episode_losses = []
        episode_total_rotation = 0
        episode_min_proximity = float("inf")

        while True:
            action = agent.select_action(state)

            # Count rotation actions INSIDE the step loop
            if action in [3, 4]:
                episode_total_rotation += 1

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
                is_success = 1 if info.get("reached_goal", False) else 0

                rewards_history.append(float(episode_reward))
                success_history.append(is_success)
                success_window.append(is_success)
                obstacle_history.append(current_obstacles)

                avg_loss = float(np.mean(episode_losses)) if episode_losses else 0.0
                loss_history.append(avg_loss)

                if episode_min_proximity == float("inf"):
                    episode_min_proximity = 0.0

                min_proximity_history.append(float(episode_min_proximity))
                total_rotation_history.append(int(episode_total_rotation))

                break

        agent.decay_epsilon()

        if episode % target_update_frequency == 0:
            agent.update_target_network()

        if episode % 20 == 0:
            current_sr = (
                sum(success_window) / len(success_window) * 100
                if len(success_window) > 0
                else 0.0
            )

            print(
                f"Ep {episode:04d} | "
                f"Obs: {current_obstacles} | "
                f"Rolling SR: {current_sr:5.1f}% | "
                f"Reward: {episode_reward:8.2f} | "
                f"Loss: {loss_history[-1]:.4f} | "
                f"d_min: {min_proximity_history[-1]:.3f}m | "
                f"Rot: {total_rotation_history[-1]} | "
                f"Goal: {bool(info.get('reached_goal', False))} | "
                f"Eps: {agent.epsilon:.3f}"
            )

        if episode % 100 == 0:
            checkpoint_path = os.path.join(
                checkpoint_dir,
                f"dqn_adaptive_obs_{current_obstacles}_ep_{episode}.pth"
            )

            torch.save(
                {
                    "episode": episode,
                    "obstacles": current_obstacles,
                    "model_state_dict": agent.q_network.state_dict(),
                    "target_model_state_dict": agent.target_network.state_dict(),
                    "optimizer_state_dict": agent.optimizer.state_dict(),
                    "epsilon": agent.epsilon,
                },
                checkpoint_path
            )

            np.save(
                os.path.join(save_dir, "rewards_history_dynamic.npy"),
                np.array(rewards_history)
            )
            np.save(
                os.path.join(save_dir, "loss_history_dynamic.npy"),
                np.array(loss_history)
            )
            np.save(
                os.path.join(save_dir, "success_history_dynamic.npy"),
                np.array(success_history)
            )
            np.save(
                os.path.join(save_dir, "obstacle_history_dynamic.npy"),
                np.array(obstacle_history)
            )
            np.save(
                os.path.join(save_dir, "min_proximity_history.npy"),
                np.array(min_proximity_history)
            )
            np.save(
                os.path.join(save_dir, "total_rotation_history.npy"),
                np.array(total_rotation_history)
            )

            print(f"Saved checkpoint: {checkpoint_path}")

    print("\nAdaptive training complete.")


if __name__ == "__main__":
    train_dqn_dynamic_adaptive_colab(checkpoint_file=None)