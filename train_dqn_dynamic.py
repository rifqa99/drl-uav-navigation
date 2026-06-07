import os
import argparse
import torch
import numpy as np
from collections import deque
from tqdm import tqdm

from env.uav_env_dynamic import UAVLiDARDynamicEnv
from agents.dqn_agent import DQNAgent
from agents.replay_buffer import ReplayBuffer


def make_env(current_obstacles, reward_mode):
    return UAVLiDARDynamicEnv(
        n_obstacles=current_obstacles,
        reward_mode=reward_mode,
        seed=42
    )


def save_histories(
    save_dir,
    rewards_history,
    loss_history,
    success_history,
    obstacle_history,
    min_proximity_history,
    total_rotation_history,
    speed_history,
    omega_history,
    steps_history,
    collision_history,
    timeout_history,
    stage_sr_history
):
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
        os.path.join(save_dir, "min_proximity_history_dynamic.npy"),
        np.array(min_proximity_history)
    )
    np.save(
        os.path.join(save_dir, "total_rotation_history_dynamic.npy"),
        np.array(total_rotation_history)
    )
    np.save(
        os.path.join(save_dir, "speed_history_dynamic.npy"),
        np.array(speed_history)
    )
    np.save(
        os.path.join(save_dir, "omega_history_dynamic.npy"),
        np.array(omega_history)
    )
    np.save(
        os.path.join(save_dir, "steps_history_dynamic.npy"),
        np.array(steps_history)
    )
    np.save(
        os.path.join(save_dir, "collision_history_dynamic.npy"),
        np.array(collision_history)
    )
    np.save(
        os.path.join(save_dir, "timeout_history_dynamic.npy"),
        np.array(timeout_history)
    )
    np.save(
        os.path.join(save_dir, "stage_sr_history_dynamic.npy"),
        np.array(stage_sr_history)
    )


def train_dqn_dynamic(
    reward_mode="standard",
    checkpoint_file=None
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")
    print(f"Reward mode: {reward_mode}")

    episodes = 6000
    batch_size = 64
    gamma = 0.99
    lr = 1e-4
    stack_size = 3
    buffer_capacity = 50000
    target_update_frequency = 10

    current_obstacles = 2
    max_obstacles = 8
    curriculum_threshold = 0.70

    start_episode = 1

    save_dir = f"/content/drive/MyDrive/drl-uav-new/outputs_dynamic_{reward_mode}"
    checkpoint_dir = os.path.join(save_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    rewards_history = []
    loss_history = []
    success_history = []
    obstacle_history = []
    min_proximity_history = []
    total_rotation_history = []
    speed_history = []
    omega_history = []
    steps_history = []
    collision_history = []
    timeout_history = []
    stage_sr_history = []

    success_window = deque(maxlen=100)

    env = make_env(
        current_obstacles=current_obstacles,
        reward_mode=reward_mode
    )

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

        agent.q_network.load_state_dict(
            checkpoint["model_state_dict"]
        )

        if "target_model_state_dict" in checkpoint:
            agent.target_network.load_state_dict(
                checkpoint["target_model_state_dict"]
            )
        else:
            agent.target_network.load_state_dict(
                checkpoint["model_state_dict"]
            )

        if "optimizer_state_dict" in checkpoint:
            agent.optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )

        start_episode = checkpoint.get("episode", 0) + 1
        current_obstacles = checkpoint.get("obstacles", 2)
        agent.epsilon = checkpoint.get("epsilon", agent.epsilon)

        env = make_env(
            current_obstacles=current_obstacles,
            reward_mode=reward_mode
        )

        print(
            f"Resumed from episode {start_episode}, "
            f"obstacles={current_obstacles}"
        )
    else:
        print("Starting training from scratch.")

    print(
        "\nStarting adaptive dynamic DQN training | "
        f"Reward: {reward_mode} | "
        f"Obstacles: {current_obstacles} | "
        f"Start episode: {start_episode}\n"
    )

    for episode in tqdm(range(start_episode, episodes + 1)):

        # Curriculum update based on STAGE rolling success rate
        if len(success_window) == success_window.maxlen:
            rolling_sr = sum(success_window) / len(success_window)

            if rolling_sr >= curriculum_threshold and current_obstacles < max_obstacles:
                current_obstacles += 2
                agent.epsilon = max(agent.epsilon, 0.40)

                print("\n" + "=" * 60)
                print(
                    f"Stage cleared: rolling success rate = "
                    f"{rolling_sr * 100:.1f}%"
                )
                print(
                    f"Increasing dynamic obstacles to "
                    f"{current_obstacles}"
                )
                print("=" * 60 + "\n")

                env = make_env(
                    current_obstacles=current_obstacles,
                    reward_mode=reward_mode
                )

                success_window.clear()

        obs, _ = env.reset(seed=42 + episode)

        frame_stack = deque(
            [obs] * stack_size,
            maxlen=stack_size
        )

        state = np.concatenate(
            list(frame_stack),
            axis=0
        )

        episode_reward = 0.0
        episode_losses = []
        episode_total_rotation = 0
        episode_min_proximity = float("inf")
        episode_speeds = []
        episode_omegas = []
        episode_steps = 0

        final_info = {}

        while True:
            action = agent.select_action(state)

            if action in [3, 4]:
                episode_total_rotation += 1

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            frame_stack.append(next_obs)

            next_state = np.concatenate(
                list(frame_stack),
                axis=0
            )

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

            episode_speeds.append(
                float(info.get("speed", 0.0))
            )
            episode_omegas.append(
                abs(float(info.get("omega", 0.0)))
            )

            if "min_lidar_distance" in info:
                episode_min_proximity = min(
                    episode_min_proximity,
                    float(info["min_lidar_distance"])
                )

            if len(replay_buffer) >= batch_size:
                loss = agent.train_step(
                    replay_buffer,
                    batch_size
                )

                if loss is not None:
                    episode_losses.append(loss)

            if done:
                final_info = info
                break

        agent.decay_epsilon()

        if episode % target_update_frequency == 0:
            agent.update_target_network()

        is_success = 1 if final_info.get("reached_goal", False) else 0
        is_collision = 1 if final_info.get("collision", False) else 0
        is_timeout = 1 if (
            (not final_info.get("reached_goal", False)) and
            (not final_info.get("collision", False))
        ) else 0

        success_window.append(is_success)

        rolling_sr = (
            sum(success_window) / len(success_window) * 100
            if len(success_window) > 0
            else 0.0
        )

        if episode_min_proximity == float("inf"):
            episode_min_proximity = 0.0

        avg_loss = (
            float(np.mean(episode_losses))
            if episode_losses
            else 0.0
        )

        avg_speed = (
            float(np.mean(episode_speeds))
            if episode_speeds
            else 0.0
        )

        avg_omega = (
            float(np.mean(episode_omegas))
            if episode_omegas
            else 0.0
        )

        rewards_history.append(float(episode_reward))
        loss_history.append(avg_loss)
        success_history.append(is_success)
        obstacle_history.append(current_obstacles)
        min_proximity_history.append(float(episode_min_proximity))
        total_rotation_history.append(int(episode_total_rotation))
        speed_history.append(avg_speed)
        omega_history.append(avg_omega)
        steps_history.append(int(episode_steps))
        collision_history.append(is_collision)
        timeout_history.append(is_timeout)
        stage_sr_history.append(float(rolling_sr))

        if episode % 20 == 0:
            print(
                f"Ep {episode:04d} | "
                f"Obs: {current_obstacles} | "
                f"RollingSR: {rolling_sr:5.1f}% | "
                f"Reward: {episode_reward:8.2f} | "
                f"Steps: {episode_steps} | "
                f"Rot: {episode_total_rotation} | "
                f"Speed: {avg_speed:.3f} | "
                f"Omega: {avg_omega:.3f} | "
                f"Goal: {bool(final_info.get('reached_goal', False))} | "
                f"Eps: {agent.epsilon:.3f}"
            )

        if episode % 100 == 0:
            checkpoint_path = os.path.join(
                checkpoint_dir,
                f"dqn_dynamic_{reward_mode}_obs_"
                f"{current_obstacles}_ep_{episode}.pth"
            )

            torch.save(
                {
                    "episode": episode,
                    "reward_mode": reward_mode,
                    "obstacles": current_obstacles,
                    "model_state_dict": agent.q_network.state_dict(),
                    "target_model_state_dict": agent.target_network.state_dict(),
                    "optimizer_state_dict": agent.optimizer.state_dict(),
                    "epsilon": agent.epsilon,
                },
                checkpoint_path
            )

            save_histories(
                save_dir=save_dir,
                rewards_history=rewards_history,
                loss_history=loss_history,
                success_history=success_history,
                obstacle_history=obstacle_history,
                min_proximity_history=min_proximity_history,
                total_rotation_history=total_rotation_history,
                speed_history=speed_history,
                omega_history=omega_history,
                steps_history=steps_history,
                collision_history=collision_history,
                timeout_history=timeout_history,
                stage_sr_history=stage_sr_history
            )

            print(
                f"Saved checkpoint and histories: "
                f"{checkpoint_path}"
            )

    save_histories(
        save_dir=save_dir,
        rewards_history=rewards_history,
        loss_history=loss_history,
        success_history=success_history,
        obstacle_history=obstacle_history,
        min_proximity_history=min_proximity_history,
        total_rotation_history=total_rotation_history,
        speed_history=speed_history,
        omega_history=omega_history,
        steps_history=steps_history,
        collision_history=collision_history,
        timeout_history=timeout_history,
        stage_sr_history=stage_sr_history
    )

    print(
        "\nAdaptive dynamic training complete. "
        f"Saved outputs to: {save_dir}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--reward_mode",
        type=str,
        default="standard",
        choices=["standard", "risk_aware"]
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None
    )

    args = parser.parse_args()

    train_dqn_dynamic(
        reward_mode=args.reward_mode,
        checkpoint_file=args.checkpoint
    )