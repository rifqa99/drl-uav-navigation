import os
import random
import torch
import numpy as np

from collections import deque
from tqdm import tqdm

from env.uav_env_dynamic import UAVLiDARDynamicEnv
from agents.dqn_agent import DQNAgent
from agents.replay_buffer import ReplayBuffer


def train_direct8(seed=42):

    # ============================================================
    # REPRODUCIBILITY
    # ============================================================

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("DIRECT-8 DDDQN BASELINE")
    print(f"Training seed : {seed}")
    print(f"Device        : {device}")
    print("Obstacles     : 8 from Episode 1")
    print("Curriculum    : NONE")
    print("=" * 70)

    # ============================================================
    # HYPERPARAMETERS
    # ============================================================

    episodes = 8000

    batch_size = 64
    gamma = 0.99
    lr = 1e-4

    stack_size = 3
    buffer_capacity = 50000

    target_update_frequency = 10

    # ============================================================
    # OUTPUT DIRECTORIES
    # ============================================================

    save_dir = os.path.join(
        "/content/drive/MyDrive/drl-uav-navigation/outputs_direct8_fixed",
        f"seed_{seed}"
    )

    checkpoint_dir = os.path.join(save_dir, "checkpoints")

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ============================================================
    # ENVIRONMENT
    # ============================================================

    # IMPORTANT:
    # Direct baseline always contains 8 dynamic obstacles.
    env = UAVLiDARDynamicEnv(
        n_obstacles=8,
        reward_mode="standard",
        n_lidar=256,
        seed=seed
    )

    state_dim = env.observation_space.shape[0] * stack_size
    action_dim = env.action_space.n

    print(f"Observation dimension : {env.observation_space.shape[0]}")
    print(f"Stacked state dim     : {state_dim}")
    print(f"Action dimension      : {action_dim}")
    print(f"Number of obstacles   : {env.n_obstacles}")

    # ============================================================
    # AGENT
    # ============================================================

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=lr,
        gamma=gamma,
        device=device
    )

    # ============================================================
    # REPLAY BUFFER
    # ============================================================

    replay_buffer = ReplayBuffer(
        capacity=buffer_capacity
    )

    # ============================================================
    # METRIC STORAGE
    # ============================================================

    rewards_history = []
    loss_history = []
    success_history = []

    min_proximity_history = []
    total_rotation_history = []

    # Used ONLY for monitoring.
    # It does NOT affect training.
    success_window = deque(maxlen=100)

    # ============================================================
    # TRAINING
    # ============================================================

    for episode in tqdm(range(1, episodes + 1)):

        obs, _ = env.reset(seed=seed + episode)   

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

        episode_min_proximity = float("inf")
        episode_total_rotation = 0

        # --------------------------------------------------------
        # EPISODE
        # --------------------------------------------------------

        for step in range(env.max_steps):

            action = agent.select_action(state)

            next_obs, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated

            # ----------------------------------------------------
            # METRICS
            # ----------------------------------------------------

            if "min_lidar_distance" in info:
                episode_min_proximity = min(
                    episode_min_proximity,
                    float(info["min_lidar_distance"])
                )

            if action in [3, 4]:
                episode_total_rotation += 1


            # ----------------------------------------------------
            # FRAME STACK
            # ----------------------------------------------------

            frame_stack.append(next_obs)

            next_state = np.concatenate(
                list(frame_stack),
                axis=0
            )

            # ----------------------------------------------------
            # REPLAY BUFFER
            # ----------------------------------------------------
            replay_buffer.push(
                state,
                action,
                reward,
                next_state,
                done
            )

            episode_reward += reward

            # ----------------------------------------------------
            # NETWORK UPDATE
            # ----------------------------------------------------

            if len(replay_buffer) >= batch_size:

                loss = agent.train_step(
                    replay_buffer,
                    batch_size
                )

                if loss is not None:
                    episode_losses.append(loss)

            if done:
                break

        # ========================================================
        # END EPISODE
        # ========================================================

        agent.decay_epsilon()

        if episode % target_update_frequency == 0:
            agent.update_target_network()

        reached_goal = bool(
            info.get("reached_goal", False)
        )

        is_success = int(reached_goal)

        success_history.append(is_success)
        success_window.append(is_success)

        rewards_history.append(
            episode_reward
        )

        avg_loss = (
            float(np.mean(episode_losses))
            if episode_losses
            else 0.0
        )

        loss_history.append(avg_loss)

        if episode_min_proximity == float("inf"):
            episode_min_proximity = 0.0

        min_proximity_history.append(
            episode_min_proximity
        )

        total_rotation_history.append(
            episode_total_rotation
        )

        # ========================================================
        # MONITORING
        # ========================================================

        if episode % 20 == 0:

            rolling_sr = (
                100.0 * np.mean(success_window)
                if success_window
                else 0.0
            )

            print(
                f"\nEp {episode:04d} | "
                f"Obs: 8 | "
                f"Rolling SR: {rolling_sr:5.1f}% | "
                f"Reward: {episode_reward:8.2f} | "
                f"Loss: {avg_loss:.4f} | "
                f"Eps: {agent.epsilon:.3f} | "
                f"d_min: {episode_min_proximity:.3f} m | "
                f"Goal: {reached_goal}"
            )

        # ========================================================
        # SAVE EVERY 100 EPISODES
        # ========================================================

        if episode % 100 == 0:

            checkpoint_path = os.path.join(
                checkpoint_dir,
                f"direct8_seed{seed}_ep{episode}.pth"
            )

            torch.save(
                {
                    "episode": episode,
                    "seed": seed,
                    "obstacles": 8,

                    "model_state_dict":
                        agent.q_network.state_dict(),

                    "target_state_dict":
                        agent.target_network.state_dict(),

                    "optimizer_state_dict":
                        agent.optimizer.state_dict(),

                    "epsilon":
                        agent.epsilon,
                },
                checkpoint_path
            )

            np.save(
                os.path.join(
                    save_dir,
                    "rewards_history.npy"
                ),
                np.asarray(rewards_history)
            )

            np.save(
                os.path.join(
                    save_dir,
                    "loss_history.npy"
                ),
                np.asarray(loss_history)
            )

            np.save(
                os.path.join(
                    save_dir,
                    "success_history.npy"
                ),
                np.asarray(success_history)
            )

            np.save(
                os.path.join(
                    save_dir,
                    "min_proximity_history.npy"
                ),
                np.asarray(min_proximity_history)
            )

            np.save(
                os.path.join(
                    save_dir,
                    "total_rotation_history.npy"
                ),
                np.asarray(total_rotation_history)
            )

            print(
                f"\nSaved Direct-8 checkpoint: "
                f"Episode {episode}"
            )

    # ============================================================
    # FINAL SAVE
    # ============================================================

    final_path = os.path.join(
        checkpoint_dir,
        f"direct8_seed{seed}_FINAL.pth"
    )

    torch.save(
        {
            "episode": episodes,
            "seed": seed,
            "obstacles": 8,

            "model_state_dict":
                agent.q_network.state_dict(),

            "target_state_dict":
                agent.target_network.state_dict(),

            "optimizer_state_dict":
                agent.optimizer.state_dict(),

            "epsilon":
                agent.epsilon,
        },
        final_path
    )

    print("\n" + "=" * 70)
    print("DIRECT-8 TRAINING COMPLETE")
    print(f"Final model: {final_path}")
    print("=" * 70)


if __name__ == "__main__":
    train_direct8(seed=42)