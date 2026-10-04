import os
import random
import torch
import numpy as np

from collections import deque
from tqdm import tqdm

from env.uav_env_dynamic import UAVLiDARDynamicEnv
from agents.dqn_agent import DQNAgent
from agents.replay_buffer import ReplayBuffer


# ============================================================
# FIXED CURRICULUM SCHEDULE
# ============================================================

def get_obstacle_count(episode):
    """
    Fixed 4-stage curriculum:
        Episodes    1-2000 -> 2 obstacles
        Episodes 2001-4000 -> 4 obstacles
        Episodes 4001-6000 -> 6 obstacles
        Episodes 6001-8000 -> 8 obstacles
    """
    if episode <= 2000:
        return 2
    elif episode <= 4000:
        return 4
    elif episode <= 6000:
        return 6
    else:
        return 8


# ============================================================
# TRAINING
# ============================================================

def train_fixed_curriculum(seed=42):

    # --------------------------------------------------------
    # REPRODUCIBILITY
    # --------------------------------------------------------

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("FIXED-EPISODE CURRICULUM DDDQN")
    print(f"Training seed : {seed}")
    print(f"Device        : {device}")
    print("Schedule      : 2 -> 4 -> 6 -> 8")
    print("Stage length  : 2000 episodes")
    print("Criterion     : FIXED EPISODE NUMBER")
    print("=" * 70)

    # --------------------------------------------------------
    # HYPERPARAMETERS
    # --------------------------------------------------------

    episodes = 8000

    batch_size = 64
    gamma = 0.99
    lr = 1e-4

    stack_size = 3
    buffer_capacity = 50000

    target_update_frequency = 10

    # --------------------------------------------------------
    # OUTPUT DIRECTORIES
    # --------------------------------------------------------

    save_dir = os.path.join(
        "/content/drive/MyDrive/drl-uav-navigation/outputs_fixed_curriculum",
        f"seed_{seed}"
    )

    checkpoint_dir = os.path.join(save_dir, "checkpoints")

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # --------------------------------------------------------
    # INITIAL ENVIRONMENT
    # --------------------------------------------------------

    current_obstacles = 2

    env = UAVLiDARDynamicEnv(
        n_obstacles=current_obstacles,
        reward_mode="standard",
        n_lidar=256,
        seed=seed
    )

    state_dim = env.observation_space.shape[0] * stack_size
    action_dim = env.action_space.n

    print(f"Observation dimension : {env.observation_space.shape[0]}")
    print(f"Stacked state dim     : {state_dim}")
    print(f"Action dimension      : {action_dim}")
    print(f"Initial obstacles     : {current_obstacles}")

    # --------------------------------------------------------
    # AGENT
    # --------------------------------------------------------

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=lr,
        gamma=gamma,
        device=device
    )

    # --------------------------------------------------------
    # REPLAY BUFFER
    # --------------------------------------------------------

    replay_buffer = ReplayBuffer(
        capacity=buffer_capacity
    )

    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

    rewards_history = []
    loss_history = []
    success_history = []
    obstacle_history = []

    min_proximity_history = []
    total_rotation_history = []
    steps_history = []
    collision_history = []
    timeout_history = []
    epsilon_history = []

    # Overall rolling window used ONLY for monitoring.
    success_window = deque(maxlen=100)

    # Stage-specific rolling window.
    # Clear this whenever the fixed stage changes.
    stage_success_window = deque(maxlen=100)

    # --------------------------------------------------------
    # TRAINING LOOP
    # --------------------------------------------------------

    for episode in tqdm(range(1, episodes + 1)):

        # ====================================================
        # FIXED CURRICULUM CONTROLLER
        # ====================================================

        required_obstacles = get_obstacle_count(episode)

        if required_obstacles != current_obstacles:

            old_obstacles = current_obstacles
            current_obstacles = required_obstacles

            print("\n" + "=" * 70)
            print(
                f"FIXED CURRICULUM TRANSITION AT EPISODE {episode}"
            )
            print(
                f"{old_obstacles} obstacles -> "
                f"{current_obstacles} obstacles"
            )

            # Re-create exactly the same environment type,
            # changing only obstacle density.
            env = UAVLiDARDynamicEnv(
                n_obstacles=current_obstacles,
                reward_mode="standard",
                n_lidar=256,
                seed=seed
            )

            # Restore exploration after difficulty increase.
            agent.epsilon = max(agent.epsilon, 0.40)

            # Stage statistics start fresh.
            stage_success_window.clear()

            print(
                f"Epsilon restored to: {agent.epsilon:.3f}"
            )
            print("=" * 70 + "\n")

        # ====================================================
        # EPISODE RESET
        # ====================================================

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

        final_info = {}

        # ====================================================
        # EPISODE LOOP
        # ====================================================

        while True:

            action = agent.select_action(state)

            next_obs, reward, terminated, truncated, info = env.step(
                action
            )

            done = terminated or truncated

            # ------------------------------------------------
            # TELEMETRY
            # ------------------------------------------------

            if "min_lidar_distance" in info:
                episode_min_proximity = min(
                    episode_min_proximity,
                    float(info["min_lidar_distance"])
                )

            if action in [3, 4]:
                episode_total_rotation += 1

            # ------------------------------------------------
            # FRAME STACK
            # ------------------------------------------------

            frame_stack.append(next_obs)

            next_state = np.concatenate(
                list(frame_stack),
                axis=0
            )

            # ------------------------------------------------
            # REPLAY
            # ------------------------------------------------

            replay_buffer.push(
                state,
                action,
                reward,
                next_state,
                done
            )

            # ------------------------------------------------
            # TRAINING STEP
            # ------------------------------------------------

            if len(replay_buffer) >= batch_size:

                loss = agent.train_step(
                    replay_buffer,
                    batch_size
                )

                if loss is not None:
                    episode_losses.append(loss)

            state = next_state
            episode_reward += reward
            final_info = info

            if done:
                break

        # ====================================================
        # END OF EPISODE
        # ====================================================

        agent.decay_epsilon()

        if episode % target_update_frequency == 0:
            agent.update_target_network()

        # ----------------------------------------------------
        # OUTCOME
        # ----------------------------------------------------

        success = bool(
            final_info.get("reached_goal", False)
        )

        collision = bool(
            final_info.get("collision", False)
        )

        timeout = bool(
            final_info.get("timeout", False)
        )

        success_value = 1 if success else 0

        success_window.append(success_value)
        stage_success_window.append(success_value)

        avg_loss = (
            float(np.mean(episode_losses))
            if episode_losses
            else 0.0
        )

        if episode_min_proximity == float("inf"):
            episode_min_proximity = 0.0

        # ----------------------------------------------------
        # SAVE HISTORY IN MEMORY
        # ----------------------------------------------------

        rewards_history.append(episode_reward)
        loss_history.append(avg_loss)
        success_history.append(success_value)
        obstacle_history.append(current_obstacles)

        min_proximity_history.append(
            episode_min_proximity
        )

        total_rotation_history.append(
            episode_total_rotation
        )

        steps_history.append(env.steps)
        collision_history.append(1 if collision else 0)
        timeout_history.append(1 if timeout else 0)
        epsilon_history.append(agent.epsilon)

        # ----------------------------------------------------
        # MONITORING
        # ----------------------------------------------------

        rolling_sr = (
            100.0 * sum(success_window) / len(success_window)
        )

        stage_sr = (
            100.0
            * sum(stage_success_window)
            / len(stage_success_window)
        )

        if episode % 20 == 0:

            print(
                f"Ep {episode:04d} | "
                f"Obs: {current_obstacles} | "
                f"Rolling SR: {rolling_sr:5.1f}% | "
                f"Stage SR: {stage_sr:5.1f}% | "
                f"Reward: {episode_reward:8.2f} | "
                f"Loss: {avg_loss:9.4f} | "
                f"Eps: {agent.epsilon:.3f} | "
                f"d_min: {episode_min_proximity:.3f}m | "
                f"Goal: {success}"
            )

        # ====================================================
        # SAVE EVERY 100 EPISODES
        # ====================================================

        if episode % 100 == 0:

            checkpoint_path = os.path.join(
                checkpoint_dir,
                (
                    f"fixed_obs_{current_obstacles}_"
                    f"seed{seed}_ep_{episode}.pth"
                )
            )

            torch.save(
                {
                    "episode": episode,
                    "seed": seed,
                    "obstacles": current_obstacles,

                    "model_state_dict":
                        agent.q_network.state_dict(),

                    "target_state_dict":
                        agent.target_network.state_dict(),

                    "optimizer_state_dict":
                        agent.optimizer.state_dict(),

                    "epsilon": agent.epsilon,

                    "curriculum": "fixed",
                    "stage_length": 2000,
                },
                checkpoint_path
            )

            # -----------------------------------------------
            # SAVE ALL METRICS
            # -----------------------------------------------

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
                    "obstacle_history.npy"
                ),
                np.asarray(obstacle_history)
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

            np.save(
                os.path.join(
                    save_dir,
                    "steps_history.npy"
                ),
                np.asarray(steps_history)
            )

            np.save(
                os.path.join(
                    save_dir,
                    "collision_history.npy"
                ),
                np.asarray(collision_history)
            )

            np.save(
                os.path.join(
                    save_dir,
                    "timeout_history.npy"
                ),
                np.asarray(timeout_history)
            )

            np.save(
                os.path.join(
                    save_dir,
                    "epsilon_history.npy"
                ),
                np.asarray(epsilon_history)
            )

    # ========================================================
    # FINAL CHECKPOINT
    # ========================================================

    final_checkpoint = os.path.join(
        checkpoint_dir,
        f"fixed_seed{seed}_FINAL.pth"
    )

    torch.save(
        {
            "episode": episodes,
            "seed": seed,
            "obstacles": current_obstacles,

            "model_state_dict":
                agent.q_network.state_dict(),

            "target_state_dict":
                agent.target_network.state_dict(),

            "optimizer_state_dict":
                agent.optimizer.state_dict(),

            "epsilon": agent.epsilon,

            "curriculum": "fixed",
            "stage_length": 2000,
        },
        final_checkpoint
    )

    # Final metric protection save
    np.save(
        os.path.join(save_dir, "rewards_history.npy"),
        np.asarray(rewards_history)
    )

    np.save(
        os.path.join(save_dir, "loss_history.npy"),
        np.asarray(loss_history)
    )

    np.save(
        os.path.join(save_dir, "success_history.npy"),
        np.asarray(success_history)
    )

    np.save(
        os.path.join(save_dir, "obstacle_history.npy"),
        np.asarray(obstacle_history)
    )

    np.save(
        os.path.join(save_dir, "min_proximity_history.npy"),
        np.asarray(min_proximity_history)
    )

    np.save(
        os.path.join(save_dir, "total_rotation_history.npy"),
        np.asarray(total_rotation_history)
    )

    np.save(
        os.path.join(save_dir, "steps_history.npy"),
        np.asarray(steps_history)
    )

    np.save(
        os.path.join(save_dir, "collision_history.npy"),
        np.asarray(collision_history)
    )

    np.save(
        os.path.join(save_dir, "timeout_history.npy"),
        np.asarray(timeout_history)
    )

    np.save(
        os.path.join(save_dir, "epsilon_history.npy"),
        np.asarray(epsilon_history)
    )

    print("\n" + "=" * 70)
    print("FIXED CURRICULUM TRAINING COMPLETE")
    print(f"Seed             : {seed}")
    print(f"Final obstacles  : {current_obstacles}")
    print(f"Final epsilon    : {agent.epsilon:.4f}")
    print(f"Final checkpoint : {final_checkpoint}")
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    train_fixed_curriculum(seed=42)