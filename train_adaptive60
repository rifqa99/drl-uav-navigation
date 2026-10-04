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
# ADAPTIVE CURRICULUM — 60% THRESHOLD
# ============================================================

def train_adaptive60(seed=42):

    # ========================================================
    # REPRODUCIBILITY
    # ========================================================

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ========================================================
    # HYPERPARAMETERS
    # ========================================================

    episodes = 8000

    batch_size = 64
    gamma = 0.99
    lr = 1e-4

    stack_size = 3
    buffer_capacity = 50000

    target_update_frequency = 10

    curriculum_threshold = 0.60
    success_window_size = 100

    max_obstacles = 8

    # ========================================================
    # OUTPUT
    # ========================================================

    save_dir = os.path.join(
        "/content/drive/MyDrive/drl-uav-navigation/"
        "outputs_adaptive60",
        f"seed_{seed}"
    )

    checkpoint_dir = os.path.join(
        save_dir,
        "checkpoints"
    )

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ========================================================
    # HEADER
    # ========================================================

    print("=" * 70)
    print("ADAPTIVE CURRICULUM DDDQN — THRESHOLD 60%")
    print(f"Training seed       : {seed}")
    print(f"Device              : {device}")
    print("Progression         : 2 -> 4 -> 6 -> 8")
    print("Window              : 100 episodes")
    print("Threshold           : 60%")
    print("Total episodes      : 8000")
    print("=" * 70)

    # ========================================================
    # INITIAL ENVIRONMENT
    # ========================================================

    current_obstacles = 2

    env = UAVLiDARDynamicEnv(
        n_obstacles=current_obstacles,
        reward_mode="standard",
        n_lidar=256,
        seed=seed
    )

    state_dim = (
        env.observation_space.shape[0]
        * stack_size
    )

    action_dim = env.action_space.n

    print(
        f"Observation dimension : "
        f"{env.observation_space.shape[0]}"
    )

    print(
        f"Stacked state dim     : "
        f"{state_dim}"
    )

    print(
        f"Action dimension      : "
        f"{action_dim}"
    )

    print(
        f"Initial obstacles     : "
        f"{current_obstacles}"
    )

    # ========================================================
    # AGENT
    # ========================================================

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=lr,
        gamma=gamma,
        device=device
    )

    # ========================================================
    # REPLAY BUFFER
    # ========================================================

    replay_buffer = ReplayBuffer(
        capacity=buffer_capacity
    )

    # ========================================================
    # HISTORIES
    # ========================================================

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

    # Curriculum transition episodes
    transition_history = []

    # IMPORTANT:
    # This is the stage-specific rolling window.
    # It is cleared after each curriculum transition.
    success_window = deque(
        maxlen=success_window_size
    )

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for episode in tqdm(
        range(1, episodes + 1)
    ):

        # ====================================================
        # ADAPTIVE CURRICULUM CONTROLLER
        # ====================================================

        if (
            len(success_window)
            == success_window_size
        ):

            rolling_sr = (
                sum(success_window)
                / len(success_window)
            )

            if (
                rolling_sr
                >= curriculum_threshold
                and current_obstacles
                < max_obstacles
            ):

                old_obstacles = current_obstacles
                current_obstacles += 2

                print("\n" + "=" * 70)

                print(
                    f"ADAPTIVE CURRICULUM "
                    f"TRANSITION AT EPISODE "
                    f"{episode}"
                )

                print(
                    f"Rolling SR: "
                    f"{rolling_sr * 100:.1f}%"
                )

                print(
                    f"{old_obstacles} obstacles "
                    f"-> {current_obstacles} obstacles"
                )

                # --------------------------------------------
                # Re-create environment changing ONLY density
                # --------------------------------------------

                env = UAVLiDARDynamicEnv(
                    n_obstacles=current_obstacles,
                    reward_mode="standard",
                    n_lidar=256,
                    seed=seed
                )

                # --------------------------------------------
                # Restore exploration
                # --------------------------------------------

                agent.epsilon = max(
                    agent.epsilon,
                    0.40
                )

                # --------------------------------------------
                # Record transition
                # --------------------------------------------

                transition_history.append(
                    {
                        "episode": episode,
                        "from_obstacles":
                            old_obstacles,
                        "to_obstacles":
                            current_obstacles,
                        "rolling_sr":
                            rolling_sr
                    }
                )

                # New stage gets a fresh window
                success_window.clear()

                print(
                    f"Epsilon restored to: "
                    f"{agent.epsilon:.3f}"
                )

                print("=" * 70 + "\n")

        # ====================================================
        # EPISODE RESET
        # ====================================================

        obs, _ = env.reset(
            seed=seed + episode
        )

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

            action = agent.select_action(
                state
            )

            (
                next_obs,
                reward,
                terminated,
                truncated,
                info
            ) = env.step(action)

            done = (
                terminated
                or truncated
            )

            # =================================================
            # TELEMETRY
            # =================================================

            if "min_lidar_distance" in info:

                episode_min_proximity = min(
                    episode_min_proximity,
                    float(
                        info[
                            "min_lidar_distance"
                        ]
                    )
                )

            if action in [3, 4]:
                episode_total_rotation += 1

            # =================================================
            # FRAME STACK
            # =================================================

            frame_stack.append(
                next_obs
            )

            next_state = np.concatenate(
                list(frame_stack),
                axis=0
            )

            # =================================================
            # REPLAY
            # =================================================

            replay_buffer.push(
                state,
                action,
                reward,
                next_state,
                done
            )

            # =================================================
            # TRAINING STEP
            # =================================================

            if (
                len(replay_buffer)
                >= batch_size
            ):

                loss = agent.train_step(
                    replay_buffer,
                    batch_size
                )

                if loss is not None:
                    episode_losses.append(
                        loss
                    )

            state = next_state
            episode_reward += reward
            final_info = info

            if done:
                break

        # ====================================================
        # END OF EPISODE
        # ====================================================

        agent.decay_epsilon()

        if (
            episode
            % target_update_frequency
            == 0
        ):
            agent.update_target_network()

        # ====================================================
        # OUTCOME
        # ====================================================

        success = bool(
            final_info.get(
                "reached_goal",
                False
            )
        )

        collision = bool(
            final_info.get(
                "collision",
                False
            )
        )

        timeout = bool(
            final_info.get(
                "timeout",
                False
            )
        )

        success_value = (
            1 if success else 0
        )

        # Stage-specific curriculum window
        success_window.append(
            success_value
        )

        avg_loss = (
            float(
                np.mean(
                    episode_losses
                )
            )
            if episode_losses
            else 0.0
        )

        if (
            episode_min_proximity
            == float("inf")
        ):
            episode_min_proximity = 0.0

        # ====================================================
        # SAVE HISTORY IN MEMORY
        # ====================================================

        rewards_history.append(
            episode_reward
        )

        loss_history.append(
            avg_loss
        )

        success_history.append(
            success_value
        )

        obstacle_history.append(
            current_obstacles
        )

        min_proximity_history.append(
            episode_min_proximity
        )

        total_rotation_history.append(
            episode_total_rotation
        )

        steps_history.append(
            env.steps
        )

        collision_history.append(
            1 if collision else 0
        )

        timeout_history.append(
            1 if timeout else 0
        )

        epsilon_history.append(
            agent.epsilon
        )

        # ====================================================
        # MONITORING
        # ====================================================

        current_sr = (
            100.0
            * sum(success_window)
            / len(success_window)
        )

        if episode % 20 == 0:

            print(
                f"Ep {episode:04d} | "
                f"Obs: {current_obstacles} | "
                f"Stage Rolling SR: "
                f"{current_sr:5.1f}% | "
                f"Reward: "
                f"{episode_reward:8.2f} | "
                f"Loss: "
                f"{avg_loss:9.4f} | "
                f"Eps: "
                f"{agent.epsilon:.3f} | "
                f"d_min: "
                f"{episode_min_proximity:.3f}m | "
                f"Goal: {success}"
            )

        # ====================================================
        # SAVE EVERY 100 EPISODES
        # ====================================================

        if episode % 100 == 0:

            checkpoint_path = os.path.join(
                checkpoint_dir,
                (
                    f"adaptive60_obs_"
                    f"{current_obstacles}_"
                    f"seed{seed}_"
                    f"ep_{episode}.pth"
                )
            )

            torch.save(
                {
                    "episode": episode,
                    "seed": seed,

                    "obstacles":
                        current_obstacles,

                    "model_state_dict":
                        agent.q_network.state_dict(),

                    "target_state_dict":
                        agent.target_network.state_dict(),

                    "optimizer_state_dict":
                        agent.optimizer.state_dict(),

                    "epsilon":
                        agent.epsilon,

                    "curriculum":
                        "adaptive",

                    "threshold":
                        curriculum_threshold,

                    "window":
                        success_window_size,

                },
                checkpoint_path
            )

            # ================================================
            # SAVE METRICS
            # ================================================

            np.save(
                os.path.join(
                    save_dir,
                    "rewards_history.npy"
                ),
                np.asarray(
                    rewards_history
                )
            )

            np.save(
                os.path.join(
                    save_dir,
                    "loss_history.npy"
                ),
                np.asarray(
                    loss_history
                )
            )

            np.save(
                os.path.join(
                    save_dir,
                    "success_history.npy"
                ),
                np.asarray(
                    success_history
                )
            )

            np.save(
                os.path.join(
                    save_dir,
                    "obstacle_history.npy"
                ),
                np.asarray(
                    obstacle_history
                )
            )

            np.save(
                os.path.join(
                    save_dir,
                    "min_proximity_history.npy"
                ),
                np.asarray(
                    min_proximity_history
                )
            )

            np.save(
                os.path.join(
                    save_dir,
                    "total_rotation_history.npy"
                ),
                np.asarray(
                    total_rotation_history
                )
            )

            np.save(
                os.path.join(
                    save_dir,
                    "steps_history.npy"
                ),
                np.asarray(
                    steps_history
                )
            )

            np.save(
                os.path.join(
                    save_dir,
                    "collision_history.npy"
                ),
                np.asarray(
                    collision_history
                )
            )

            np.save(
                os.path.join(
                    save_dir,
                    "timeout_history.npy"
                ),
                np.asarray(
                    timeout_history
                )
            )

            np.save(
                os.path.join(
                    save_dir,
                    "epsilon_history.npy"
                ),
                np.asarray(
                    epsilon_history
                )
            )

    # ========================================================
    # FINAL CHECKPOINT
    # ========================================================

    final_checkpoint = os.path.join(
        checkpoint_dir,
        f"adaptive60_seed{seed}_FINAL.pth"
    )

    torch.save(
        {
            "episode":
                episodes,

            "seed":
                seed,

            "obstacles":
                current_obstacles,

            "model_state_dict":
                agent.q_network.state_dict(),

            "target_state_dict":
                agent.target_network.state_dict(),

            "optimizer_state_dict":
                agent.optimizer.state_dict(),

            "epsilon":
                agent.epsilon,

            "curriculum":
                "adaptive",

            "threshold":
                curriculum_threshold,

            "window":
                success_window_size,

            "transitions":
                transition_history,
        },
        final_checkpoint
    )

    # ========================================================
    # FINAL METRIC PROTECTION SAVE
    # ========================================================

    histories = {
        "rewards_history.npy":
            rewards_history,

        "loss_history.npy":
            loss_history,

        "success_history.npy":
            success_history,

        "obstacle_history.npy":
            obstacle_history,

        "min_proximity_history.npy":
            min_proximity_history,

        "total_rotation_history.npy":
            total_rotation_history,

        "steps_history.npy":
            steps_history,

        "collision_history.npy":
            collision_history,

        "timeout_history.npy":
            timeout_history,

        "epsilon_history.npy":
            epsilon_history,
    }

    for filename, values in histories.items():

        np.save(
            os.path.join(
                save_dir,
                filename
            ),
            np.asarray(values)
        )

    # ========================================================
    # PRINT TRANSITIONS
    # ========================================================

    print("\n" + "=" * 70)
    print("ADAPTIVE-60 TRAINING COMPLETE")
    print(f"Seed             : {seed}")
    print(f"Final obstacles  : {current_obstacles}")
    print(f"Final epsilon    : {agent.epsilon:.4f}")

    print("\nCurriculum transitions:")

    if transition_history:

        for transition in transition_history:

            print(
                f"Episode "
                f"{transition['episode']} : "
                f"{transition['from_obstacles']} -> "
                f"{transition['to_obstacles']} "
                f"(SR = "
                f"{transition['rolling_sr'] * 100:.1f}%)"
            )

    else:
        print("No curriculum transitions.")

    print(
        f"\nFinal checkpoint : "
        f"{final_checkpoint}"
    )

    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    train_adaptive60(seed=42)
    