import torch
import numpy as np
from collections import deque

from env.uav_env_dynamic import UAVLiDARDynamicEnv
from agents.dqn_agent import DQNAgent


def evaluate_checkpoint(checkpoint_path, obstacle_stages=[2, 4, 6, 8], num_seeds=100):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stack_size = 3

    np.random.seed(99)
    test_seeds = np.random.randint(9000, 9999, size=num_seeds)

    print("| Obstacles | Success Rate (%) | Avg Steps | Avg Reward | Collision Rate (%) | Timeout Rate (%) | Avg Rotations | Avg d_min (m) |")
    print("|-----------|------------------|-----------|------------|--------------------|------------------|---------------|---------------|")

    for n_obs in obstacle_stages:
        env = UAVLiDARDynamicEnv(n_obstacles=n_obs)

        state_dim = env.observation_space.shape[0] * stack_size
        action_dim = env.action_space.n

        agent = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            device=device
        )

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        agent.q_network.load_state_dict(checkpoint["model_state_dict"])
        agent.target_network.load_state_dict(checkpoint["model_state_dict"])
        agent.epsilon = 0.0
        agent.q_network.eval()

        successes = 0
        collisions = 0
        timeouts = 0

        steps_list = []
        rewards_list = []
        rotation_list = []
        dmin_list = []

        for seed in test_seeds:
            obs, _ = env.reset(seed=int(seed))

            frame_stack = deque([obs] * stack_size, maxlen=stack_size)
            state = np.concatenate(list(frame_stack), axis=0)

            episode_reward = 0.0
            steps = 0
            rotations = 0
            episode_dmin = float("inf")

            while True:
                action = agent.select_action(state)

                if action in [3, 4]:
                    rotations += 1

                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                if "min_lidar_distance" in info:
                    episode_dmin = min(episode_dmin, info["min_lidar_distance"])

                frame_stack.append(next_obs)
                state = np.concatenate(list(frame_stack), axis=0)

                episode_reward += reward
                steps += 1

                if done:
                    if info.get("reached_goal", False):
                        successes += 1
                    elif info.get("collision", False):
                        collisions += 1
                    else:
                        timeouts += 1

                    break

            steps_list.append(steps)
            rewards_list.append(episode_reward)
            rotation_list.append(rotations)

            if episode_dmin == float("inf"):
                episode_dmin = 0.0
            dmin_list.append(episode_dmin)

        total = len(test_seeds)

        print(
            f"| {n_obs:9d} | "
            f"{100 * successes / total:16.1f}% | "
            f"{np.mean(steps_list):9.1f} | "
            f"{np.mean(rewards_list):10.2f} | "
            f"{100 * collisions / total:18.1f}% | "
            f"{100 * timeouts / total:16.1f}% | "
            f"{np.mean(rotation_list):13.1f} | "
            f"{np.mean(dmin_list):13.3f} |"
        )


evaluate_checkpoint(
    checkpoint_path="/content/drive/MyDrive/drl-uav-navigation/outputs_dynamic_risk_aware_new/checkpoints/dqn_adaptive_obs_8_ep_6000.pth"
)