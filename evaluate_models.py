import os
import numpy as np
import torch
from collections import deque
import pandas as pd

from env.uav_env_dynamic import UAVLiDARDynamicEnv
from agents.dqn_agent import DQNAgent


BASE_DIR = r"G:\UNİ\BAHAR\DRL\Project\drl-uav-navigation\outputs"

CHECKPOINTS = {
    "Dynamic Standard":
        os.path.join(BASE_DIR, "checkpoints",
                     "dqn_dynamic_standard_obs_8_ep_6000.pth"),

    # "Risk-Aware":
    #     os.path.join(BASE_DIR, "checkpoints",
    #                  "dqn_dynamic_risk_aware_obs_8_ep_6000.pth"),
}

SAVE_DIR = os.path.join(BASE_DIR, "test_results")
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STACK_SIZE = 3
TEST_SEEDS = list(range(100))
OBSTACLE_LEVELS = [2, 4, 6, 8]


def load_agent(checkpoint_path, env):
    state_dim = env.observation_space.shape[0] * STACK_SIZE
    action_dim = env.action_space.n

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=DEVICE
    )

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

    if "model_state_dict" in checkpoint:
        agent.q_network.load_state_dict(checkpoint["model_state_dict"])
    elif "q_network" in checkpoint:
        agent.q_network.load_state_dict(checkpoint["q_network"])
    else:
        raise KeyError("Could not find model weights in checkpoint.")

    agent.epsilon = 0.0
    agent.q_network.eval()
    return agent


def evaluate_one_model(model_name, checkpoint_path):
    all_rows = []

    for n_obs in OBSTACLE_LEVELS:
        print(f"\nEvaluating {model_name} | Obstacles: {n_obs}")

        env = UAVLiDARDynamicEnv(n_obstacles=n_obs)
        agent = load_agent(checkpoint_path, env)

        for seed in TEST_SEEDS:
            obs, _ = env.reset(seed=seed)

            frame_stack = deque([obs] * STACK_SIZE, maxlen=STACK_SIZE)
            state = np.concatenate(list(frame_stack), axis=0)

            total_reward = 0.0
            total_rotations = 0
            min_distance = float("inf")
            max_speed_seen = 0.0
            steps = 0

            done = False
            final_info = {}

            while not done:
                action = agent.select_action(state)

                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                frame_stack.append(next_obs)
                state = np.concatenate(list(frame_stack), axis=0)

                total_reward += reward
                steps += 1

                if action in [3, 4]:
                    total_rotations += 1

                lidar = next_obs[:env.n_lidar]
                current_min = float(np.min(lidar)) * env.world_size
                min_distance = min(min_distance, current_min)

                max_speed_seen = max(max_speed_seen, float(info.get("speed", 0.0)))
                final_info = info

            success = int(final_info.get("reached_goal", False))
            collision = int(final_info.get("collision", False))
            timeout = int((not success) and (not collision))

            all_rows.append({
                "model": model_name,
                "obstacles": n_obs,
                "seed": seed,
                "success": success,
                "collision": collision,
                "timeout": timeout,
                "steps": steps,
                "reward": total_reward,
                "min_distance_m": min_distance,
                "max_speed": max_speed_seen,
                "rotation_actions": total_rotations,
            })

    return pd.DataFrame(all_rows)


def summarize(df):
    summary = []

    for (model, obs), group in df.groupby(["model", "obstacles"]):
        summary.append({
            "Model": model,
            "Obstacles": obs,
            "Success Rate (%)": group["success"].mean() * 100,
            "Collision Rate (%)": group["collision"].mean() * 100,
            "Timeout Rate (%)": group["timeout"].mean() * 100,
            "Steps Mean": group["steps"].mean(),
            "Steps Std": group["steps"].std(),
            "Reward Mean": group["reward"].mean(),
            "Reward Std": group["reward"].std(),
            "Min Distance Mean (m)": group["min_distance_m"].mean(),
            "Min Distance Std (m)": group["min_distance_m"].std(),
            "Max Speed Mean": group["max_speed"].mean(),
            "Max Speed Std": group["max_speed"].std(),
            "Rotation Mean": group["rotation_actions"].mean(),
            "Rotation Std": group["rotation_actions"].std(),
        })

    return pd.DataFrame(summary)


if __name__ == "__main__":
    print("Device:", DEVICE)

    dfs = []

    for model_name, ckpt in CHECKPOINTS.items():
        if not os.path.exists(ckpt):
            print(f"Missing checkpoint: {ckpt}")
            continue

        df_model = evaluate_one_model(model_name, ckpt)
        dfs.append(df_model)

    results = pd.concat(dfs, ignore_index=True)
    summary = summarize(results)

    raw_path = os.path.join(SAVE_DIR, "raw_test_results.csv")
    summary_path = os.path.join(SAVE_DIR, "summary_test_results.csv")

    results.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\nSaved raw results to:", raw_path)
    print("Saved summary to:", summary_path)

    print("\nFinal Summary:")
    print(summary.round(3))