import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import deque
from tqdm import tqdm

from env.uav_env_dynamic import UAVLiDARDynamicEnv
from agents.dqn_agent import DQNAgent


# ============================================================
# CONFIGURATION
# ============================================================

CHECKPOINT = (
    "/content/drive/MyDrive/drl-uav-navigation/"
    "outputs_fixed_curriculum/seed_42/checkpoints/"
    "fixed_seed42_FINAL.pth"
)

OUTPUT_DIR = (
    "/content/drive/MyDrive/drl-uav-navigation/"
    "outputs_fixed_curriculum/seed_42/evaluation"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

STACK_SIZE = 3
N_LIDAR = 256

OBSTACLE_LEVELS = [2, 4, 6, 8]

# SAME 100 held-out seeds for every obstacle density
TEST_SEEDS = list(range(10000, 10100))


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 75)
print("FIXED CURRICULUM — 100-SEED GENERALIZATION EVALUATION")
print("=" * 75)
print("Device     :", DEVICE)
print("Checkpoint :", CHECKPOINT)
print("Test seeds : 10000-10099")
print("Densities  :", OBSTACLE_LEVELS)
print("=" * 75)

assert os.path.exists(CHECKPOINT), (
    f"Checkpoint not found:\n{CHECKPOINT}"
)

# Temporary environment only to obtain dimensions
temp_env = UAVLiDARDynamicEnv(
    n_obstacles=8,
    reward_mode="standard",
    n_lidar=N_LIDAR,
    seed=42
)

obs_dim = temp_env.observation_space.shape[0]
state_dim = obs_dim * STACK_SIZE
action_dim = temp_env.action_space.n

print(f"\nObservation dimension : {obs_dim}")
print(f"Stacked state dim     : {state_dim}")
print(f"Action dimension      : {action_dim}")

agent = DQNAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    lr=1e-4,
    gamma=0.99,
    device=DEVICE
)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=False
)

agent.q_network.load_state_dict(
    checkpoint["model_state_dict"]
)

# ------------------------------------------------------------
# PURE GREEDY EVALUATION
# ------------------------------------------------------------

agent.epsilon = 0.0
agent.q_network.eval()

print("\nCheckpoint loaded successfully.")
print("Evaluation epsilon = 0.0")


# ============================================================
# EVALUATION
# ============================================================

all_results = []

with torch.no_grad():

    for n_obstacles in OBSTACLE_LEVELS:

        print("\n" + "=" * 75)
        print(f"EVALUATING {n_obstacles} OBSTACLES")
        print("=" * 75)

        for test_seed in tqdm(TEST_SEEDS):

            # ------------------------------------------------
            # EXACT SAME ENVIRONMENT CONFIGURATION AS TRAINING
            # except obstacle count and evaluation seed
            # ------------------------------------------------

            env = UAVLiDARDynamicEnv(
                n_obstacles=n_obstacles,
                reward_mode="standard",
                n_lidar=N_LIDAR,
                seed=test_seed
            )

            obs, _ = env.reset(seed=test_seed)

            # Same 3-frame initialization as training
            frame_stack = deque(
                [obs] * STACK_SIZE,
                maxlen=STACK_SIZE
            )

            state = np.concatenate(
                list(frame_stack),
                axis=0
            )

            episode_reward = 0.0
            episode_min_lidar = float("inf")

            final_info = {}

            # ------------------------------------------------
            # RUN EPISODE
            # ------------------------------------------------

            while True:

                action = agent.select_action(state)

                next_obs, reward, terminated, truncated, info = (
                    env.step(action)
                )

                done = terminated or truncated

                # Same LiDAR metric used during training
                if "min_lidar_distance" in info:
                    episode_min_lidar = min(
                        episode_min_lidar,
                        float(info["min_lidar_distance"])
                    )

                frame_stack.append(next_obs)

                state = np.concatenate(
                    list(frame_stack),
                    axis=0
                )

                episode_reward += reward
                final_info = info

                if done:
                    break

            # ------------------------------------------------
            # OUTCOME
            # ------------------------------------------------

            success = bool(
                final_info.get("reached_goal", False)
            )

            collision = bool(
                final_info.get("collision", False)
            )

            timeout = bool(
                final_info.get("timeout", False)
            )

            if episode_min_lidar == float("inf"):
                episode_min_lidar = np.nan

            final_goal_distance = float(
                final_info.get(
                    "distance_to_goal",
                    np.linalg.norm(env.pos - env.goal)
                )
            )

            all_results.append(
                {
                    "obstacles": n_obstacles,
                    "seed": test_seed,

                    "success": int(success),
                    "collision": int(collision),
                    "timeout": int(timeout),

                    "steps": int(env.steps),

                    "cumulative_reward":
                        float(episode_reward),

                    "min_lidar_distance":
                        float(episode_min_lidar),

                    "final_goal_distance":
                        final_goal_distance,
                }
            )


# ============================================================
# RAW RESULTS
# ============================================================

results = pd.DataFrame(all_results)

raw_path = os.path.join(
    OUTPUT_DIR,
    "fixed_seed42_evaluation_raw.csv"
)

results.to_csv(
    raw_path,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

summary_rows = []

for n_obstacles in OBSTACLE_LEVELS:

    df = results[
        results["obstacles"] == n_obstacles
    ]

    summary_rows.append(
        {
            "Obstacles": n_obstacles,

            "Success Rate (%)":
                100 * df["success"].mean(),

            "Collision Rate (%)":
                100 * df["collision"].mean(),

            "Timeout Rate (%)":
                100 * df["timeout"].mean(),

            "Mean Steps":
                df["steps"].mean(),

            "Mean Reward":
                df["cumulative_reward"].mean(),

            "Mean Min LiDAR Distance (m)":
                df["min_lidar_distance"].mean(),

            "Median Min LiDAR Distance (m)":
                df["min_lidar_distance"].median(),

            "Mean Final Goal Distance (m)":
                df["final_goal_distance"].mean(),
        }
    )

summary = pd.DataFrame(summary_rows)

summary_path = os.path.join(
    OUTPUT_DIR,
    "fixed_seed42_evaluation_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n")
print("=" * 105)
print("FIXED CURRICULUM — 100-SEED GENERALIZATION RESULTS")
print("=" * 105)

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}"
    )
)

print("=" * 105)

print("\nRaw results saved to:")
print(raw_path)

print("\nSummary saved to:")
print(summary_path)


# ============================================================
# SUCCESS-RATE PLOT
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    summary["Obstacles"],
    summary["Success Rate (%)"],
    marker="o",
    linewidth=2
)

plt.xticks([2, 4, 6, 8])
plt.ylim(0, 100)

plt.xlabel("Number of Dynamic Obstacles")
plt.ylabel("Success Rate (%)")

plt.title(
    "Fixed Curriculum — Generalization Performance\n"
    "100 Unseen Seeds per Obstacle Density"
)

plt.grid(alpha=0.3)

plt.tight_layout()

plot_path = os.path.join(
    OUTPUT_DIR,
    "fixed_seed42_success_rate.png"
)

plt.savefig(
    plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nPlot saved to:")
print(plot_path)