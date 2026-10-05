# ============================================================
# UNIVERSAL ADAPTIVE-THRESHOLD EVALUATOR
#
# Examples:
#   THRESHOLD = 60
#   THRESHOLD = 70
#   THRESHOLD = 80
#
# Also accepts:
#   THRESHOLD = 0.60
#   THRESHOLD = 0.70
#   THRESHOLD = 0.80
#
# Change ONLY THRESHOLD to evaluate another adaptive model.
# ============================================================

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from collections import deque

from env.uav_env_dynamic import UAVLiDARDynamicEnv
from agents.dqn_agent import DQNAgent


# ============================================================
# USER INPUT
# ============================================================

THRESHOLD = 70
SEED = 42


# ============================================================
# NORMALIZE THRESHOLD INPUT
# ============================================================

# Allows either:
#   80
# or:
#   0.80

if THRESHOLD <= 1:
    THRESHOLD_PERCENT = int(round(THRESHOLD * 100))
else:
    THRESHOLD_PERCENT = int(round(THRESHOLD))

THRESHOLD_FRACTION = THRESHOLD_PERCENT / 100.0

EXPERIMENT_NAME = f"adaptive{THRESHOLD_PERCENT}"
METHOD_NAME = f"Adaptive-{THRESHOLD_PERCENT}"


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.join(
    "/content/drive/MyDrive/drl-uav-navigation",
    f"outputs_{EXPERIMENT_NAME}",
    f"seed_{SEED}"
)

CHECKPOINT_DIR = os.path.join(
    BASE_DIR,
    "checkpoints"
)

EVAL_DIR = os.path.join(
    BASE_DIR,
    "evaluation"
)

os.makedirs(
    EVAL_DIR,
    exist_ok=True
)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

STACK_SIZE = 3
N_LIDAR = 256

# EXACT SAME held-out test environments
# for every method.
TEST_SEEDS = list(
    range(10000, 10100)
)

OBSTACLE_COUNTS = [
    2,
    4,
    6,
    8
]


# ============================================================
# EXPERIMENT INFORMATION
# ============================================================

print("=" * 70)

print(
    f"{METHOD_NAME} — "
    f"TRAINING CURVES + 100-SEED EVALUATION"
)

print("=" * 70)

print(
    f"Threshold          : "
    f"{THRESHOLD_PERCENT}%"
)

print(
    f"Training seed      : "
    f"{SEED}"
)

print(
    f"Device             : "
    f"{DEVICE}"
)

print(
    f"Base directory     : "
    f"{BASE_DIR}"
)

print(
    f"Test seeds         : "
    f"{TEST_SEEDS[0]}-{TEST_SEEDS[-1]}"
)

print(
    f"Obstacle densities : "
    f"{OBSTACLE_COUNTS}"
)


# ============================================================
# VERIFY DIRECTORY
# ============================================================

if not os.path.exists(BASE_DIR):

    raise FileNotFoundError(
        f"\nCould not find:\n"
        f"{BASE_DIR}\n\n"
        f"Make sure {METHOD_NAME} "
        f"has finished training."
    )


# ============================================================
# PART 1 — LOAD TRAINING HISTORY
# ============================================================

history_files = {
    "rewards":
        "rewards_history.npy",

    "success":
        "success_history.npy",

    "losses":
        "loss_history.npy",

    "obstacles":
        "obstacle_history.npy",

    "epsilon":
        "epsilon_history.npy",

    "collisions":
        "collision_history.npy",

    "steps":
        "steps_history.npy"
}


loaded = {}

for key, filename in history_files.items():

    path = os.path.join(
        BASE_DIR,
        filename
    )

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Missing training history:\n"
            f"{path}"
        )

    loaded[key] = np.load(path)


rewards = loaded["rewards"]
success = loaded["success"]
losses = loaded["losses"]
obstacles = loaded["obstacles"]
epsilon = loaded["epsilon"]
collisions = loaded["collisions"]
steps = loaded["steps"]


# ============================================================
# CHECK HISTORY LENGTHS
# ============================================================

history_lengths = {
    key: len(value)
    for key, value in loaded.items()
}

if len(set(history_lengths.values())) != 1:

    print(
        "\nWARNING: Training history arrays "
        "do not all have the same length."
    )

    for key, length in history_lengths.items():
        print(
            f"{key:12s}: {length}"
        )


print(
    "\nTraining episodes:",
    len(success)
)


# ============================================================
# EPISODES AT EACH OBSTACLE DENSITY
# ============================================================

unique_obs, counts = np.unique(
    obstacles,
    return_counts=True
)

print(
    "\nEpisodes spent at each "
    "obstacle density:"
)

for obs, count in zip(
    unique_obs,
    counts
):

    print(
        f"{int(obs)} obstacles : "
        f"{count} episodes"
    )


# ============================================================
# DETECT ACTUAL CURRICULUM TRANSITIONS
# ============================================================

transition_episodes = []

for i in range(
    1,
    len(obstacles)
):

    if (
        obstacles[i]
        != obstacles[i - 1]
    ):

        transition_episodes.append(
            (
                i + 1,
                int(obstacles[i - 1]),
                int(obstacles[i])
            )
        )


print(
    "\nDetected curriculum transitions:"
)

if transition_episodes:

    for (
        ep,
        old_obs,
        new_obs
    ) in transition_episodes:

        print(
            f"Episode {ep}: "
            f"{old_obs} -> "
            f"{new_obs} obstacles"
        )

else:

    print(
        "No transitions detected."
    )


# ============================================================
# HELPER — GLOBAL ROLLING MEAN
# ============================================================

def rolling_mean(
    data,
    window=100
):

    return (
        pd.Series(data)
        .rolling(
            window=window,
            min_periods=1
        )
        .mean()
        .to_numpy()
    )


episodes = np.arange(
    1,
    len(success) + 1
)


# ============================================================
# HELPER — DRAW TRANSITION LINES
# ============================================================

def draw_transitions():

    for (
        ep,
        old_obs,
        new_obs
    ) in transition_episodes:

        plt.axvline(
            ep,
            linestyle="--",
            alpha=0.7
        )


# ============================================================
# CURVE 1 — ROLLING SUCCESS RATE
# ============================================================

rolling_success = (
    rolling_mean(
        success,
        100
    )
    * 100
)

plt.figure(
    figsize=(12, 5)
)

plt.plot(
    episodes,
    rolling_success,
    linewidth=1.5,
    label="Global rolling success rate"
)

for (
    ep,
    old_obs,
    new_obs
) in transition_episodes:

    plt.axvline(
        ep,
        linestyle="--",
        alpha=0.7
    )

    plt.text(
        ep,
        5,
        f"{old_obs}→{new_obs}",
        rotation=90,
        verticalalignment="bottom"
    )


plt.axhline(
    THRESHOLD_PERCENT,
    linestyle=":",
    alpha=0.8,
    label=(
        f"{THRESHOLD_PERCENT}% "
        f"curriculum threshold"
    )
)

plt.xlabel(
    "Training Episode"
)

plt.ylabel(
    "Rolling Success Rate (%)"
)

plt.title(
    f"{METHOD_NAME} Curriculum Training — "
    f"100-Episode Rolling Success Rate"
)

plt.ylim(
    0,
    105
)

plt.grid(
    alpha=0.25
)

plt.legend()

plt.tight_layout()


success_curve_path = os.path.join(
    BASE_DIR,
    f"{EXPERIMENT_NAME}_success_curve.png"
)

plt.savefig(
    success_curve_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    "Saved:",
    success_curve_path
)


# ============================================================
# CURVE 2 — ROLLING REWARD
# ============================================================

rolling_reward = rolling_mean(
    rewards,
    100
)

plt.figure(
    figsize=(12, 5)
)

plt.plot(
    episodes,
    rolling_reward,
    linewidth=1.5
)

draw_transitions()

plt.xlabel(
    "Training Episode"
)

plt.ylabel(
    "Mean Episode Reward"
)

plt.title(
    f"{METHOD_NAME} Curriculum Training — "
    f"100-Episode Mean Reward"
)

plt.grid(
    alpha=0.25
)

plt.tight_layout()


reward_curve_path = os.path.join(
    BASE_DIR,
    f"{EXPERIMENT_NAME}_reward_curve.png"
)

plt.savefig(
    reward_curve_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    "Saved:",
    reward_curve_path
)


# ============================================================
# CURVE 3 — TRAINING LOSS
# ============================================================

rolling_loss = rolling_mean(
    losses,
    100
)

plt.figure(
    figsize=(12, 5)
)

plt.plot(
    episodes,
    rolling_loss,
    linewidth=1.5
)

draw_transitions()

plt.xlabel(
    "Training Episode"
)

plt.ylabel(
    "Mean Training Loss"
)

plt.title(
    f"{METHOD_NAME} Curriculum Training — "
    f"100-Episode Mean Loss"
)

plt.grid(
    alpha=0.25
)

plt.tight_layout()


loss_curve_path = os.path.join(
    BASE_DIR,
    f"{EXPERIMENT_NAME}_loss_curve.png"
)

plt.savefig(
    loss_curve_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    "Saved:",
    loss_curve_path
)


# ============================================================
# CURVE 4 — EPSILON
# ============================================================

plt.figure(
    figsize=(12, 4)
)

plt.plot(
    episodes,
    epsilon,
    linewidth=1.5
)

draw_transitions()

plt.xlabel(
    "Training Episode"
)

plt.ylabel(
    "Epsilon"
)

plt.title(
    f"{METHOD_NAME} Curriculum — "
    f"Exploration Schedule"
)

plt.grid(
    alpha=0.25
)

plt.tight_layout()


epsilon_curve_path = os.path.join(
    BASE_DIR,
    f"{EXPERIMENT_NAME}_epsilon_curve.png"
)

plt.savefig(
    epsilon_curve_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    "Saved:",
    epsilon_curve_path
)


# ============================================================
# PART 2 — FIND FINAL CHECKPOINT
# ============================================================

preferred_checkpoint = os.path.join(
    CHECKPOINT_DIR,
    (
        f"{EXPERIMENT_NAME}_"
        f"seed{SEED}_FINAL.pth"
    )
)


if os.path.exists(
    preferred_checkpoint
):

    checkpoint_path = (
        preferred_checkpoint
    )

else:

    candidates = glob.glob(
        os.path.join(
            CHECKPOINT_DIR,
            "*FINAL*.pth"
        )
    )

    if not candidates:

        raise FileNotFoundError(
            f"Could not find the final "
            f"{METHOD_NAME} checkpoint "
            f"inside:\n"
            f"{CHECKPOINT_DIR}"
        )

    if len(candidates) > 1:

        print(
            "\nWARNING: Multiple FINAL "
            "checkpoints found:"
        )

        for candidate in candidates:
            print(candidate)

        raise RuntimeError(
            "Multiple final checkpoints found. "
            "Refusing to guess which one to use."
        )

    checkpoint_path = candidates[0]


print(
    "\n" + "=" * 70
)

print(
    "FINAL CHECKPOINT"
)

print(
    "=" * 70
)

print(
    checkpoint_path
)


# ============================================================
# CREATE TEMP ENVIRONMENT TO GET DIMENSIONS
# ============================================================

temp_env = UAVLiDARDynamicEnv(
    n_obstacles=8,
    reward_mode="standard",
    n_lidar=N_LIDAR,
    seed=SEED
)

obs_dim = (
    temp_env
    .observation_space
    .shape[0]
)

state_dim = (
    obs_dim
    * STACK_SIZE
)

action_dim = (
    temp_env
    .action_space
    .n
)


print(
    "\nObservation dimension:",
    obs_dim
)

print(
    "Stacked state dimension:",
    state_dim
)

print(
    "Actions:",
    action_dim
)


# ============================================================
# LOAD AGENT
# ============================================================

agent = DQNAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    lr=1e-4,
    gamma=0.99,
    device=DEVICE
)

checkpoint = torch.load(
    checkpoint_path,
    map_location=DEVICE
)


# ============================================================
# ROBUST CHECKPOINT LOADING
# ============================================================

if (
    isinstance(
        checkpoint,
        dict
    )
    and
    "model_state_dict"
    in checkpoint
):

    agent.q_network.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

elif (
    isinstance(
        checkpoint,
        dict
    )
    and
    "q_network_state_dict"
    in checkpoint
):

    agent.q_network.load_state_dict(
        checkpoint[
            "q_network_state_dict"
        ]
    )

else:

    agent.q_network.load_state_dict(
        checkpoint
    )


# ============================================================
# CHECK CHECKPOINT METADATA
# ============================================================

if isinstance(
    checkpoint,
    dict
):

    saved_threshold = checkpoint.get(
        "threshold",
        None
    )

    if saved_threshold is not None:

        if saved_threshold <= 1:

            saved_threshold_percent = int(
                round(
                    saved_threshold
                    * 100
                )
            )

        else:

            saved_threshold_percent = int(
                round(
                    saved_threshold
                )
            )

        print(
            "\nCheckpoint threshold:",
            f"{saved_threshold_percent}%"
        )

        if (
            saved_threshold_percent
            != THRESHOLD_PERCENT
        ):

            raise ValueError(
                "\nTHRESHOLD MISMATCH!\n"
                f"Evaluator requested: "
                f"{THRESHOLD_PERCENT}%\n"
                f"Checkpoint reports: "
                f"{saved_threshold_percent}%\n"
                "Stopping to prevent evaluating "
                "the wrong experiment."
            )


# ============================================================
# FREEZE POLICY
# ============================================================

agent.epsilon = 0.0
agent.q_network.eval()


print(
    "\nPolicy loaded."
)

print(
    "Evaluation epsilon:",
    agent.epsilon
)


# ============================================================
# PART 3 — 100-SEED EVALUATION
# ============================================================

all_results = []


for n_obstacles in OBSTACLE_COUNTS:

    print(
        "\n" + "=" * 70
    )

    print(
        f"EVALUATING "
        f"{n_obstacles} OBSTACLES "
        f"— {len(TEST_SEEDS)} "
        f"UNSEEN SEEDS"
    )

    print(
        "=" * 70
    )

    env = UAVLiDARDynamicEnv(
        n_obstacles=n_obstacles,
        reward_mode="standard",
        n_lidar=N_LIDAR,
        seed=SEED
    )

    for (
        test_index,
        test_seed
    ) in enumerate(
        TEST_SEEDS,
        start=1
    ):

        # ====================================================
        # RESET USING HELD-OUT TEST SEED
        # ====================================================

        obs, _ = env.reset(
            seed=test_seed
        )

        frame_stack = deque(
            [obs] * STACK_SIZE,
            maxlen=STACK_SIZE
        )

        state = np.concatenate(
            list(frame_stack),
            axis=0
        )

        total_reward = 0.0
        episode_steps = 0

        min_lidar_distance = (
            float("inf")
        )

        final_info = {}

        # ====================================================
        # EVALUATION EPISODE
        # ====================================================

        while True:

            # -----------------------------------------------
            # GREEDY POLICY
            # -----------------------------------------------

            with torch.no_grad():

                state_tensor = (
                    torch
                    .FloatTensor(state)
                    .unsqueeze(0)
                    .to(DEVICE)
                )

                q_values = (
                    agent.q_network(
                        state_tensor
                    )
                )

                action = int(
                    torch.argmax(
                        q_values,
                        dim=1
                    ).item()
                )

            # -----------------------------------------------
            # ENV STEP
            # -----------------------------------------------

            (
                next_obs,
                reward,
                terminated,
                truncated,
                info
            ) = env.step(
                action
            )

            episode_steps += 1
            total_reward += reward

            # -----------------------------------------------
            # SENSOR-DERIVED MINIMUM LiDAR DISTANCE
            # -----------------------------------------------

            if (
                "min_lidar_distance"
                in info
            ):

                min_lidar_distance = min(
                    min_lidar_distance,
                    float(
                        info[
                            "min_lidar_distance"
                        ]
                    )
                )

            # -----------------------------------------------
            # UPDATE FRAME STACK
            # -----------------------------------------------

            frame_stack.append(
                next_obs
            )

            state = np.concatenate(
                list(frame_stack),
                axis=0
            )

            final_info = info

            if (
                terminated
                or truncated
            ):
                break

        # ====================================================
        # OUTCOME
        # ====================================================

        success_flag = int(
            final_info.get(
                "reached_goal",
                False
            )
        )

        collision_flag = int(
            final_info.get(
                "collision",
                False
            )
        )

        timeout_flag = int(
            final_info.get(
                "timeout",
                False
            )
        )

        if (
            min_lidar_distance
            == float("inf")
        ):

            min_lidar_distance = (
                np.nan
            )

        # ====================================================
        # STORE RESULT
        # ====================================================

        all_results.append(
            {
                "method":
                    METHOD_NAME,

                "threshold_percent":
                    THRESHOLD_PERCENT,

                "training_seed":
                    SEED,

                "n_obstacles":
                    n_obstacles,

                "test_seed":
                    test_seed,

                "success":
                    success_flag,

                "collision":
                    collision_flag,

                "timeout":
                    timeout_flag,

                "steps":
                    episode_steps,

                "cumulative_reward":
                    total_reward,

                # Sensor-derived LiDAR minimum.
                # NOT exact geometric clearance.
                "min_lidar_distance":
                    min_lidar_distance
            }
        )

        if (
            test_index
            % 20 == 0
        ):

            print(
                f"{test_index:3d}/"
                f"{len(TEST_SEEDS)} "
                f"completed"
            )


# ============================================================
# RAW RESULTS
# ============================================================

results_df = pd.DataFrame(
    all_results
)

raw_path = os.path.join(
    EVAL_DIR,
    (
        f"{EXPERIMENT_NAME}_"
        f"seed{SEED}_"
        f"100seed_raw.csv"
    )
)

results_df.to_csv(
    raw_path,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

summary_rows = []


for n_obstacles in OBSTACLE_COUNTS:

    subset = results_df[
        results_df[
            "n_obstacles"
        ]
        == n_obstacles
    ]

    summary_rows.append(
        {
            "n_obstacles":
                n_obstacles,

            "success_rate_percent":
                100
                * subset[
                    "success"
                ].mean(),

            "collision_rate_percent":
                100
                * subset[
                    "collision"
                ].mean(),

            "timeout_rate_percent":
                100
                * subset[
                    "timeout"
                ].mean(),

            "mean_steps":
                subset[
                    "steps"
                ].mean(),

            "std_steps":
                subset[
                    "steps"
                ].std(),

            "mean_cumulative_reward":
                subset[
                    "cumulative_reward"
                ].mean(),

            "std_cumulative_reward":
                subset[
                    "cumulative_reward"
                ].std(),

            "mean_min_lidar_distance":
                subset[
                    "min_lidar_distance"
                ].mean(),

            "std_min_lidar_distance":
                subset[
                    "min_lidar_distance"
                ].std()
        }
    )


summary_df = pd.DataFrame(
    summary_rows
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_path = os.path.join(
    EVAL_DIR,
    (
        f"{EXPERIMENT_NAME}_"
        f"seed{SEED}_"
        f"100seed_summary.csv"
    )
)

summary_df.to_csv(
    summary_path,
    index=False
)


# ============================================================
# PRINT TABLE
# ============================================================

print("\n")

print(
    "=" * 100
)

print(
    f"{METHOD_NAME.upper()} — "
    f"100 UNSEEN SEED EVALUATION"
)

print(
    "=" * 100
)

print(
    summary_df.round(3).to_string(index=False)
)


print(
    "\nRaw results saved to:"
)

print(
    raw_path
)


print(
    "\nSummary saved to:"
)

print(
    summary_path
)


# ============================================================
# EVALUATION CURVE
# ============================================================

plt.figure(
    figsize=(9, 5)
)

plt.plot(
    summary_df[
        "n_obstacles"
    ],
    summary_df[
        "success_rate_percent"
    ],
    marker="o",
    linewidth=2,
    label="Success"
)

plt.plot(
    summary_df[
        "n_obstacles"
    ],
    summary_df[
        "collision_rate_percent"
    ],
    marker="o",
    linewidth=2,
    label="Collision"
)

plt.plot(
    summary_df[
        "n_obstacles"
    ],
    summary_df[
        "timeout_rate_percent"
    ],
    marker="o",
    linewidth=2,
    label="Timeout"
)

plt.xticks(
    OBSTACLE_COUNTS
)

plt.ylim(
    0,
    105
)

plt.xlabel(
    "Number of Dynamic Obstacles"
)

plt.ylabel(
    "Rate (%)"
)

plt.title(
    f"{METHOD_NAME} — "
    f"Generalization over "
    f"{len(TEST_SEEDS)} Unseen Seeds"
)

plt.grid(
    alpha=0.25
)

plt.legend()

plt.tight_layout()


eval_plot_path = os.path.join(
    EVAL_DIR,
    f"{EXPERIMENT_NAME}_generalization.png"
)

plt.savefig(
    eval_plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


print(
    "Saved:",
    eval_plot_path
)


# ============================================================
# FINAL EXPERIMENT SUMMARY
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    f"{METHOD_NAME} EVALUATION COMPLETE"
)

print(
    "=" * 70
)

print(
    f"Threshold       : "
    f"{THRESHOLD_PERCENT}%"
)

print(
    f"Training seed   : "
    f"{SEED}"
)

print(
    f"Test seeds      : "
    f"{TEST_SEEDS[0]}-"
    f"{TEST_SEEDS[-1]}"
)

print(
    f"Checkpoint      : "
    f"{checkpoint_path}"
)

print(
    f"Evaluation dir  : "
    f"{EVAL_DIR}"
)

print(
    "=" * 70
)