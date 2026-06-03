import os
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================
BASE_DIR = r"G:\UNİ\BAHAR\DRL\Project\drl-uav-navigation\outputs"

DYNAMIC_STANDARD_DIR = os.path.join(BASE_DIR, "dynamic_standard")
DYNAMIC_RISK_DIR = os.path.join(BASE_DIR, "dynamic_risk")
STATIC_DIR = os.path.join(BASE_DIR, "static")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

os.makedirs(PLOTS_DIR, exist_ok=True)


# ============================================================
# Helpers
# ============================================================
def load_npy(folder, filename):
    path = os.path.join(folder, filename)
    if not os.path.exists(path):
        print(f"Missing: {path}")
        return None
    return np.load(path, allow_pickle=True)


def moving_average(x, window=100):
    x = np.asarray(x, dtype=float)
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")


def rolling_rate(binary_array, window=100):
    binary_array = np.asarray(binary_array, dtype=float)
    if len(binary_array) < window:
        return binary_array * 100
    return moving_average(binary_array, window) * 100


def savefig(name):
    path = os.path.join(PLOTS_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ============================================================
# Load histories
# ============================================================

# Dynamic Standard
std_rewards = load_npy(DYNAMIC_STANDARD_DIR, "rewards_history_dynamic.npy")
std_success = load_npy(DYNAMIC_STANDARD_DIR, "success_history_dynamic.npy")
std_obstacles = load_npy(DYNAMIC_STANDARD_DIR, "obstacle_history_dynamic.npy")
std_steps = load_npy(DYNAMIC_STANDARD_DIR, "steps_history_dynamic.npy")
std_collision = load_npy(DYNAMIC_STANDARD_DIR, "collision_history_dynamic.npy")
std_timeout = load_npy(DYNAMIC_STANDARD_DIR, "timeout_history_dynamic.npy")
std_minprox = load_npy(DYNAMIC_STANDARD_DIR, "min_proximity_history_dynamic.npy")
std_speed = load_npy(DYNAMIC_STANDARD_DIR, "speed_history_dynamic.npy")
std_omega = load_npy(DYNAMIC_STANDARD_DIR, "omega_history_dynamic.npy")
std_rot = load_npy(DYNAMIC_STANDARD_DIR, "total_rotation_history_dynamic.npy")

# Dynamic Risk-Aware
risk_rewards = load_npy(DYNAMIC_RISK_DIR, "rewards_history_dynamic.npy")
risk_success = load_npy(DYNAMIC_RISK_DIR, "success_history_dynamic.npy")
risk_obstacles = load_npy(DYNAMIC_RISK_DIR, "obstacle_history_dynamic.npy")
risk_loss = load_npy(DYNAMIC_RISK_DIR, "loss_history_dynamic.npy")
risk_minprox = load_npy(DYNAMIC_RISK_DIR, "min_proximity_history.npy")
risk_rot = load_npy(DYNAMIC_RISK_DIR, "total_rotation_history.npy")

# Static
static_rewards = load_npy(STATIC_DIR, "rewards_history_static.npy")
static_success = load_npy(STATIC_DIR, "success_history_static.npy")
static_obstacles = load_npy(STATIC_DIR, "obstacle_history_static.npy")
static_steps = load_npy(STATIC_DIR, "steps_history_static.npy")
static_minprox = load_npy(STATIC_DIR, "min_proximity_history_static.npy")
static_speed = load_npy(STATIC_DIR, "speed_history_static.npy")
static_omega = load_npy(STATIC_DIR, "omega_history_static.npy")
static_rot = load_npy(STATIC_DIR, "rotation_history_static.npy")


# ============================================================
# 1. Reward convergence: Dynamic Standard vs Risk-Aware
# ============================================================
plt.figure(figsize=(10, 5))

if std_rewards is not None:
    ma = moving_average(std_rewards, 100)
    plt.plot(range(len(ma)), ma, label="Dynamic Standard")

if risk_rewards is not None:
    ma = moving_average(risk_rewards, 100)
    plt.plot(range(len(ma)), ma, label="Risk-Aware Dynamic")

plt.xlabel("Training Episode")
plt.ylabel("Smoothed Episode Reward")
plt.title("Reward Convergence Comparison")
plt.grid(True, alpha=0.3)
plt.legend()
savefig("01_reward_convergence_comparison.png")


# ============================================================
# 2. Rolling success rate comparison
# ============================================================
plt.figure(figsize=(10, 5))

if std_success is not None:
    sr = rolling_rate(std_success, 100)
    plt.plot(range(len(sr)), sr, label="Dynamic Standard")

if risk_success is not None:
    sr = rolling_rate(risk_success, 100)
    plt.plot(range(len(sr)), sr, label="Risk-Aware Dynamic")

plt.axhline(70, linestyle="--", label="Curriculum Threshold 70%")
plt.xlabel("Training Episode")
plt.ylabel("Rolling Success Rate (%)")
plt.title("Rolling Success Rate Comparison")
plt.grid(True, alpha=0.3)
plt.legend()
savefig("02_rolling_success_rate_comparison.png")


# ============================================================
# 3. Curriculum obstacle progression
# ============================================================
plt.figure(figsize=(10, 5))

if std_obstacles is not None:
    plt.plot(std_obstacles, label="Dynamic Standard")

if risk_obstacles is not None:
    plt.plot(risk_obstacles, label="Risk-Aware Dynamic")

plt.xlabel("Training Episode")
plt.ylabel("Obstacle Count")
plt.title("Curriculum Progression")
plt.grid(True, alpha=0.3)
plt.legend()
savefig("03_curriculum_progression.png")


# ============================================================
# 4. Combined paper-style figure for Risk-Aware
# ============================================================
if risk_rewards is not None and risk_success is not None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(risk_rewards, alpha=0.15, label="Raw Episode Reward")
    axes[0].plot(
        range(99, len(risk_rewards)),
        moving_average(risk_rewards, 100),
        linewidth=2,
        label="100-Episode Moving Average"
    )
    axes[0].set_title("(a) Reward Convergence")
    axes[0].set_xlabel("Training Episode")
    axes[0].set_ylabel("Accumulated Episode Reward")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    sr = rolling_rate(risk_success, 100)
    axes[1].plot(range(len(sr)), sr, linewidth=2, label="Rolling Success Rate")
    axes[1].axhline(70, linestyle="--", label="70% Threshold")
    axes[1].set_title("(b) Curriculum Progress Loop")
    axes[1].set_xlabel("Training Episode")
    axes[1].set_ylabel("Rolling Success Rate (%)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "04_risk_aware_training_telemetry.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ============================================================
# 5. Minimum proximity comparison
# ============================================================
plt.figure(figsize=(10, 5))

if std_minprox is not None:
    plt.plot(moving_average(std_minprox, 100), label="Dynamic Standard")

if risk_minprox is not None:
    plt.plot(moving_average(risk_minprox, 100), label="Risk-Aware Dynamic")

plt.xlabel("Training Episode")
plt.ylabel("Minimum LiDAR Proximity")
plt.title("Minimum Obstacle Proximity During Training")
plt.grid(True, alpha=0.3)
plt.legend()
savefig("05_minimum_proximity_comparison.png")


# ============================================================
# 6. Rotation behavior comparison
# ============================================================
plt.figure(figsize=(10, 5))

if std_rot is not None:
    plt.plot(moving_average(std_rot, 100), label="Dynamic Standard")

if risk_rot is not None:
    plt.plot(moving_average(risk_rot, 100), label="Risk-Aware Dynamic")

plt.xlabel("Training Episode")
plt.ylabel("Rotation Actions per Episode")
plt.title("Rotational Behavior Comparison")
plt.grid(True, alpha=0.3)
plt.legend()
savefig("06_rotation_behavior_comparison.png")


# ============================================================
# 7. Static vs Dynamic vs Risk-Aware reward comparison
# ============================================================
plt.figure(figsize=(10, 5))

if static_rewards is not None:
    plt.plot(moving_average(static_rewards, 100), label="Static Standard")

if std_rewards is not None:
    plt.plot(moving_average(std_rewards, 100), label="Dynamic Standard")

if risk_rewards is not None:
    plt.plot(moving_average(risk_rewards, 100), label="Risk-Aware Dynamic")

plt.xlabel("Training Episode")
plt.ylabel("Smoothed Episode Reward")
plt.title("Training Reward Across Framework Stages")
plt.grid(True, alpha=0.3)
plt.legend()
savefig("07_all_stage_reward_comparison.png")


# ============================================================
# 8. Evaluation-style summary from training histories
#    Mean ± std by obstacle count
# ============================================================
def summarize_by_obstacle(name, obstacles, success, steps=None, rewards=None, minprox=None, rotations=None):
    if obstacles is None or success is None:
        return

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    unique_obs = sorted(np.unique(obstacles).astype(int))

    for obs in unique_obs:
        idx = np.where(obstacles == obs)[0]

        sr = np.mean(success[idx]) * 100
        sr_std = np.std(success[idx]) * 100

        line = f"Obs {obs}: Success = {sr:.1f}%"

        if steps is not None and len(steps) == len(obstacles):
            line += f" | Steps = {np.mean(steps[idx]):.1f} ± {np.std(steps[idx]):.1f}"

        if rewards is not None and len(rewards) == len(obstacles):
            line += f" | Reward = {np.mean(rewards[idx]):.1f} ± {np.std(rewards[idx]):.1f}"

        if minprox is not None and len(minprox) == len(obstacles):
            line += f" | MinProx = {np.mean(minprox[idx]):.3f} ± {np.std(minprox[idx]):.3f}"

        if rotations is not None and len(rotations) == len(obstacles):
            line += f" | Rot = {np.mean(rotations[idx]):.1f} ± {np.std(rotations[idx]):.1f}"

        print(line)


summarize_by_obstacle(
    "Dynamic Standard Training Summary",
    std_obstacles,
    std_success,
    steps=std_steps,
    rewards=std_rewards,
    minprox=std_minprox,
    rotations=std_rot
)

summarize_by_obstacle(
    "Risk-Aware Dynamic Training Summary",
    risk_obstacles,
    risk_success,
    rewards=risk_rewards,
    minprox=risk_minprox,
    rotations=risk_rot
)

summarize_by_obstacle(
    "Static Training Summary",
    static_obstacles,
    static_success,
    steps=static_steps,
    rewards=static_rewards,
    minprox=static_minprox,
    rotations=static_rot
)


print("\nAll plots saved inside:")
print(PLOTS_DIR)