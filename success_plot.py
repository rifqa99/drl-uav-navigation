import os
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = r"G:\UNİ\BAHAR\EEM5044_Derin_Pekiştirmeli_Öğrenme\Project\drl-uav-navigation\outputs\logs\256beams_standard"

print(os.path.exists(BASE_DIR))
print(os.listdir(BASE_DIR))
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

def load(name):
    path = os.path.join(BASE_DIR, name)
    if not os.path.exists(path):
        print("Missing:", name)
        return None
    return np.load(path, allow_pickle=True)

def rolling_mean(x, window=100):
    x = np.asarray(x, dtype=float)
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")

rewards = load("rewards_history_dynamic.npy")
loss = load("loss_history_dynamic.npy")
success = load("success_history_dynamic.npy")
obstacles = load("obstacle_history_dynamic.npy")
min_prox = load("min_proximity_histor y_dynamic.npy")
rotations = load("total_rotation_history_dynamic.npy")
speed = load("speed_history_dynamic.npy")
omega = load("omega_history_dynamic.npy")
steps = load("steps_history_dynamic.npy")
collision = load("collision_history_dynamic.npy")
timeout = load("timeout_history_dynamic.npy")
stage_sr = load("stage_sr_history_dynamic.npy")

window = 100
MAX_EP = 5000

def cut(x):
    return x[:MAX_EP] if x is not None else None

rewards = cut(rewards)
loss = cut(loss)
success = cut(success)
obstacles = cut(obstacles)
min_prox = cut(min_prox)
rotations = cut(rotations)
speed = cut(speed)
omega = cut(omega)
steps = cut(steps)
collision = cut(collision)
timeout = cut(timeout)
stage_sr = cut(stage_sr)

episodes = np.arange(1, len(success) + 1)

# 1. Curriculum success loop
sr = rolling_mean(success, window) * 100
sr_ep = np.arange(window, window + len(sr))

stage_changes = []
for i in range(1, len(obstacles)):
    if obstacles[i] != obstacles[i - 1]:
        stage_changes.append((i + 1, obstacles[i]))

plt.figure(figsize=(14, 5))
plt.plot(sr_ep, sr, linewidth=2.5, label="Rolling Success Rate (100 Ep Window)")
plt.axhline(70, linestyle="--", linewidth=2, label="Curriculum Stage Clear Target (70%)")

for ep, obs in stage_changes:
    plt.axvline(ep, linestyle=":", linewidth=2)
    plt.text(ep + 20, 35, f"To {int(obs)} Obs", rotation=90, va="center")

plt.xlabel("Training Episode Count")
plt.ylabel("Sliding Window Success Rate (%)")
plt.title("Performance-Driven Curriculum Scaling Loop")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "01_curriculum_success_loop.png"), dpi=300)
plt.show()
# 2. Failure modes
collision_rate = rolling_mean(collision, window) * 100
timeout_rate = rolling_mean(timeout, window) * 100
rate_ep = np.arange(window, window + len(collision_rate))

plt.figure(figsize=(14, 5))
plt.plot(rate_ep, collision_rate, linewidth=2, label="Collision Rate")
plt.plot(rate_ep, timeout_rate, linewidth=2, label="Timeout Rate")

plt.xlabel("Training Episode")
plt.ylabel("Rolling Rate (%)")
plt.title("Failure Mode Rates During Training")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "02_failure_modes.png"), dpi=300)
plt.show()
# 3. Obstacle curriculum
plt.figure(figsize=(12, 4))
plt.step(episodes, obstacles, where="post", linewidth=2.5)

plt.xlabel("Training Episode")
plt.ylabel("Number of Dynamic Obstacles")
plt.title("Curriculum Obstacle Progression")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "03_obstacle_curriculum.png"), dpi=300)
plt.show()
# 4. Reward curve
reward_ma = rolling_mean(rewards, window)
reward_ep = np.arange(window, window + len(reward_ma))

plt.figure(figsize=(14, 5))
plt.plot(episodes, rewards, alpha=0.25, label="Episode Reward")
plt.plot(reward_ep, reward_ma, linewidth=2.5, label="Rolling Reward (100 Ep)")

plt.xlabel("Training Episode")
plt.ylabel("Reward")
plt.title("Training Reward Progression")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "04_reward_curve.png"), dpi=300)
plt.show()
# 5. Loss curve
loss_ma = rolling_mean(loss, window)
loss_ep = np.arange(window, window + len(loss_ma))

plt.figure(figsize=(14, 5))
plt.plot(episodes, loss, alpha=0.25, label="Episode Loss")
plt.plot(loss_ep, loss_ma, linewidth=2.5, label="Rolling Loss (100 Ep)")

plt.xlabel("Training Episode")
plt.ylabel("Average TD Loss")
plt.title("DQN Loss During Training")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "05_loss_curve.png"), dpi=300)
plt.show()
# 6. Minimum proximity
prox_ma = rolling_mean(min_prox, window)
prox_ep = np.arange(window, window + len(prox_ma))

plt.figure(figsize=(14, 5))
plt.plot(episodes, min_prox, alpha=0.25, label="Episode Minimum Proximity")
plt.plot(prox_ep, prox_ma, linewidth=2.5, label="Rolling Minimum Proximity (100 Ep)")

plt.xlabel("Training Episode")
plt.ylabel("Minimum Distance to Obstacle (m)")
plt.title("Minimum Obstacle Proximity During Training")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "06_minimum_proximity.png"), dpi=300)
plt.show()
# 7. Rotation actions
rot_ma = rolling_mean(rotations, window)
rot_ep = np.arange(window, window + len(rot_ma))

plt.figure(figsize=(14, 5))
plt.plot(episodes, rotations, alpha=0.25, label="Episode Rotation Actions")
plt.plot(rot_ep, rot_ma, linewidth=2.5, label="Rolling Rotation Actions (100 Ep)")

plt.xlabel("Training Episode")
plt.ylabel("Number of Rotation Actions")
plt.title("Rotation Action Frequency During Training")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "07_rotation_actions.png"), dpi=300)
plt.show()
# 8. Omega curve
omega_ma = rolling_mean(omega, window)
omega_ep = np.arange(window, window + len(omega_ma))

plt.figure(figsize=(14, 5))
plt.plot(episodes, omega, alpha=0.25, label="Episode Mean Angular Velocity")
plt.plot(omega_ep, omega_ma, linewidth=2.5, label="Rolling Mean Omega (100 Ep)")

plt.xlabel("Training Episode")
plt.ylabel("Mean |Omega|")
plt.title("Angular Velocity During Training")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "08_omega_curve.png"), dpi=300)
plt.show()
# 9. Speed curve
speed_ma = rolling_mean(speed, window)
speed_ep = np.arange(window, window + len(speed_ma))

plt.figure(figsize=(14, 5))
plt.plot(episodes, speed, alpha=0.25, label="Episode Mean Speed")
plt.plot(speed_ep, speed_ma, linewidth=2.5, label="Rolling Mean Speed (100 Ep)")

plt.xlabel("Training Episode")
plt.ylabel("Mean Speed (m/s)")
plt.title("UAV Speed During Training")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "09_speed_curve.png"), dpi=300)
plt.show()
# 10. Episode steps
steps_ma = rolling_mean(steps, window)
steps_ep = np.arange(window, window + len(steps_ma))

plt.figure(figsize=(14, 5))
plt.plot(episodes, steps, alpha=0.25, label="Episode Steps")
plt.plot(steps_ep, steps_ma, linewidth=2.5, label="Rolling Steps (100 Ep)")

plt.xlabel("Training Episode")
plt.ylabel("Steps")
plt.title("Episode Length During Training")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "10_episode_steps.png"), dpi=300)
plt.show()
# 11. Combined telemetry: speed, omega, rotation
fig, ax1 = plt.subplots(figsize=(14, 5))

ax1.plot(speed_ep, speed_ma, label="Mean Speed", linewidth=2)
ax1.plot(omega_ep, omega_ma, label="Mean |Omega|", linewidth=2)
ax1.set_xlabel("Training Episode")
ax1.set_ylabel("Speed / Omega")
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(rot_ep, rot_ma, linestyle="--", linewidth=2, label="Rotation Actions")
ax2.set_ylabel("Rotation Actions")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

plt.title("Motion Smoothness Telemetry")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "11_motion_smoothness_telemetry.png"), dpi=300)
plt.show()
# 12. Success, collision, timeout stacked/evaluation-style training rates
plt.figure(figsize=(14, 5))

plt.plot(sr_ep, sr, linewidth=2.5, label="Success Rate")
plt.plot(rate_ep, collision_rate, linewidth=2.5, label="Collision Rate")
plt.plot(rate_ep, timeout_rate, linewidth=2.5, label="Timeout Rate")

plt.xlabel("Training Episode")
plt.ylabel("Rolling Rate (%)")
plt.title("Rolling Training Outcome Rates")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "12_training_outcome_rates.png"), dpi=300)
plt.show()
print("Saved all figures to:")
print(FIG_DIR)