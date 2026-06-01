import numpy as np
import matplotlib.pyplot as plt

# Load the file directly from the secure Drive path
history_path = "rewards_history_dynamic.npy"

rewards = np.load(history_path)

# Calculate a rolling average over 50 episodes to show a smooth convergence trend
window_size = 50
smoothed_rewards = np.convolve(rewards, np.ones(
    window_size)/window_size, mode='valid')

plt.figure(figsize=(10, 6))
plt.plot(rewards, alpha=0.2, color='blue', label='Raw Episode Reward')
plt.plot(range(window_size - 1, len(rewards)), smoothed_rewards,
         color='red', linewidth=2.5, label='Moving Average (Window=50)')

plt.title('DD-DQN in dynamic obs Navigation Training Convergence',
          fontsize=14, fontweight='bold')
plt.xlabel('Training Episodes', fontsize=12)
plt.ylabel('Cumulative Reward', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=11)

# Save the figure cleanly as a publication-ready vector graphic
# plt.savefig('/content/drive/MyDrive/drl-uav-navigation/outputs_dynamic/convergence_curve.pdf', bbox_inches='tight')
plt.show()
