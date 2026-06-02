import os
import torch
import numpy as np
from collections import deque
from tqdm import tqdm

from env.uav_env_dynamic import UAVLiDARDynamicEnv
from agents.dqn_agent import DQNAgent
from agents.replay_buffer import ReplayBuffer

def train_dqn_dynamic_adaptive_colab(checkpoint_file=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- Hyperparameters ---
    episodes = 6000  # Extended to give the agent room to progress past harder phases
    batch_size = 64
    gamma = 0.99
    lr = 1e-4
    stack_size = 3
    buffer_capacity = 50000
    target_update_frequency = 10

    # Point directly to your permanent Google Drive project paths
    save_dir = "/content/drive/MyDrive/drl-uav-navigation/outputs_dynamic_risk_aware"
    checkpoint_dir = os.path.join(save_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Setup the baseline tracker
    current_obstacles = 2
    start_episode = 1
    rewards_history = []
    success_window = deque(maxlen=100)

    # Initialize environment
    env = UAVLiDARDynamicEnv(n_obstacles=current_obstacles)
    state_dim = env.observation_space.shape[0] * stack_size
    action_dim = env.action_space.n

    replay_buffer = ReplayBuffer(capacity=buffer_capacity)
    agent = DQNAgent(
        state_dim=state_dim, 
        action_dim=action_dim, 
        lr=lr, 
        gamma=gamma, 
        device=device
    )

    # --- LOAD CHECKPOINT IF PROVIDED ---
    if checkpoint_file and os.path.exists(checkpoint_file):
        print(f"-> Loading weights from checkpoint: {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
        agent.q_network.load_state_dict(checkpoint["model_state_dict"])
        agent.target_network.load_state_dict(checkpoint["model_state_dict"])
        
        # Resume cleanly at the correct episode count
        start_episode = checkpoint.get("episode", 2000) + 1
        current_obstacles = checkpoint.get("obstacles", 2)
        
        # Update the active environment difficulty to match the checkpoint profile
        env = UAVLiDARDynamicEnv(n_obstacles=current_obstacles)
        
        # Load past history tracking arrays if they exist
        history_path = os.path.join(save_dir, "rewards_history_dynamic.npy")
        if os.path.exists(history_path):
            rewards_history = list(np.load(history_path))
            print(f"-> Loaded {len(rewards_history)} episodes of past reward data.")
            
        # Set epsilon to its lowest stable exploration level since the agent is trained
        agent.epsilon = agent.epsilon_min
    else:
        print("-> No active checkpoint loaded. Starting training sequence from scratch.")

    print(f"\nStarting Adaptive Curriculum. Active Obstacles: {current_obstacles} | Resuming at Ep: {start_episode}\n")

    for episode in tqdm(range(start_episode, episodes + 1)):
        # --- ADAPTIVE CURRICULUM CONTROLLER ---
        if len(success_window) >= 50:
            rolling_sr = sum(success_window) / len(success_window)
            
            # Upgrade obstacle counts if the agent passes the 70% threshold
            if rolling_sr >= 0.70 and current_obstacles < 8:
                current_obstacles += 2
                print(f"\n" + "="*60)
                print(f"  STAGE CLEANED! Rolling Success Rate: {rolling_sr*100:.1f}%")
                print(f"  UPGRADING COLAB ENVIRONMENT TO {current_obstacles} DYNAMIC OBSTACLES")
                print("="*60 + "\n")
                
                env = UAVLiDARDynamicEnv(n_obstacles=current_obstacles)
                success_window.clear()  # Empty window to force a clean test profile

        obs, _ = env.reset()
        frame_stack = deque([obs] * stack_size, maxlen=stack_size)
        state = np.concatenate(list(frame_stack), axis=0)
        
        episode_reward = 0

        while True:
            action = agent.select_action(state)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            frame_stack.append(next_obs)
            next_state = np.concatenate(list(frame_stack), axis=0)

            replay_buffer.push(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward

            if len(replay_buffer) > batch_size:
                agent.train_step(replay_buffer, batch_size)

            if done:
                rewards_history.append(episode_reward)
                success_window.append(1 if info['reached_goal'] else 0)
                break

        agent.decay_epsilon()

        if episode % target_update_frequency == 0:
            agent.update_target_network()

        # Monitoring Printout
        if episode % 20 == 0:
            current_sr = (sum(success_window) / len(success_window)) * 100 if success_window else 0.0
            print(f"Ep {episode:04d} | Obstacles: {current_obstacles} | Rolling SR: {current_sr:5.1f}% | Eps: {agent.epsilon:.3f} | Goal: {info['reached_goal']}")

        # Save Checkpoint out directly to Drive
        if episode % 100 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"dqn_adaptive_obs_{current_obstacles}_ep_{episode}.pth")
            torch.save({
                'episode': episode,
                'obstacles': current_obstacles,
                'model_state_dict': agent.q_network.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
            }, checkpoint_path)
            np.save(os.path.join(save_dir, "rewards_history_dynamic.npy"), np.array(rewards_history))

    print("\nAdaptive Training Complete.")

if __name__ == "__main__":
    # Point directly to your episode 2000 weight path on your drive
    target_checkpoint = "/content/drive/MyDrive/drl-uav-navigation/outputs_dynamic_risk_aware/checkpoints/dqn_dynamic_2000.pth"
    
    train_dqn_dynamic_adaptive_colab(checkpoint_file=target_checkpoint)