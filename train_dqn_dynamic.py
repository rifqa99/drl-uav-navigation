import os
import torch
import numpy as np
from collections import deque
from tqdm import tqdm

from env.uav_env_dynamic import UAVLiDARDynamicEnv
from agents.dqn_agent import DQNAgent
from agents.replay_buffer import ReplayBuffer
from env.rewards import UAVRewardShaping 

def train_dqn_dynamic_adaptive_colab(checkpoint_file=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- Hyperparameters ---
    episodes = 6000  
    batch_size = 64
    gamma = 0.99
    lr = 1e-4
    stack_size = 3
    buffer_capacity = 50000
    target_update_frequency = 10

    # Permanent Google Drive paths for the risk-aware framework run
    save_dir = "/content/drive/MyDrive/drl-uav-navigation/outputs_dynamic_risk_aware"
    checkpoint_dir = os.path.join(save_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    current_obstacles = 2
    start_episode = 1
    
    # --- CORE PERFORMANCE HISTORY ARRAYS ---
    rewards_history = []
    loss_history = []
    success_history = []     
    obstacle_history = []    
    success_window = deque(maxlen=100)

    # --- ADVANCED TELEMETRY METRICS TRACKERS ---
    min_proximity_history = []   # Stores the absolute closest approach (d_min) per episode
    total_rotation_history = []  # Stores total count of rotation actions per episode

    # Initialize environment and custom reward shaper
    env = UAVLiDARDynamicEnv(n_obstacles=current_obstacles)
    reward_shaper = UAVRewardShaping(world_size=env.world_size)
    
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

    # --- LOAD CHECKPOINT ARTIFACTS IF RESUMING ---
    if checkpoint_file and os.path.exists(checkpoint_file):
        print(f"-> Loading weights from checkpoint: {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
        agent.q_network.load_state_dict(checkpoint["model_state_dict"])
        agent.target_network.load_state_dict(checkpoint["model_state_dict"])
        
        start_episode = checkpoint.get("episode", 2000) + 1
        current_obstacles = checkpoint.get("obstacles", 2)
        
        env = UAVLiDARDynamicEnv(n_obstacles=current_obstacles)
        reward_shaper = UAVRewardShaping(world_size=env.world_size)
        
        # Reload continuous metric historical profiles if they exist on Drive
        if os.path.exists(os.path.join(save_dir, "rewards_history_dynamic.npy")):
            rewards_history = list(np.load(os.path.join(save_dir, "rewards_history_dynamic.npy")))
        if os.path.exists(os.path.join(save_dir, "loss_history_dynamic.npy")):
            loss_history = list(np.load(os.path.join(save_dir, "loss_history_dynamic.npy")))
        if os.path.exists(os.path.join(save_dir, "success_history_dynamic.npy")):
            success_history = list(np.load(os.path.join(save_dir, "success_history_dynamic.npy")))
        if os.path.exists(os.path.join(save_dir, "obstacle_history_dynamic.npy")):
            obstacle_history = list(np.load(os.path.join(save_dir, "obstacle_history_dynamic.npy")))
        if os.path.exists(os.path.join(save_dir, "min_proximity_history.npy")):
            min_proximity_history = list(np.load(os.path.join(save_dir, "min_proximity_history.npy")))
        if os.path.exists(os.path.join(save_dir, "total_rotation_history.npy")):
            total_rotation_history = list(np.load(os.path.join(save_dir, "total_rotation_history.npy")))
            
        print(f"-> Resuming with historic checkpoints loaded successfully.")
        agent.epsilon = agent.epsilon_min
    else:
        print("-> No active checkpoint loaded. Starting training sequence from scratch.")

    print(f"\nStarting Risk-Aware Adaptive Curriculum. Active Obstacles: {current_obstacles} | Resuming at Ep: {start_episode}\n")

    for episode in tqdm(range(start_episode, episodes + 1)):
        # --- ADAPTIVE CURRICULUM CONTROLLER ---
        if len(success_window) >= 50:
            rolling_sr = sum(success_window) / len(success_window)
            
            if rolling_sr >= 0.70 and current_obstacles < 8:
                current_obstacles += 2
                print(f"\n" + "="*60)
                print(f"  STAGE CLEANED! Rolling Success Rate: {rolling_sr*100:.1f}%")
                print(f"  UPGRADING COLAB ENVIRONMENT TO {current_obstacles} DYNAMIC OBSTACLES")
                print("="*60 + "\n")
                
                env = UAVLiDARDynamicEnv(n_obstacles=current_obstacles)
                reward_shaper = UAVRewardShaping(world_size=env.world_size)
                success_window.clear()  

        obs, _ = env.reset()
        frame_stack = deque([obs] * stack_size, maxlen=stack_size)
        state = np.concatenate(list(frame_stack), axis=0)
        
        episode_reward = 0
        episode_losses = []  
        
        # Local per-flight step parameters initialized
        episode_min_proximity = float('inf') 
        episode_total_rotation = 0            

        while True:
            action = agent.select_action(state)
            next_obs, env_reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Handle type-safety checking components for reward script processing
            if 'raw_lidar' not in info:
                info['raw_lidar'] = next_obs  
            
            info['reached_goal'] = info.get('reached_goal', False) or (done and not info.get('collision', False))
            info['collision'] = info.get('collision', False)

            # --- ADVANCED TELEMETRY METRIC COLLECTION ---
            if info['raw_lidar'] is not None and len(info['raw_lidar']) > 0:
                frame_min_lidar = float(np.min(info['raw_lidar']))
                if frame_min_lidar < episode_min_proximity:
                    episode_min_proximity = frame_min_lidar
            
            # Count rotational behavior (assuming indices 3 and 4 are Turn actions)
            if action in [3, 4]:
                episode_total_rotation += 1

            # Intercept and compute reward through our dual-compatible custom shaper
            custom_reward = reward_shaper.compute_reward(info, action_idx=action)

            frame_stack.append(next_obs)
            next_state = np.concatenate(list(frame_stack), axis=0)

            replay_buffer.push(state, action, custom_reward, next_state, done)
            state = next_state
            episode_reward += custom_reward

            if len(replay_buffer) > batch_size:
                loss = agent.train_step(replay_buffer, batch_size)
                if loss is not None:
                    episode_losses.append(loss)

            if done:
                rewards_history.append(episode_reward)
                obstacle_history.append(current_obstacles)
                min_proximity_history.append(episode_min_proximity if episode_min_proximity != float('inf') else 0.0)
                total_rotation_history.append(episode_total_rotation)
                
                is_success = 1 if info['reached_goal'] else 0
                success_history.append(is_success)
                success_window.append(is_success)
                
                avg_episode_loss = float(np.mean(episode_losses)) if episode_losses else 0.0
                loss_history.append(avg_episode_loss)
                break

        agent.decay_epsilon()

        if episode % target_update_frequency == 0:
            agent.update_target_network()

        # Monitoring Printout
        if episode % 20 == 0:
            current_sr = (sum(success_window) / len(success_window)) * 100 if success_window else 0.0
            print(f"Ep {episode:04d} | Obs: {current_obstacles} | Rolling SR: {current_sr:5.1f}% | Avg Loss: {loss_history[-1]:.4f} | d_min: {min_proximity_history[-1]:.3f}m | Goal: {info['reached_goal']}")

        # Flush tracking profiles directly out to Google Drive
        if episode % 100 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"dqn_adaptive_obs_{current_obstacles}_ep_{episode}.pth")
            torch.save({
                'episode': episode,
                'obstacles': current_obstacles,
                'model_state_dict': agent.q_network.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
            }, checkpoint_path)
            
            np.save(os.path.join(save_dir, "rewards_history_dynamic.npy"), np.array(rewards_history))
            np.save(os.path.join(save_dir, "loss_history_dynamic.npy"), np.array(loss_history))
            np.save(os.path.join(save_dir, "success_history_dynamic.npy"), np.array(success_history))
            np.save(os.path.join(save_dir, "obstacle_history_dynamic.npy"), np.array(obstacle_history))
            np.save(os.path.join(save_dir, "min_proximity_history.npy"), np.array(min_proximity_history))
            np.save(os.path.join(save_dir, "total_rotation_history.npy"), np.array(total_rotation_history))

    print("\nAdaptive Training Complete.")

if __name__ == "__main__":
    # Specify checkpoint path here to resume, or leave as None to run fresh
    train_dqn_dynamic_adaptive_colab(checkpoint_file=None)