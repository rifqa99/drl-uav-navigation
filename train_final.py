import os
import torch
import numpy as np
from collections import deque
from env.uav_env_dynamic import UAVLiDARDynamicEnv
from agents.dqn_agent import DQNAgent
from utils.reward_shaping import UAVRewardShaping  # Assumes your reward class is here

def train_anti_rotation_risk_aware():
    # --- 1. INITIALIZE DIRECTORIES & HYPERPARAMETERS ---
    save_dir = "./outputs/dynamic_risk"
    os.makedirs(save_dir, exist_ok=True)
    
    # Core Training Configuration Matching Table I
    total_episodes = 6000
    curriculum_threshold = 0.70  # 70% rolling success target
    window_size = 100
    frame_stack_size = 3
    
    # Instantiate Environment & Reward Shaper
    env = UAVLiDARDynamicEnv(n_obstacles=2, seed=42)  # Training starts with 2 obstacles
    reward_shaper = UAVRewardShaping(world_size=env.world_size)
    
    # Calculate dimensional sizes (69 features * 3 frames = 207 inputs)
    state_dim = env.observation_space.shape[0] * frame_stack_size
    action_dim = env.action_space.n
    
    # Instantiate the Dueling Double DQN Agent
    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, device="cuda" if torch.cuda.is_available() else "cpu")
    
    # History queues for performance evaluation and metrics generation
    episode_rewards = []
    success_history = []
    obstacle_history = []
    rolling_success_window = deque(maxlen=window_size)
    
    print(f"-> Starting Anti-Rotation Risk-Aware Pipeline setup.")
    print(f"-> Architecture Input Matrix Dimensions: Expected [{state_dim}], Output Actions: [{action_dim}].")
    
    # --- 2. CORE TRAINING LOOP ---
    for episode in range(1, total_episodes + 1):
        # Reset environment with localized curriculum parameters
        obs, _ = env.reset()
        
        # Initialize temporal frame stacking history queue
        frame_stack = deque([obs] * frame_stack_size, maxlen=frame_stack_size)
        state = np.concatenate(list(frame_stack), axis=0)
        
        ep_reward = 0.0
        done = False
        
        while not done:
            # Epsilon-Greedy Action Selection
            action = agent.select_action(state)
            
            # Environment step (extracting raw transition)
            next_obs, base_reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Extract tracking variables for Kinematic and Proximity Constraints
            progress = info.get("progress", 0.0)
            collision = info.get("collision", False)
            reached_goal = info.get("reached_goal", False)
            
            # Extract current speed vector magnitude directly from environment registers
            current_speed = np.linalg.norm(env.vel) if hasattr(env, "vel") else info.get("speed", 0.0)
            current_omega = env.omega if hasattr(env, "omega") else info.get("omega", 0.0)
            
            # --- OVERRIDE BASE REWARD WITH CRITICAL CONSTRAINT REWARD ---
            custom_reward = reward_shaper.compute_reward(
                progress=progress,
                action=action,
                lidar_readings=next_obs[:64],  # Slices raw 64-beam LiDAR vector
                collision=collision,
                reached_goal=reached_goal,
                speed=current_speed,
                omega=current_omega
            )
            
            # Manage frame stack projection pipeline
            frame_stack.append(next_obs)
            next_state = np.concatenate(list(frame_stack), axis=0)
            
            # Store transition tuple into memory buffer and optimize network weights
            agent.remember(state, action, custom_reward, next_state, done)
            agent.train_step()
            
            state = next_state
            ep_reward += custom_reward
            
        # --- 3. EVALUATE CURRICULUM TRANSITION STAGE ---
        reached_goal_status = info.get("reached_goal", False)
        rolling_success_window.append(1.0 if reached_goal_status else 0.0)
        current_sr = np.mean(rolling_success_window)
        
        episode_rewards.append(ep_reward)
        success_history.append(1.0 if reached_goal_status else 0.0)
        obstacle_history.append(env.n_obstacles)
        
        # Check if agent qualifies for obstacle increment ($2 \rightarrow 4 \rightarrow 6 \rightarrow 8$)
        if len(rolling_success_window) == window_size and current_sr >= curriculum_threshold:
            if env.n_obstacles < 8:  # Maximum benchmark ceiling is 8 dynamic hazards
                env.n_obstacles += 2
                rolling_success_window.clear()  # Clear rolling success rate metric tracking window for new phase
                print(f"\n[CURRICULUM MILESTONE COMPLETE]: Scaled complexity to {env.n_obstacles} Obstacles at Episode {episode}!")
        
        # Terminal print tracking status
        if episode % 20 == 0:
            print(f"Episode: {episode}/{total_episodes} | Reward: {ep_reward:.2f} | Rolling SR: {current_sr*100:.1f}% | Active Obs: {env.n_obstacles}")
            
        # --- 4. PERIODIC BACKUP SAVE CHECKS ---
        if episode % 500 == 0 or episode == total_episodes:
            checkpoint_path = os.path.join(save_dir, f"dqn_anti_rotation_ep_{episode}.pth")
            torch.save({
                'episode': episode,
                'model_state_dict': agent.q_network.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'obstacle_count': env.n_obstacles
            }, checkpoint_path)
            
    # Save the absolute historical traces to produce your analytical IEEE charts
    np.save(os.path.join(save_dir, "rewards_history_dynamic.npy"), np.array(episode_rewards))
    np.save(os.path.join(save_dir, "success_history_dynamic.npy"), np.array(success_history))
    np.save(os.path.join(save_dir, "obstacle_history_dynamic.npy"), np.array(obstacle_history))
    print("-> System training pipeline saved successfully.")

if __name__ == "__main__":
    train_anti_rotation_risk_aware()