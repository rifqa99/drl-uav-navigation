import os
import torch
import numpy as np
from collections import deque

# Cleaner modular imports pointing to your verified project structure
from env.uav_env_continuous import UAVLiDARContinuousEnv
from agents.ppo_agent import PPOAgent
from agents.ppo_memory import PPOMemory


def train_ppo(checkpoint_to_load=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- Hyperparameters ---
    max_episodes = 3000
    # Fixed step collection window size ensuring clean rectangular tensors
    rollout_horizon = 512
    lr = 3e-4
    gamma = 0.99
    K_epochs = 10               # Optimize parameters for 10 epochs per batch
    eps_clip = 0.2              # PPO surrogate loss clipping range
    stack_size = 3
    num_obstacles_phase1 = 2    # Initial easy tracking layout
    num_obstacles_phase2 = 8    # Target dense obstacle testing maze

    # Setup Google Drive output directories
    save_dir = "/content/drive/MyDrive/drl-uav-navigation/outputs_ppo"
    checkpoint_dir = os.path.join(save_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize Continuous Environment (Phase 1)
    env = UAVLiDARContinuousEnv(n_obstacles=num_obstacles_phase1)

    # Calculate dimensions for input tensors (3 frames * 69 features = 207 dim)
    original_obs_dim = env.observation_space.shape[0]
    stacked_state_dim = original_obs_dim * stack_size
    action_dim = env.action_space.shape[0]  # [Thrust, Torque]

    # Instantiate Buffer and PPO Core Agent
    memory = PPOMemory()
    agent = PPOAgent(
        state_dim=stacked_state_dim,
        action_dim=action_dim,
        lr=lr,
        gamma=gamma,
        K_epochs=K_epochs,
        eps_clip=eps_clip,
        device=device
    )

    start_episode = 1
    rewards_history = []

    # Handle Resuming Checkpoints / Phase Changes
    if checkpoint_to_load is not None:
        print(f"Loading weights from checkpoint: {checkpoint_to_load}")
        checkpoint = torch.load(
            checkpoint_to_load, map_location=device, weights_only=False)
        agent.policy.load_state_dict(checkpoint["policy_state_dict"])
        agent.policy_old.load_state_dict(checkpoint["policy_state_dict"])
        start_episode = checkpoint.get("episode", start_episode) + 1

        # Pull past historical tracking arrays safely if they exist
        history_path = os.path.join(save_dir, "rewards_history_ppo.npy")
        if os.path.exists(history_path):
            rewards_history = list(np.load(history_path))
            print(
                f"-> Successfully imported {len(rewards_history)} episodes of past history.")

        # Handle Phase 2 environment adjustment if checkpoint is loaded post-episode 1000
        if start_episode > 1000:
            env = UAVLiDARContinuousEnv(n_obstacles=num_obstacles_phase2)
            print(
                f"-> Curriculum Phase 2 Activated: {num_obstacles_phase2} obstacles initialized.")

    print(f"\nStarting PPO Training at Episode: {start_episode}")
    print(f"Initial Obstacle Count: {env.n_obstacles}\n")

    for episode in range(start_episode, max_episodes + 1):
        # Curriculum Trigger for Phase 2 when training completely from scratch
        if episode == 1001 and checkpoint_to_load is None:
            print("\n" + "="*50)
            print("  SWITCHING TO CURRICULUM PHASE 2: 8 DENSE OBSTACLES")
            print("="*50 + "\n")
            env = UAVLiDARContinuousEnv(n_obstacles=num_obstacles_phase2)

        obs, _ = env.reset()
        frame_stack = deque([obs] * stack_size, maxlen=stack_size)
        state = np.concatenate(list(frame_stack), axis=0)

        episode_reward = 0

        while True:
            action, action_logprob, state_value = agent.select_action(state)
            next_obs, reward, terminated, truncated, info = env.step(action)

            # --- FIXED: Store true 'done' state flag containing timeouts ---
            done = terminated or truncated

            memory.states.append(state)
            memory.actions.append(action)
            memory.logprobs.append(action_logprob)
            memory.values.append(state_value)
            memory.rewards.append(reward)
            # Appending done instead of terminated
            memory.is_terminals.append(done)

            frame_stack.append(next_obs)
            state = np.concatenate(list(frame_stack), axis=0)
            episode_reward += reward

            if len(memory.states) >= rollout_horizon:
                agent.update(memory)
                memory.clear()

            if terminated or truncated:
                rewards_history.append(episode_reward)
                break

        # Console Progress Monitoring
        if episode % 10 == 0:
            print(
                f"Ep {episode:04d} | Reward: {episode_reward:7.2f} | Goal: {info['reached_goal']} | Speed: {info['speed']:.2f} m/s")

        # Save Structural State Checkpoints every 100 episodes
        if episode % 100 == 0:
            checkpoint_path = os.path.join(
                checkpoint_dir, f"ppo_episode_{episode}.pth")
            torch.save({
                'episode': episode,
                'policy_state_dict': agent.policy.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
            }, checkpoint_path)

            # Save historical numpy progression track logs to Drive
            np.save(os.path.join(save_dir, "rewards_history_ppo.npy"),
                    np.array(rewards_history))
            print(
                f"-> Secured Checkpoint {episode} and appended reward history matrices to Drive.")

    print("\nTraining completed successfully.")


if __name__ == "__main__":
    # Start pure Phase 1 from scratch
    train_ppo(checkpoint_to_load=None)
