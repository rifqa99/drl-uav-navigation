import os
import torch
import numpy as np
from collections import deque

from env.uav_env_continuous import UAVLiDARContinuousEnv
from agents.ppo_agent import PPOAgent
from agents.ppo_memory import PPOMemory


def train_ppo(checkpoint_to_load=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- Hyperparameters ---
    max_episodes = 3000
    rollout_horizon = 512
    lr = 3e-4
    gamma = 0.99
    K_epochs = 10
    eps_clip = 0.2
    stack_size = 3
    num_obstacles_phase1 = 2
    num_obstacles_phase2 = 8

    save_dir = "/content/drive/MyDrive/drl-uav-navigation/outputs_ppo"
    checkpoint_dir = os.path.join(save_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize at Phase 1 (2 Obstacles)
    env = UAVLiDARContinuousEnv(n_obstacles=0)

    stacked_state_dim = env.observation_space.shape[0] * stack_size
    action_dim = env.action_space.shape[0]

    memory = PPOMemory()
    agent = PPOAgent(
        state_dim=stacked_state_dim, action_dim=action_dim,
        lr=lr, gamma=gamma, K_epochs=K_epochs, eps_clip=eps_clip, device=device
    )

    start_episode = 1
    rewards_history = []

    # NEW: Rolling tracking window to evaluate true success metrics
    success_window = deque(maxlen=100)
    phase_2_activated = False

    if checkpoint_to_load is not None:
        print(f"Loading weights from checkpoint: {checkpoint_to_load}")
        checkpoint = torch.load(
            checkpoint_to_load, map_location=device, weights_only=False)
        agent.policy.load_state_dict(checkpoint["policy_state_dict"])
        agent.policy_old.load_state_dict(checkpoint["policy_state_dict"])
        start_episode = checkpoint.get("episode", start_episode) + 1

        history_path = os.path.join(save_dir, "rewards_history_ppo.npy")
        if os.path.exists(history_path):
            rewards_history = list(np.load(history_path))

    print(
        f"\nStarting Mathematically Aligned PPO Training at Episode: {start_episode}")
    print(f"Initial Obstacle Count: {env.n_obstacles}\n")

    for episode in range(start_episode, max_episodes + 1):

        # --- FIXED: DYNAMIC CURRICULUM TRIGGER ---
        # Only switch to Phase 2 if the agent proves it can hit the goal reliably
        if len(success_window) >= 50 and (sum(success_window) / len(success_window)) >= 0.70 and not phase_2_activated:
            print("\n" + "="*60)
            print(
                f"  CURRICULUM SUCCESS CRITERIA MET ({(sum(success_window)/len(success_window))*100}%)")
            print("  SWITCHING TO CURRICULUM PHASE 2: 8 DENSE OBSTACLES")
            print("="*60 + "\n")
            env = UAVLiDARContinuousEnv(n_obstacles=num_obstacles_phase2)
            phase_2_activated = True
            success_window.clear()  # Reset tracking for the harder map evaluation

        obs, _ = env.reset()
        frame_stack = deque([obs] * stack_size, maxlen=stack_size)
        state = np.concatenate(list(frame_stack), axis=0)

        episode_reward = 0

        while True:
            action, action_logprob, state_value = agent.select_action(state)
            next_obs, reward, terminated, truncated, info = env.step(action)

            # Fix from reviewer: combine terminal flags for timeout tracking
            done = terminated or truncated

            memory.states.append(state)
            memory.actions.append(action)
            memory.logprobs.append(action_logprob)
            memory.values.append(state_value)
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            frame_stack.append(next_obs)
            state = np.concatenate(list(frame_stack), axis=0)

            episode_reward += reward

            if len(memory.states) >= rollout_horizon:
                agent.update(memory)
                memory.clear()

            if terminated or truncated:
                rewards_history.append(episode_reward)
                # Log success flag directly to validation history tracking queue
                success_window.append(1 if info['reached_goal'] else 0)
                break

        if episode % 10 == 0:
            current_sr = (sum(success_window) / len(success_window)
                          ) * 100 if success_window else 0.0
            print(
                f"Ep {episode:04d} | Reward: {episode_reward:7.2f} | Rolling SR: {current_sr:5.1f}% | Goal: {info['reached_goal']} | Speed: {info['speed']:.2f} m/s")

        if episode % 100 == 0:
            checkpoint_path = os.path.join(
                checkpoint_dir, f"ppo_episode_{episode}.pth")
            torch.save({
                'episode': episode,
                'policy_state_dict': agent.policy.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
            }, checkpoint_path)

            np.save(os.path.join(save_dir, "rewards_history_ppo.npy"),
                    np.array(rewards_history))
            print(
                f"-> Secured Checkpoint {episode} and saved arrays to Drive.")

    print("\nTraining completed successfully.")


if __name__ == "__main__":
    train_ppo(checkpoint_to_load=None)
