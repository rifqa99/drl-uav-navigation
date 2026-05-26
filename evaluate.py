import torch
import numpy as np
from collections import deque

from env.uav_env import UAVLiDAREnv
from agents.dqn_agent import DQNAgent


def evaluate():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    checkpoint_path = "outputs/checkpoints/dqn_episode_400.pth"

    env = UAVLiDAREnv(
        seed=123,
        n_lidar=64,  # Explicitly matched
        drag=0.15,
        thrust=0.6,
        max_speed=1.2,
    )

    stack_size = 3
    original_obs_dim = env.observation_space.shape[0]
    stacked_state_dim = original_obs_dim * stack_size
    action_dim = env.action_space.n

    agent = DQNAgent(
        state_dim=stacked_state_dim,  # Fixed shape mismatch
        action_dim=action_dim,
        device=device,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )
    agent.q_network.load_state_dict(checkpoint["q_network"])
    agent.epsilon = 0.0

    episodes = 20
    successes = 0
    collisions = 0
    rewards = []

    for ep in range(episodes):
        obs, _ = env.reset()
        # Create identical frame stack setup for inference phase
        frame_stack = deque([obs] * stack_size, maxlen=stack_size)
        state = np.concatenate(list(frame_stack), axis=0)
        total_reward = 0

        for step in range(env.max_steps):
            action = agent.select_action(state)

            next_obs, reward, terminated, truncated, info = env.step(action)

            frame_stack.append(next_obs)
            state = np.concatenate(list(frame_stack), axis=0)
            total_reward += reward

            if terminated or truncated:
                break

        successes += int(info["reached_goal"])
        collisions += int(info["collision"])
        rewards.append(total_reward)

        print(
            f"Episode {ep+1:02d} | "
            f"Reward: {total_reward:.2f} | "
            f"Success: {info['reached_goal']} | "
            f"Collision: {info['collision']} | "
            f"Steps: {step+1}"
        )

    print("\nEvaluation Summary")
    print("------------------")
    print(f"Success rate: {successes / episodes * 100:.1f}%")
    print(f"Collision rate: {collisions / episodes * 100:.1f}%")
    print(f"Average reward: {np.mean(rewards):.2f}")


if __name__ == "__main__":
    evaluate()
