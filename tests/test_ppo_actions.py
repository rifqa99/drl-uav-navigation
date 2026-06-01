import torch
import numpy as np
from collections import deque

from env.uav_env_continuous import UAVLiDARContinuousEnv
from agents.ppo_agent import PPOAgent


device = "cuda" if torch.cuda.is_available() else "cpu"

env = UAVLiDARContinuousEnv(
    n_obstacles=0
)

stack_size = 3

obs, _ = env.reset()

frame_stack = deque([obs] * stack_size, maxlen=stack_size)

state = np.concatenate(list(frame_stack), axis=0)

state_dim = len(state)
action_dim = env.action_space.shape[0]

agent = PPOAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    device=device
)

print("\nSTART:", env.pos)
print("GOAL :", env.goal)

for step in range(200):

    action, logprob, value = agent.select_action(state)

    thrust = action[0]
    torque = action[1]

    next_obs, reward, terminated, truncated, info = env.step(action)

    distance = np.linalg.norm(env.goal - env.pos)

    speed = np.linalg.norm(env.vel)

    print(
        f"Step {step:03d} | "
        f"Thrust={thrust:.3f} | "
        f"Torque={torque:.3f} | "
        f"Speed={speed:.3f} | "
        f"Dist={distance:.3f}"
    )

    frame_stack.append(next_obs)
    state = np.concatenate(list(frame_stack), axis=0)

    if terminated or truncated:
        print("\nDONE")
        print(info)
        break
