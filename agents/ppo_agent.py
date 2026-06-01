import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

# --- NEW IMPORT LAYER FROM YOUR MODELS DIRECTORY ---
from models.ppo_network import PPONetwork


class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, K_epochs=10, eps_clip=0.2, device="cuda"):
        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        # Point directly to your newly created model file classes
        self.policy = PPONetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = PPONetwork(state_dim, action_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, state, evaluate=False):
        state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)

        with torch.no_grad():
            mean, std, value = self.policy_old(state_tensor)

        if evaluate:
            return mean.cpu().data.numpy().flatten(), 0.0, 0.0

        dist = Normal(mean, std)
        action = dist.sample()
        action_logprob = dist.log_prob(action).sum(dim=-1)

        return (
            np.clip(action.cpu().data.numpy().flatten(),
                    [0.0, -1.0], [1.0, 1.0]),
            action_logprob.item(),
            value.item()
        )

    def update(self, memory):
        old_states = torch.FloatTensor(np.array(memory.states)).to(self.device)
        old_actions = torch.FloatTensor(
            np.array(memory.actions)).to(self.device)
        old_logprobs = torch.FloatTensor(
            np.array(memory.logprobs)).to(self.device)
        old_values = torch.FloatTensor(np.array(memory.values)).to(self.device)
        rewards = memory.rewards
        is_terminals = memory.is_terminals

        returns = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(rewards), reversed(is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)

        returns = torch.FloatTensor(returns).to(self.device)
        advantages = returns - old_values
        advantages = (advantages - advantages.mean()) / \
            (advantages.std() + 1e-7)

        for _ in range(self.K_epochs):
            mean, std, values = self.policy(old_states)
            dist = Normal(mean, std)
            logprobs = dist.log_prob(old_actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)

            ratios = torch.exp(logprobs - old_logprobs)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - self.eps_clip,
                                1.0 + self.eps_clip) * advantages

            loss = (
                -torch.min(surr1, surr2) +
                0.5 * self.MseLoss(values.squeeze(), returns) -
                0.01 * entropy
            )

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
