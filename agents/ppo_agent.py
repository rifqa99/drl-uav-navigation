import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        # --- Shared Feature Extractor Base ---
        # Fits your 3x69 stacked state vector structure cleanly
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        # --- Actor Network (Policy Head) ---
        # Outputs mean vector for [Thrust, Torque]
        self.actor_mean = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

        # Log standard deviation parameter for exploration scaling
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

        # --- Critic Network (Value Head) ---
        # Estimates scalar V(s) state values for Generalized Advantage Estimation (GAE)
        self.critic = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        features = self.feature_extractor(state)

        # Compute policy distribution parameters
        mean = self.actor_mean(features)
        # Apply Tanh to mean[1] (Torque) to bound within [-1, 1], and Sigmoid to mean[0] (Thrust) to bound within [0, 1]
        thrust_mean = torch.sigmoid(mean[..., 0:1])
        torque_mean = torch.tanh(mean[..., 1:2])
        bounded_mean = torch.cat([thrust_mean, torque_mean], dim=-1)

        std = torch.exp(self.actor_logstd)

        # Compute value function baseline
        value = self.critic(features)

        return bounded_mean, std, value


class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, K_epochs=10, eps_clip=0.2, device="cuda"):
        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(state_dim, action_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, state, evaluate=False):
        """Samples continuous actions from the Gaussian policy."""
        state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)

        with torch.no_grad():
            mean, std, value = self.policy_old(state_tensor)

        if evaluate:
            # Under evaluation mode, bypass exploration and return the pure deterministic mean vector
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
        """Optimizes policy using collected memory buffers via PPO Clip surrogate loss."""
        # Convert memory buffer lists directly to structured tensors
        old_states = torch.FloatTensor(np.array(memory.states)).to(self.device)
        old_actions = torch.FloatTensor(
            np.array(memory.actions)).to(self.device)
        old_logprobs = torch.FloatTensor(
            np.array(memory.logprobs)).to(self.device)
        old_values = torch.FloatTensor(np.array(memory.values)).to(self.device)
        rewards = memory.rewards
        is_terminals = memory.is_terminals

        # --- Compute Returns and Normalized Advantages ---
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

        # --- K-Epoch Optimization Cycle ---
        for _ in range(self.K_epochs):
            # Evaluate log probabilities and values under active shifting network weights
            mean, std, values = self.policy(old_states)
            dist = Normal(mean, std)
            logprobs = dist.log_prob(old_actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)

            # Compute probability ratio: r(theta) = pi(a|s) / pi_old(a|s)
            ratios = torch.exp(logprobs - old_logprobs)

            # Actor loss configuration using clipped surrogate target equations
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - self.eps_clip,
                                1.0 + self.eps_clip) * advantages

            # Minimize negative surrogate objective, value error, and add policy entropy to encourage exploration
            loss = (
                -torch.min(surr1, surr2) +
                0.5 * self.MseLoss(values.squeeze(), returns) -
                0.01 * entropy
            )

            # Gradient step optimization execution block
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy current optimal parameters back to synchronize memory reference baselines
        self.policy_old.load_state_dict(self.policy.state_dict())
