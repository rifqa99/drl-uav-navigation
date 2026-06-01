import torch
import torch.nn as nn


class PPONetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        # --- Shared Feature Extractor Base ---
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        # --- Actor Network (Policy Head) ---
        self.actor_mean = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

        # Log standard deviation parameter for exploration scaling
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

        # --- Critic Network (Value Head) ---
        self.critic = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        features = self.feature_extractor(state)

        # Compute raw continuous policy heads
        mean = self.actor_mean(features)

        # Bound Thrust directly to [0.0, 1.0] and Torque directly to [-1.0, 1.0]
        thrust_mean = torch.sigmoid(mean[..., 0:1])
        torque_mean = torch.tanh(mean[..., 1:2])
        bounded_mean = torch.cat([thrust_mean, torque_mean], dim=-1)

        std = torch.exp(self.actor_logstd)
        value = self.critic(features)

        return bounded_mean, std, value
