import torch
import torch.nn as nn


class PPONetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        self.actor_mean = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

        self.actor_logstd = nn.Parameter(torch.full((1, action_dim), -1.0))
        self.critic = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        features = self.feature_extractor(state)
        mean = self.actor_mean(features)

        # Graceful continuous activation capping mappings
        thrust_mean = torch.sigmoid(mean[..., 0:1])  # Strict [0.0, 1.0] bound
        torque_mean = torch.tanh(mean[..., 1:2])     # Strict [-1.0, 1.0] bound
        bounded_mean = torch.cat([thrust_mean, torque_mean], dim=-1)

        logstd = torch.clamp(self.actor_logstd, -2.0, 0.0)
        std = torch.exp(logstd).expand_as(bounded_mean)
        value = self.critic(features)

        return bounded_mean, std, value
