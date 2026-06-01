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

        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

        self.critic = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        features = self.feature_extractor(state)

        # Output raw, unconstrained mean values on the real number line
        mean = self.actor_mean(features)
        std = torch.exp(self.actor_logstd)
        value = self.critic(features)

        return mean, std, value
