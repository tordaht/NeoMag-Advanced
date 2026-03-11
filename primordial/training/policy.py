from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from primordial.core import config as cfg


_ACTION_LOW = torch.tensor(cfg.ACTION_LOW, dtype=torch.float32)
_ACTION_HIGH = torch.tensor(cfg.ACTION_HIGH, dtype=torch.float32)


@dataclass
class PolicySample:
    action_mean: torch.Tensor
    sampled_action: torch.Tensor
    log_prob: torch.Tensor
    value: torch.Tensor
    entropy: torch.Tensor


class PrimordialPolicy(nn.Module):
    """
    Shared actor-critic policy used by the observatory, async worker, and trainer.
    """

    def __init__(self):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(cfg.OBSERVATION_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.actor_mean = nn.Linear(64, cfg.ACTION_DIM)
        self.actor_log_std = nn.Parameter(torch.full((cfg.ACTION_DIM,), -2.45))
        self.critic = nn.Linear(64, 1)
        self._init_actor_bias()

    def _init_actor_bias(self):
        with torch.no_grad():
            self.actor_mean.bias.zero_()
            self.actor_mean.bias[0] = -1.1   # low initial thrust
            self.actor_mean.bias[1] = 0.0    # neutral steer
            self.actor_mean.bias[2] = -1.6   # low metabolic shift
            self.actor_mean.bias[3] = -2.2   # quiet signal by default
            self.actor_mean.bias[4] = -2.4   # mimic is learned, not random
            self.actor_mean.bias[5] = -2.2   # altruism is learned, not random

    def _action_bounds(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        return _ACTION_LOW.to(device=device), _ACTION_HIGH.to(device=device)

    def bounded_mean(self, logits: torch.Tensor) -> torch.Tensor:
        steer = torch.tanh(logits[..., 1:2])
        forward_parts = [
            torch.sigmoid(logits[..., 0:1]),
            steer,
            torch.sigmoid(logits[..., 2:]),
        ]
        return torch.cat(forward_parts, dim=-1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encode(x)
        mean = self.bounded_mean(self.actor_mean(features))
        value = self.critic(features)
        return mean, value

    def dist(self, x: torch.Tensor):
        mean, value = self.forward(x)
        std = self.actor_log_std.exp().to(device=x.device).expand_as(mean)
        dist = torch.distributions.Independent(torch.distributions.Normal(mean, std), 1)
        return dist, mean, value

    def evaluate_actions(self, x: torch.Tensor, actions: torch.Tensor):
        dist, mean, value = self.dist(x)
        low, high = self._action_bounds(actions.device)
        clipped = torch.clamp(actions, low, high)
        log_prob = dist.log_prob(clipped)
        entropy = dist.entropy()
        return {
            "mean": mean,
            "value": value.squeeze(-1),
            "log_prob": log_prob,
            "entropy": entropy,
        }

    @torch.no_grad()
    def sample_actions(self, x: torch.Tensor) -> PolicySample:
        dist, mean, value = self.dist(x)
        low, high = self._action_bounds(x.device)
        sampled = torch.clamp(dist.sample(), low, high)
        log_prob = dist.log_prob(sampled)
        entropy = dist.entropy()
        return PolicySample(
            action_mean=mean,
            sampled_action=sampled,
            log_prob=log_prob,
            value=value.squeeze(-1),
            entropy=entropy,
        )
