from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch

from .config import DEFAULT_CONFIG, NeoCoreConfig
from .policy import NeoCorePolicy, PolicySample
from .world import NeoCoreWorld


@dataclass
class RolloutItem:
    observations: torch.Tensor
    neighbor_state: torch.Tensor
    neighbor_messages: torch.Tensor
    neighbor_trust: torch.Tensor
    neighbor_mask: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    values: torch.Tensor
    log_prob: torch.Tensor


class NeoCoreAsyncBridge:
    """
    Honest async CPU bridge for Faz 5.
    Taichi simulation remains separate from the CPU policy/training surface.
    """

    def __init__(self, config: NeoCoreConfig = DEFAULT_CONFIG, capacity: int = 256):
        self.config = config
        self.world = NeoCoreWorld(config=config, prefer_cuda=False)
        self.policy = NeoCorePolicy(config=config).to(torch.device("cpu"))
        self.rollouts: deque[RolloutItem] = deque(maxlen=capacity)

    @torch.no_grad()
    def collect_step(self) -> tuple[PolicySample, dict]:
        obs_cpu = self.world.get_observation_tensor().to("cpu")
        social_context = self.world.get_social_context(device=torch.device("cpu"))
        sample = self.policy.sample_actions(obs_cpu, social_context)
        self.world.step(sample.sampled_action.to(self.world.device))
        aux = self.policy.auxiliary_losses(obs_cpu, social_context)
        self.rollouts.append(
            RolloutItem(
                observations=obs_cpu.clone(),
                neighbor_state=social_context["neighbor_state"].clone(),
                neighbor_messages=social_context["neighbor_messages"].clone(),
                neighbor_trust=social_context["neighbor_trust"].clone(),
                neighbor_mask=social_context["neighbor_mask"].clone(),
                actions=sample.sampled_action.clone(),
                rewards=self.world.rewards.to("cpu").clone(),
                values=sample.value.clone(),
                log_prob=sample.log_prob.clone(),
            )
        )
        metrics = self.world.metrics()
        metrics["ae_loss"] = float(aux["ae_loss"].item())
        metrics["social_influence_reward"] = float(aux["social_influence_reward"].item())
        metrics["positive_listening_loss"] = float(aux["positive_listening_loss"].item())
        return sample, metrics

    def bridge_report(self) -> dict:
        return {
            "bridge_mode": self.world.runtime.bridge_mode,
            "simulation_device": str(self.world.device),
            "policy_device": "cpu",
            "rollout_items": len(self.rollouts),
            "action_dim": self.config.action_dim,
            "syntax_bits": self.config.syntax_bits,
            "architecture": "ae_comm + trust_biased_gat",
        }
