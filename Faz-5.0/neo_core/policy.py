from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import TrustBiasedGAT
from .config import DEFAULT_CONFIG, NeoCoreConfig
from .social_math import (
    autoencoder_grounding_loss,
    bits_to_one_hot,
    causal_social_influence_reward,
    positive_listening_loss,
    straight_through_bits,
)


@dataclass
class PolicySample:
    env_action_mean: torch.Tensor
    sampled_env_action: torch.Tensor
    message_logits: torch.Tensor
    sampled_message_bits: torch.Tensor
    message_probs: torch.Tensor
    message_symbol: torch.Tensor
    reconstructed_obs: torch.Tensor
    social_context: torch.Tensor
    attention_weights: torch.Tensor
    log_prob: torch.Tensor
    value: torch.Tensor
    entropy: torch.Tensor
    action_mean: torch.Tensor
    sampled_action: torch.Tensor


class NeoCorePolicy(nn.Module):
    def __init__(self, config: NeoCoreConfig = DEFAULT_CONFIG):
        super().__init__()
        self.config = config
        self.obs_encoder = nn.Sequential(
            nn.Linear(config.observation_dim, config.latent_dim),
            nn.SiLU(),
            nn.Linear(config.latent_dim, config.latent_dim),
            nn.SiLU(),
        )
        self.neighbor_project = nn.Linear(config.entity_state_dim + config.syntax_bits, config.gat_hidden_dim)
        self.neighbor_conv = nn.Conv1d(config.gat_hidden_dim, config.gat_hidden_dim, kernel_size=1)
        self.trust_gat = TrustBiasedGAT(
            ego_dim=config.latent_dim,
            neighbor_dim=config.gat_hidden_dim,
            hidden_dim=config.gat_hidden_dim,
            heads=config.gat_heads,
        )

        self.message_head = nn.Sequential(
            nn.Linear(config.latent_dim, config.latent_dim),
            nn.SiLU(),
            nn.Linear(config.latent_dim, config.syntax_bits * 2),
        )
        self.message_decoder = nn.Sequential(
            nn.Linear(config.syntax_vocab_size, 64),
            nn.SiLU(),
            nn.Linear(64, config.latent_dim),
        )

        joint_dim = config.latent_dim + config.gat_hidden_dim
        self.env_actor = nn.Sequential(
            nn.Linear(joint_dim, 128),
            nn.SiLU(),
            nn.Linear(128, config.base_action_dim),
        )
        self.env_actor_log_std = nn.Parameter(torch.full((config.base_action_dim,), -1.8))
        self.critic = nn.Sequential(
            nn.Linear(joint_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )
        self.listener_probe = nn.Linear(joint_dim, config.base_action_dim)
        self._init_bias()

    def _init_bias(self):
        with torch.no_grad():
            final_linear = self.env_actor[-1]
            final_linear.bias.zero_()
            final_linear.bias[0] = -1.1
            final_linear.bias[2] = -0.6
            final_linear.bias[3] = -1.2
            final_linear.bias[4] = -2.0
            final_linear.bias[5] = -1.7

    def _default_social_context(self, observations: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = observations.shape[0]
        k = self.config.fov_neighbors
        device = observations.device
        return {
            "neighbor_state": torch.zeros((batch, k, self.config.entity_state_dim), device=device, dtype=observations.dtype),
            "neighbor_messages": torch.zeros((batch, k, self.config.syntax_bits), device=device, dtype=observations.dtype),
            "neighbor_trust": torch.zeros((batch, k), device=device, dtype=observations.dtype),
            "neighbor_mask": torch.zeros((batch, k), device=device, dtype=torch.bool),
        }

    def _bounded_env_mean(self, logits: torch.Tensor) -> torch.Tensor:
        thrust = torch.sigmoid(logits[..., 0:1])
        steer = torch.tanh(logits[..., 1:2])
        tail = torch.sigmoid(logits[..., 2:])
        return torch.cat((thrust, steer, tail), dim=-1)

    def encode_observation(self, observations: torch.Tensor) -> torch.Tensor:
        return self.obs_encoder(observations)

    def encode_social_context(
        self,
        obs_embedding: torch.Tensor,
        social_context: dict[str, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ctx = social_context or self._default_social_context(obs_embedding)
        neighbor_state = ctx["neighbor_state"]
        neighbor_messages = ctx["neighbor_messages"]
        neighbor_trust = ctx["neighbor_trust"]
        neighbor_mask = ctx["neighbor_mask"]

        neighbor_input = torch.cat((neighbor_state, neighbor_messages), dim=-1)
        neighbor_hidden = self.neighbor_project(neighbor_input)
        conv_in = neighbor_hidden.transpose(1, 2)
        conv_out = self.neighbor_conv(conv_in).transpose(1, 2)
        social_context_vec, attention = self.trust_gat(
            ego_embedding=obs_embedding,
            neighbor_embedding=conv_out,
            trust_scores=neighbor_trust,
            mask=neighbor_mask,
        )
        return social_context_vec, attention

    def _message_distribution(self, obs_embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.message_head(obs_embedding).view(-1, self.config.syntax_bits, 2)
        hard_bits, soft_bits = straight_through_bits(logits, tau=self.config.gumbel_tau)
        symbol = bits_to_one_hot(hard_bits, vocab_size=self.config.syntax_vocab_size)
        reconstructed = self.message_decoder(symbol)
        return logits, hard_bits, soft_bits, reconstructed

    def _joint_features(
        self,
        observations: torch.Tensor,
        social_context: dict[str, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        obs_embedding = self.encode_observation(observations)
        social_vector, attention = self.encode_social_context(obs_embedding, social_context)
        message_logits, hard_bits, soft_bits, reconstructed = self._message_distribution(obs_embedding)
        joint = torch.cat((obs_embedding, social_vector), dim=-1)
        return joint, obs_embedding, message_logits, hard_bits, soft_bits, reconstructed, attention

    def auxiliary_losses(
        self,
        observations: torch.Tensor,
        social_context: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        joint, obs_embedding, message_logits, hard_bits, soft_bits, reconstructed, attention = self._joint_features(observations, social_context)
        with_message_logits = self.listener_probe(joint)

        no_message_context = social_context
        if social_context is not None:
            no_message_context = dict(social_context)
            no_message_context["neighbor_messages"] = torch.zeros_like(social_context["neighbor_messages"])
        joint_no_message, _, _, _, _, _, _ = self._joint_features(observations, no_message_context)
        without_message_logits = self.listener_probe(joint_no_message)

        ae_loss = autoencoder_grounding_loss(obs_embedding, reconstructed)
        pl_loss = positive_listening_loss(with_message_logits, without_message_logits)
        influence_mask = social_context["neighbor_mask"].any(dim=1) if social_context is not None else None
        influence = causal_social_influence_reward(with_message_logits, without_message_logits, influence_mask)
        return {
            "ae_loss": ae_loss,
            "positive_listening_loss": pl_loss,
            "social_influence_reward": influence,
            "message_entropy": -(soft_bits.clamp_min(1e-6).log() * soft_bits).mean(),
            "attention_mean": attention.mean(),
            "message_bits_mean": hard_bits.mean(),
        }

    def forward(
        self,
        observations: torch.Tensor,
        social_context: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        joint, _, _, _, _, _, _ = self._joint_features(observations, social_context)
        env_mean = self._bounded_env_mean(self.env_actor(joint))
        value = self.critic(joint).squeeze(-1)
        return env_mean, value

    def dist(
        self,
        observations: torch.Tensor,
        social_context: dict[str, torch.Tensor] | None = None,
    ):
        joint, obs_embedding, message_logits, hard_bits, soft_bits, reconstructed, attention = self._joint_features(observations, social_context)
        env_mean = self._bounded_env_mean(self.env_actor(joint))
        value = self.critic(joint).squeeze(-1)
        std = self.env_actor_log_std.exp().to(device=observations.device).expand_as(env_mean)
        env_dist = torch.distributions.Independent(torch.distributions.Normal(env_mean, std), 1)
        return {
            "env_dist": env_dist,
            "env_mean": env_mean,
            "value": value,
            "message_logits": message_logits,
            "message_bits": hard_bits,
            "message_probs": soft_bits,
            "message_symbol": bits_to_one_hot(hard_bits, vocab_size=self.config.syntax_vocab_size),
            "reconstructed_obs": reconstructed,
            "attention_weights": attention,
            "obs_embedding": obs_embedding,
        }

    @torch.no_grad()
    def sample_actions(
        self,
        observations: torch.Tensor,
        social_context: dict[str, torch.Tensor] | None = None,
    ) -> PolicySample:
        outputs = self.dist(observations, social_context)
        env_dist = outputs["env_dist"]
        env_sample = env_dist.sample()
        env_sample = env_sample.clone()
        env_sample[..., 0] = env_sample[..., 0].clamp(0.0, 1.0)
        env_sample[..., 1] = env_sample[..., 1].clamp(-1.0, 1.0)
        env_sample[..., 2:] = env_sample[..., 2:].clamp(0.0, 1.0)

        message_bits = outputs["message_bits"]
        message_logits = outputs["message_logits"]
        message_log_probs = F.log_softmax(message_logits, dim=-1)
        selected = torch.stack((1.0 - message_bits, message_bits), dim=-1)
        msg_log_prob = (selected * message_log_probs).sum(dim=(-1, -2))
        msg_entropy = -(message_log_probs.exp() * message_log_probs).sum(dim=-1).sum(dim=-1)

        sampled_action = torch.cat((env_sample, message_bits), dim=-1)
        action_mean = torch.cat((outputs["env_mean"], outputs["message_probs"]), dim=-1)

        return PolicySample(
            env_action_mean=outputs["env_mean"],
            sampled_env_action=env_sample,
            message_logits=message_logits,
            sampled_message_bits=message_bits,
            message_probs=outputs["message_probs"],
            message_symbol=outputs["message_symbol"],
            reconstructed_obs=outputs["reconstructed_obs"],
            social_context=outputs["obs_embedding"],
            attention_weights=outputs["attention_weights"],
            log_prob=env_dist.log_prob(env_sample) + msg_log_prob,
            value=outputs["value"],
            entropy=env_dist.entropy() + msg_entropy,
            action_mean=action_mean,
            sampled_action=sampled_action,
        )
