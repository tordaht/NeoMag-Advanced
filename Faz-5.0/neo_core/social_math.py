from __future__ import annotations

import torch
import torch.nn.functional as F


def bits_to_indices(bits: torch.Tensor) -> torch.Tensor:
    powers = torch.tensor([1, 2, 4, 8], device=bits.device, dtype=bits.dtype)
    return (bits * powers).sum(dim=-1).long()


def bits_to_one_hot(bits: torch.Tensor, vocab_size: int = 16) -> torch.Tensor:
    indices = bits_to_indices(bits)
    return F.one_hot(indices, num_classes=vocab_size).to(dtype=bits.dtype)


def straight_through_bits(message_logits: torch.Tensor, tau: float = 0.65) -> tuple[torch.Tensor, torch.Tensor]:
    batch, bits, classes = message_logits.shape
    flat_logits = message_logits.reshape(batch * bits, classes)
    soft = F.gumbel_softmax(flat_logits, tau=tau, hard=False, dim=-1)
    hard = F.gumbel_softmax(flat_logits, tau=tau, hard=True, dim=-1)
    soft = soft.reshape(batch, bits, classes)
    hard = hard.reshape(batch, bits, classes)
    soft_bits = soft[..., 1]
    hard_bits = hard[..., 1]
    straight_through = hard_bits + soft_bits - soft_bits.detach()
    return straight_through, soft_bits


def autoencoder_grounding_loss(obs_embedding: torch.Tensor, reconstructed_embedding: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(reconstructed_embedding, obs_embedding)


def positive_listening_loss(with_message: torch.Tensor, without_message: torch.Tensor) -> torch.Tensor:
    return -(with_message - without_message).abs().mean()


def causal_social_influence_reward(
    conditioned_logits: torch.Tensor,
    marginal_logits: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    conditioned_log_probs = F.log_softmax(conditioned_logits, dim=-1)
    marginal_log_probs = F.log_softmax(marginal_logits, dim=-1)
    conditioned_probs = conditioned_log_probs.exp()
    kl = conditioned_probs * (conditioned_log_probs - marginal_log_probs)
    kl = kl.sum(dim=-1)
    if mask is not None:
        kl = kl * mask.to(dtype=kl.dtype)
        denom = mask.to(dtype=kl.dtype).sum().clamp_min(1.0)
        return kl.sum() / denom
    return kl.mean()


def advantage_alignment(
    ego_advantage: torch.Tensor,
    neighbor_advantage: torch.Tensor,
    memory: torch.Tensor,
    gamma: float,
    alignment_lambda: float,
) -> torch.Tensor:
    return ego_advantage + alignment_lambda * gamma * memory * neighbor_advantage


def interaction_trust_update(trust: torch.Tensor, aligned_advantage: torch.Tensor, eta: float) -> torch.Tensor:
    return torch.clamp(trust + eta * aligned_advantage, -1.0, 1.0)


def gossip_trust_update(trust_ik: torch.Tensor, trust_ij: torch.Tensor, trust_jk: torch.Tensor, beta: float) -> torch.Tensor:
    return (1.0 - beta * trust_ij) * trust_ik + beta * trust_ij * trust_jk
