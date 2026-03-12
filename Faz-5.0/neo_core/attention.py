from __future__ import annotations

import math

import torch
import torch.nn as nn


class TrustBiasedGAT(nn.Module):
    def __init__(self, ego_dim: int, neighbor_dim: int, hidden_dim: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // heads
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads")

        self.query = nn.Linear(ego_dim, hidden_dim)
        self.key = nn.Linear(neighbor_dim, hidden_dim)
        self.value = nn.Linear(neighbor_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.trust_gain = nn.Parameter(torch.tensor(1.25))

    def forward(
        self,
        ego_embedding: torch.Tensor,
        neighbor_embedding: torch.Tensor,
        trust_scores: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, neighbor_count, _ = neighbor_embedding.shape
        query = self.query(ego_embedding).view(batch, self.heads, self.head_dim)
        key = self.key(neighbor_embedding).view(batch, neighbor_count, self.heads, self.head_dim)
        value = self.value(neighbor_embedding).view(batch, neighbor_count, self.heads, self.head_dim)

        logits = torch.einsum("bhd,bnhd->bhn", query, key) / math.sqrt(self.head_dim)
        logits = logits + trust_scores.unsqueeze(1) * self.trust_gain
        invalid = ~mask.unsqueeze(1)
        logits = logits.masked_fill(invalid, -1e9)

        attention = torch.softmax(logits, dim=-1)
        attention = attention.masked_fill(invalid, 0.0)
        denom = attention.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        attention = attention / denom

        context = torch.einsum("bhn,bnhd->bhd", attention, value).reshape(batch, self.hidden_dim)
        return self.out(context), attention
