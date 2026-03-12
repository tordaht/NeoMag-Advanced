from __future__ import annotations

from dataclasses import asdict

import torch

from .config import DEFAULT_CONFIG, NeoCoreConfig
from .kernels import (
    advance_agents,
    apply_social_interactions,
    build_observations,
    decay_trust,
    reset_agents,
)
from .runtime import RuntimeInfo, ensure_runtime


class NeoCoreWorld:
    def __init__(self, config: NeoCoreConfig = DEFAULT_CONFIG, prefer_cuda: bool = False):
        self.config = config
        self.runtime: RuntimeInfo = ensure_runtime(prefer_cuda=prefer_cuda)
        self.device = torch.device(self.runtime.torch_device)
        n = config.max_agents

        self.alive = torch.zeros((n,), dtype=torch.int32, device=self.device)
        self.tribe = torch.zeros((n,), dtype=torch.int32, device=self.device)
        self.pos = torch.zeros((n, 2), dtype=torch.float32, device=self.device)
        self.vel = torch.zeros((n, 2), dtype=torch.float32, device=self.device)
        self.energy = torch.zeros((n,), dtype=torch.float32, device=self.device)
        self.actions = torch.zeros((n, config.action_dim), dtype=torch.float32, device=self.device)
        self.rewards = torch.zeros((n,), dtype=torch.float32, device=self.device)
        self.observations = torch.zeros((n, config.observation_dim), dtype=torch.float32, device=self.device)
        self.trust_matrix = torch.zeros((n, n), dtype=torch.float32, device=self.device)
        self.nearest_idx = torch.full((n,), -1, dtype=torch.int32, device=self.device)
        self.nearest_dist = torch.full((n,), 1e6, dtype=torch.float32, device=self.device)
        self.interaction_delta = torch.zeros((n,), dtype=torch.float32, device=self.device)
        self.received_syntax = torch.zeros((n, config.syntax_bits), dtype=torch.float32, device=self.device)
        self.syntax_code = torch.zeros((n,), dtype=torch.int32, device=self.device)
        self.step_count = 0
        self.reset()

    def reset(self):
        reset_agents(
            self.config.max_agents,
            self.config.initial_agents,
            self.config.world_width,
            self.config.world_height,
            self.config.base_energy,
            self.alive,
            self.tribe,
            self.pos,
            self.vel,
            self.energy,
            self.actions,
            self.rewards,
            self.observations,
            self.trust_matrix,
            self.nearest_idx,
            self.nearest_dist,
            self.interaction_delta,
            self.received_syntax,
            self.syntax_code,
        )
        self.step_count = 0
        self._rebuild_observations()
        return self.observations

    def _rebuild_observations(self):
        build_observations(
            self.config.max_agents,
            self.config.world_width,
            self.config.world_height,
            self.config.distance_norm,
            self.config.trust_clip_scale,
            self.alive,
            self.tribe,
            self.pos,
            self.vel,
            self.energy,
            self.trust_matrix,
            self.nearest_idx,
            self.nearest_dist,
            self.interaction_delta,
            self.received_syntax,
            self.observations,
        )

    def step(self, actions: torch.Tensor | None = None):
        if actions is not None:
            if actions.device != self.device:
                actions = actions.to(self.device)
            self.actions.copy_(actions)

        decay_trust(self.config.max_agents, self.config.trust_decay, self.trust_matrix)
        advance_agents(
            self.config.max_agents,
            self.config.world_width,
            self.config.world_height,
            self.config.max_speed,
            self.config.damping,
            self.config.thrust_gain,
            self.config.steering_gain,
            self.config.energy_drain,
            self.config.signal_cost,
            self.alive,
            self.pos,
            self.vel,
            self.energy,
            self.actions,
            self.rewards,
        )
        apply_social_interactions(
            self.config.max_agents,
            self.config.interaction_radius,
            self.config.altruism_threshold,
            self.config.mimic_threshold,
            self.config.signal_threshold,
            self.config.trust_min,
            self.config.trust_max,
            self.config.trust_altruism_bonus,
            self.config.trust_mimic_penalty,
            self.config.same_tribe_mimic_factor,
            self.config.altruism_transfer_amount,
            self.config.altruism_transfer_tax,
            self.config.minimum_self_energy,
            self.alive,
            self.tribe,
            self.pos,
            self.energy,
            self.actions,
            self.trust_matrix,
            self.nearest_idx,
            self.nearest_dist,
            self.interaction_delta,
            self.received_syntax,
            self.syntax_code,
            self.rewards,
        )
        self._rebuild_observations()
        self.step_count += 1
        return self.observations, self.rewards

    def get_observation_tensor(self) -> torch.Tensor:
        return self.observations

    def get_action_tensor(self) -> torch.Tensor:
        return self.actions

    def get_trust_tensor(self) -> torch.Tensor:
        return self.trust_matrix

    def get_social_context(self, device: torch.device | None = None) -> dict[str, torch.Tensor]:
        device = device or self.device
        pos = self.pos.to(device=device, dtype=torch.float32)
        vel = self.vel.to(device=device, dtype=torch.float32)
        energy = self.energy.to(device=device, dtype=torch.float32)
        tribe = self.tribe.to(device=device)
        alive = (self.alive > 0).to(device=device)
        actions = self.actions.to(device=device, dtype=torch.float32)
        trust = self.trust_matrix.to(device=device, dtype=torch.float32)

        n = pos.shape[0]
        k = min(self.config.fov_neighbors, max(1, n - 1))
        dist = torch.cdist(pos, pos)
        invalid = (~alive).unsqueeze(0) | (~alive).unsqueeze(1)
        dist = dist.masked_fill(invalid, float("inf"))
        dist.fill_diagonal_(float("inf"))

        neighbor_dist, neighbor_ids = torch.topk(dist, k=k, dim=1, largest=False)
        neighbor_mask = torch.isfinite(neighbor_dist) & (neighbor_dist <= self.config.fov_radius) & alive.unsqueeze(1)

        safe_ids = neighbor_ids.clamp_min(0)
        neighbor_pos = pos[safe_ids]
        neighbor_vel = vel[safe_ids]
        neighbor_energy = energy[safe_ids]
        neighbor_tribe = tribe[safe_ids]
        neighbor_signal = actions[safe_ids, 3]
        neighbor_messages = actions[safe_ids, 6 : 6 + self.config.syntax_bits]
        neighbor_messages = torch.where(
            (neighbor_signal > self.config.signal_threshold).unsqueeze(-1),
            neighbor_messages,
            torch.zeros_like(neighbor_messages),
        )
        neighbor_trust = trust.gather(1, safe_ids)

        rel_pos = neighbor_pos - pos.unsqueeze(1)
        rel_vel = neighbor_vel - vel.unsqueeze(1)
        same_tribe = (neighbor_tribe == tribe.unsqueeze(1)).to(torch.float32)
        distance_norm = (neighbor_dist / self.config.fov_radius).clamp(0.0, 1.0)
        neighbor_state = torch.stack(
            (
                rel_pos[..., 0] / self.config.world_width,
                rel_pos[..., 1] / self.config.world_height,
                rel_vel[..., 0] / max(self.config.max_speed, 1e-6),
                rel_vel[..., 1] / max(self.config.max_speed, 1e-6),
                (neighbor_energy / 100.0).clamp(0.0, 2.0),
                same_tribe,
                distance_norm,
                neighbor_signal.clamp(0.0, 1.0),
            ),
            dim=-1,
        )

        neighbor_state = neighbor_state * neighbor_mask.unsqueeze(-1)
        neighbor_messages = neighbor_messages * neighbor_mask.unsqueeze(-1)
        neighbor_trust = neighbor_trust * neighbor_mask
        return {
            "neighbor_state": neighbor_state,
            "neighbor_messages": neighbor_messages,
            "neighbor_trust": neighbor_trust,
            "neighbor_mask": neighbor_mask,
            "neighbor_ids": neighbor_ids,
        }

    def runtime_report(self) -> dict:
        report = asdict(self.runtime)
        report["torch_cuda_available"] = bool(torch.cuda.is_available())
        report["world_device"] = str(self.device)
        report["step_count"] = int(self.step_count)
        return report

    def metrics(self) -> dict:
        alive_mask = self.alive > 0
        alive_count = int(alive_mask.sum().item())
        if alive_count == 0:
            return {
                "alive_count": 0,
                "avg_energy": 0.0,
                "trust_mean": 0.0,
                "trust_min": 0.0,
                "trust_max": 0.0,
                "interaction_delta_mean": 0.0,
            }

        trust_active = self.trust_matrix[: self.config.initial_agents, : self.config.initial_agents]
        syntax_active = self.syntax_code[: self.config.initial_agents]
        unique_codes = int(torch.unique(syntax_active).numel())
        return {
            "alive_count": alive_count,
            "avg_energy": float(self.energy[alive_mask].mean().item()),
            "trust_mean": float(trust_active.mean().item()),
            "trust_min": float(trust_active.min().item()),
            "trust_max": float(trust_active.max().item()),
            "interaction_delta_mean": float(self.interaction_delta[alive_mask].mean().item()),
            "syntax_active_codes": unique_codes,
            "syntax_code_max": int(syntax_active.max().item()),
            "fov_neighbors": int(self.config.fov_neighbors),
        }
