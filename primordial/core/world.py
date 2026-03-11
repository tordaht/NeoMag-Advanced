import json
import numpy as np
import taichi as ti
import torch

from . import config as cfg
from .fields import EnvironmentalFields
from .organisms import SpecimenManager
from .renderer import CoreRenderer


@ti.data_oriented
class PrimordialWorld:
    """
    Async-friendly world orchestrator with a stable external data surface.
    """

    TRIBE_NAMES = ("emerald", "amber", "indigo")

    def __init__(self, headless=True, seed=42, capture_behavior_events=False):
        if not ti.lang.impl.get_runtime().prog:
            try:
                ti.init(arch=ti.cuda, device_memory_fraction=0.6, random_seed=seed)
            except Exception:
                ti.init(arch=ti.cpu, random_seed=seed)

        self.fields = EnvironmentalFields()
        self.organisms = SpecimenManager()
        self.renderer = None if headless else CoreRenderer()
        self.step_count = 0
        self.time = 0.0
        self.capture_behavior_events = capture_behavior_events
        self.last_behavior_events = []

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.fields.seed_resources()
        self.organisms.reset_all()
        self.organisms.reset_rl_buffers()
        self.step_count = 0
        self.time = 0.0
        self.organisms.update_spatial_index()
        self.organisms.compute_observations(self.fields, self.organisms.observations)

    def step(self, actions=None):
        pre_state = None
        if actions is not None:
            if isinstance(actions, torch.Tensor):
                action_array = actions.detach().cpu().numpy().astype(np.float32, copy=False)
            else:
                action_array = np.asarray(actions, dtype=np.float32)
            if self.capture_behavior_events:
                pre_state = self._capture_behavior_snapshot(action_array)
            self.organisms.actions.from_numpy(action_array)

        self.organisms.reset_step_counters()
        self.organisms.update_spatial_index()
        self.organisms.apply_repulsion()
        self.fields.update(self.time)

        type_counts = self.organisms.get_type_counts()
        alive_count = int(type_counts[0] + type_counts[1])
        self.organisms.clear_rewards()

        self.organisms.apply_physics_and_metabolism(
            self.fields,
            alive_count,
            int(type_counts[0]),
            int(type_counts[1]),
            self.time,
            self.organisms.actions,
            self.organisms.last_actions,
            self.organisms.rewards,
        )
        self.organisms.execute_reproduction(int(type_counts[0]), int(type_counts[1]))
        self.organisms.compute_observations(self.fields, self.organisms.observations)

        self.step_count += 1
        self.time += 0.01
        if self.capture_behavior_events and pre_state is not None:
            self.last_behavior_events = self._extract_behavior_events(pre_state)
        else:
            self.last_behavior_events = []

    def _capture_behavior_snapshot(self, action_array: np.ndarray):
        return {
            "actions": np.array(action_array, copy=True),
            "alive": self.organisms.alive.to_numpy().astype(np.bool_, copy=False),
            "type": self.organisms.type.to_numpy().astype(np.int32, copy=False),
            "energy": self.organisms.energy.to_numpy().astype(np.float32, copy=False),
            "pos": self.organisms.pos.to_numpy().astype(np.float32, copy=False),
            "dialect": self.organisms.dialect_state.to_numpy().astype(np.float32, copy=False),
            "mimic_cooldown": self.organisms.mimic_cooldown.to_numpy().astype(np.float32, copy=False),
            "active_signal": self.fields.active_signal.to_numpy().astype(np.float32, copy=False),
            "prey_quorum": self.fields.prey_quorum.to_numpy().astype(np.float32, copy=False),
            "pred_quorum": self.fields.pred_quorum.to_numpy().astype(np.float32, copy=False),
            "culture": self.fields.culture.to_numpy().astype(np.float32, copy=False),
        }

    def _dominant_tribes(self, dialect_np: np.ndarray):
        dominant = np.argmax(dialect_np, axis=1)
        return np.asarray([self.TRIBE_NAMES[int(idx)] for idx in dominant], dtype=object)

    def _agent_local_context(self, idx: int, state: dict) -> str:
        pos = state["pos"][idx]
        field_x = int(pos[0] * cfg.FIELD_RES[0] / cfg.WORLD_RES[0]) % cfg.FIELD_RES[0]
        field_y = int(pos[1] * cfg.FIELD_RES[1] / cfg.WORLD_RES[1]) % cfg.FIELD_RES[1]
        alive = state["alive"]
        pos_np = state["pos"]
        dist = np.linalg.norm(pos_np - pos, axis=1)
        neighbors = int(np.count_nonzero(alive & (dist < cfg.INTERACTION_RADIUS * 2.0))) - 1
        culture = state["culture"][field_x, field_y]
        return json.dumps(
            {
                "neighbors": max(0, neighbors),
                "active_signal": float(state["active_signal"][field_x, field_y]),
                "prey_quorum": float(state["prey_quorum"][field_x, field_y]),
                "pred_quorum": float(state["pred_quorum"][field_x, field_y]),
                "culture_norm": float(np.linalg.norm(culture)),
                "mimic_cooldown": float(state["mimic_cooldown"][idx]),
            },
            separators=(",", ":"),
        )

    def _extract_behavior_events(self, state: dict):
        post_alive = self.organisms.alive.to_numpy().astype(np.bool_, copy=False)
        post_energy = self.organisms.energy.to_numpy().astype(np.float32, copy=False)
        post_rewards = self.organisms.rewards.to_numpy().astype(np.float32, copy=False)
        tribe_labels = self._dominant_tribes(state["dialect"])
        pre_alive = state["alive"]
        pre_type = state["type"]
        pre_pos = state["pos"]
        pre_energy = state["energy"]
        actions = state["actions"]

        events = []
        dead_prey = set(
            np.where(pre_alive & (pre_type == cfg.TYPE_PREY) & (~post_alive))[0].astype(int).tolist()
        )

        mimic_agents = np.where(pre_alive & (pre_type == cfg.TYPE_PRED) & (actions[:, 4] > cfg.MIMIC_THRESHOLD))[0]
        for agent_id in mimic_agents.astype(int).tolist():
            context = self._agent_local_context(agent_id, state)
            reward_delta = float(post_rewards[agent_id])
            energy_delta = float(post_energy[agent_id] - pre_energy[agent_id])
            target_id = None
            target_tribe = ""
            success = False

            if state["mimic_cooldown"][agent_id] <= 0.0:
                candidates = [
                    prey_id
                    for prey_id in dead_prey
                    if np.linalg.norm(pre_pos[prey_id] - pre_pos[agent_id]) < cfg.INTERACTION_RADIUS
                ]
                if candidates:
                    target_id = min(candidates, key=lambda prey_id: np.linalg.norm(pre_pos[prey_id] - pre_pos[agent_id]))
                    target_tribe = str(tribe_labels[target_id])
                    success = True
                    dead_prey.discard(target_id)

            events.append(
                {
                    "step": self.step_count,
                    "agent_id": agent_id,
                    "tribe": str(tribe_labels[agent_id]),
                    "event_type": "mimic",
                    "target_agent_id": "" if target_id is None else target_id,
                    "target_tribe": target_tribe,
                    "local_context": context,
                    "reward_delta": reward_delta,
                    "energy_delta": energy_delta,
                    "success": success,
                }
            )

        altruism_agents = np.where(pre_alive & (actions[:, 5] > cfg.ALTRUISM_THRESHOLD))[0]
        for agent_id in altruism_agents.astype(int).tolist():
            context = self._agent_local_context(agent_id, state)
            reward_delta = float(post_rewards[agent_id])
            energy_delta = float(post_energy[agent_id] - pre_energy[agent_id])
            candidates = []
            for other_id in np.where(pre_alive & (pre_type == pre_type[agent_id]))[0].astype(int).tolist():
                if other_id == agent_id:
                    continue
                if np.linalg.norm(pre_pos[other_id] - pre_pos[agent_id]) >= cfg.INTERACTION_RADIUS:
                    continue
                if pre_energy[other_id] >= cfg.ALTRUISM_RESCUE_THRESHOLD:
                    continue
                if (post_energy[other_id] - pre_energy[other_id]) <= 0.25:
                    continue
                candidates.append(other_id)

            if not candidates:
                events.append(
                    {
                        "step": self.step_count,
                        "agent_id": agent_id,
                        "tribe": str(tribe_labels[agent_id]),
                        "event_type": "altruism",
                        "target_agent_id": "",
                        "target_tribe": "",
                        "local_context": context,
                        "reward_delta": reward_delta,
                        "energy_delta": energy_delta,
                        "success": False,
                    }
                )
                continue

            for target_id in candidates:
                events.append(
                    {
                        "step": self.step_count,
                        "agent_id": agent_id,
                        "tribe": str(tribe_labels[agent_id]),
                        "event_type": "altruism",
                        "target_agent_id": target_id,
                        "target_tribe": str(tribe_labels[target_id]),
                        "local_context": context,
                        "reward_delta": reward_delta,
                        "energy_delta": energy_delta,
                        "success": True,
                    }
                )

        return events

    def get_behavior_events(self):
        return list(self.last_behavior_events)

    def render(self, show_mode: int, cam_x: float, cam_y: float, zoom: float):
        if self.renderer:
            self.renderer.render(self.fields, self.organisms, show_mode, cam_x, cam_y, zoom, self.time)
            return self.renderer.display_buffer.to_numpy()
        return None

    def _torch_from_numpy(self, array: np.ndarray, device="cpu") -> torch.Tensor:
        return torch.from_numpy(array).to(device)

    def _safe_normalize_rows(self, values: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-6)
        return values / norms

    def _get_base_observations_np(self) -> np.ndarray:
        observations = self.organisms.observations.to_numpy()
        return observations[:, : cfg.BASE_OBSERVATION_DIM].astype(np.float32, copy=False)

    def _get_culture_context_np(self) -> np.ndarray:
        dialect = self.organisms.dialect_state.to_numpy().astype(np.float32, copy=False)
        local_marker = self.organisms.local_marker_ctx.to_numpy().astype(np.float32, copy=False)
        return np.concatenate((dialect, local_marker), axis=1)

    def _get_epigenetic_context_np(self) -> np.ndarray:
        return self.organisms.epigenetic_ctx.to_numpy().astype(np.float32, copy=False)

    def _get_behavior_context_np(self) -> np.ndarray:
        return self.organisms.behavior_ctx.to_numpy().astype(np.float32, copy=False)

    def _compose_full_observation_np(self) -> np.ndarray:
        return np.concatenate(
            (
                self._get_base_observations_np(),
                self._get_culture_context_np(),
                self._get_epigenetic_context_np(),
                self._get_behavior_context_np(),
            ),
            axis=1,
        ).astype(np.float32, copy=False)

    def get_observations_torch(self, device="cpu"):
        return self._torch_from_numpy(self._compose_full_observation_np(), device=device)

    def get_rewards_torch(self, device="cpu"):
        return self._torch_from_numpy(self.organisms.rewards.to_numpy(), device=device)

    def get_alive_mask_torch(self, device="cpu"):
        alive_np = self.organisms.alive.to_numpy().astype(np.int32, copy=False)
        return self._torch_from_numpy(alive_np, device=device)

    def get_ring_buffer_data(self, rollout: dict | None = None):
        alive_mask = self.organisms.alive.to_numpy().astype(np.bool_, copy=False)
        obs_full = self._compose_full_observation_np()
        rollout = rollout or {}
        return {
            "obs": np.copy(self._get_base_observations_np()),
            "obs_full": np.copy(obs_full),
            "act": np.copy(self.organisms.actions.to_numpy().astype(np.float32, copy=False)),
            "action_mean": np.copy(np.asarray(rollout.get("action_mean", self.organisms.actions.to_numpy()), dtype=np.float32)),
            "log_prob": np.copy(np.asarray(rollout.get("log_prob", np.zeros(cfg.MAX_AGENTS, dtype=np.float32)), dtype=np.float32)),
            "value": np.copy(np.asarray(rollout.get("value", np.zeros(cfg.MAX_AGENTS, dtype=np.float32)), dtype=np.float32)),
            "rew": np.copy(self.organisms.rewards.to_numpy().astype(np.float32, copy=False)),
            "don": np.copy(~alive_mask),
            "culture_ctx": np.copy(self._get_culture_context_np()),
            "epigenetic_ctx": np.copy(self._get_epigenetic_context_np()),
            "behavior_ctx": np.copy(self._get_behavior_context_np()),
        }

    def get_culture_metrics(self):
        metrics = {}
        alive_mask = self.organisms.alive.to_numpy() == 1
        alive_count = int(alive_mask.sum())
        family_count = cfg.CULTURE_CHANNELS

        if alive_count > 0:
            signal_np = self.organisms.signal.to_numpy()[alive_mask]
            energy_np = self.organisms.energy.to_numpy()[alive_mask]
            dialect_np = self.organisms.dialect_state.to_numpy()[alive_mask]
            marker_np = self.organisms.local_marker_ctx.to_numpy()[alive_mask]

            dialect_norm = self._safe_normalize_rows(dialect_np)
            marker_norm = self._safe_normalize_rows(np.maximum(marker_np, 0.0) + 1e-6)

            dominant = np.argmax(dialect_norm, axis=1)
            family_hist = np.bincount(dominant, minlength=family_count)
            centroid = dialect_norm.mean(axis=0, keepdims=True)
            centroid = self._safe_normalize_rows(centroid)[0]
            divergence = np.linalg.norm(dialect_norm - centroid, axis=1)
            entropy_probs = family_hist.astype(np.float32) / max(1, alive_count)
            entropy_probs = entropy_probs[entropy_probs > 1e-8]
            entropy = float(-(entropy_probs * np.log2(entropy_probs)).sum()) if entropy_probs.size else 0.0

            sync_scores = np.sum(dialect_norm * marker_norm, axis=1)
            persistence = np.max(dialect_norm, axis=1)

            metrics["avg_energy"] = float(np.mean(energy_np))
            metrics["signal_density"] = float(np.mean(np.abs(signal_np)))
            metrics["signal_activity"] = float(np.mean(np.clip(signal_np, 0.0, 1.0)))
            metrics["repeated_symbol_families"] = int(np.count_nonzero(family_hist > 1))
            metrics["adoption_rate"] = float(np.mean(np.clip(sync_scores, 0.0, 1.0)))
            metrics["inter_agent_synchronization"] = float(np.mean(sync_scores))
            metrics["cultural_persistence_window"] = float(np.mean(persistence))
            metrics["dialect_entropy"] = entropy
            metrics["dialect_divergence"] = float(np.mean(divergence))

            tribe_names = ("emerald", "amber", "indigo")
            for family_idx, family_total in enumerate(family_hist):
                metrics[f"cultural_family_{family_idx}_count"] = int(family_total)
                if family_idx < len(tribe_names):
                    metrics[f"{tribe_names[family_idx]}_count"] = int(family_total)
                    tribe_mask = dominant == family_idx
                    metrics[f"{tribe_names[family_idx]}_signal_density"] = float(np.mean(np.abs(signal_np[tribe_mask]))) if np.any(tribe_mask) else 0.0

        culture_field = getattr(self.fields, "culture", None)
        if culture_field is not None:
            culture_np = culture_field.to_numpy()
            regional_density = float(np.linalg.norm(culture_np, axis=-1).mean())
            metrics["regional_marker_density"] = regional_density
            metrics["regional_marker_concentration"] = regional_density
            overlap_mask = np.count_nonzero(culture_np > 0.05, axis=-1) > 1
            metrics["territorial_overlap"] = float(np.mean(overlap_mask))
        signal_field = getattr(self.fields, "active_signal", None)
        if signal_field is not None:
            active_signal = signal_field.to_numpy()
            signal_density = float(np.mean(active_signal))
            metrics["active_signal_density"] = signal_density
            metrics["active_signal_concentration"] = signal_density

        for source in (self.organisms, self.fields):
            for getter_name in ("get_culture_metrics", "get_cultural_metrics", "get_behavior_metrics"):
                getter = getattr(source, getter_name, None)
                if not callable(getter):
                    continue
                try:
                    data = getter()
                except Exception:
                    data = None

                if isinstance(data, dict):
                    for key, value in data.items():
                        metrics[key] = value.item() if hasattr(value, "item") else value
        return metrics

    def get_metrics(self):
        type_counts = self.organisms.get_type_counts()
        metrics = {
            "step": self.step_count,
            "prey_count": int(type_counts[0]),
            "pred_count": int(type_counts[1]),
            "alive_count": int(type_counts[0] + type_counts[1]),
            "survival_rate": float((type_counts[0] + type_counts[1]) / max(1, cfg.INITIAL_PREY_COUNT + cfg.INITIAL_PRED_COUNT)),
        }
        metrics.update(self.get_culture_metrics())
        return metrics
