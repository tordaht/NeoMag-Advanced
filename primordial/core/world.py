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

    def __init__(self, headless=True, seed=42):
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
        if actions is not None:
            if isinstance(actions, torch.Tensor):
                action_array = actions.detach().cpu().numpy().astype(np.float32, copy=False)
            else:
                action_array = np.asarray(actions, dtype=np.float32)
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

        culture_field = getattr(self.fields, "culture", None)
        if culture_field is not None:
            culture_np = culture_field.to_numpy()
            regional_density = float(np.linalg.norm(culture_np, axis=-1).mean())
            metrics["regional_marker_density"] = regional_density
            metrics["regional_marker_concentration"] = regional_density
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
