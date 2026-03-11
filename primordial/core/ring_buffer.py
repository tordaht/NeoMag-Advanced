import threading
from typing import Dict

import numpy as np

from primordial.core import config as cfg


class ReplayRingBuffer:
    """
    Context-aware rollout buffer used by both the observatory worker and the
    standalone trainer.

    The buffer stores chronological step windows. It remains backward compatible
    with the older dict payloads while carrying the additional PPO-lite fields
    required for honest policy updates.
    """

    def __init__(self, capacity: int, num_agents: int, obs_dim: int, act_dim: int):
        self.capacity = capacity
        self.num_agents = num_agents
        self.full_obs_dim = cfg.OBSERVATION_DIM

        self.observations = np.zeros((capacity, num_agents, obs_dim), dtype=np.float32)
        self.observations_full = np.zeros((capacity, num_agents, self.full_obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, num_agents, act_dim), dtype=np.float32)
        self.action_means = np.zeros((capacity, num_agents, act_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, num_agents), dtype=np.float32)
        self.log_probs = np.zeros((capacity, num_agents), dtype=np.float32)
        self.values = np.zeros((capacity, num_agents), dtype=np.float32)
        self.dones = np.zeros((capacity, num_agents), dtype=np.bool_)

        self.culture_ctx = np.zeros((capacity, num_agents, cfg.CULTURE_CONTEXT_DIM), dtype=np.float32)
        self.epigenetic_ctx = np.zeros((capacity, num_agents, cfg.EPIGENETIC_DIM), dtype=np.float32)
        self.behavior_ctx = np.zeros((capacity, num_agents, cfg.BEHAVIOR_DIM), dtype=np.float32)

        self.ptr = 0
        self.size = 0
        self.lock = threading.Lock()

    def add(self, obs, act=None, rew=None, done=None, culture=None, epi=None, beh=None):
        """
        Supports both legacy positional arguments and dict payloads emitted by
        world.get_ring_buffer_data().
        """
        with self.lock:
            if isinstance(obs, dict):
                self.observations[self.ptr] = obs["obs"]
                self.observations_full[self.ptr] = obs.get("obs_full", 0.0)
                self.actions[self.ptr] = obs["act"]
                self.action_means[self.ptr] = obs.get("action_mean", obs["act"])
                self.rewards[self.ptr] = obs["rew"]
                self.log_probs[self.ptr] = obs.get("log_prob", 0.0)
                self.values[self.ptr] = obs.get("value", 0.0)
                self.dones[self.ptr] = obs["don"]
                self.culture_ctx[self.ptr] = obs.get("culture_ctx", 0.0)
                self.epigenetic_ctx[self.ptr] = obs.get("epigenetic_ctx", 0.0)
                self.behavior_ctx[self.ptr] = obs.get("behavior_ctx", 0.0)
            else:
                self.observations[self.ptr] = obs
                self.observations_full[self.ptr] = 0.0
                if act is not None:
                    self.actions[self.ptr] = act
                    self.action_means[self.ptr] = act
                if rew is not None:
                    self.rewards[self.ptr] = rew
                if done is not None:
                    self.dones[self.ptr] = done
                if culture is not None:
                    self.culture_ctx[self.ptr] = culture
                if epi is not None:
                    self.epigenetic_ctx[self.ptr] = epi
                if beh is not None:
                    self.behavior_ctx[self.ptr] = beh
                self.log_probs[self.ptr] = 0.0
                self.values[self.ptr] = 0.0

            self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample_batch(self, batch_size: int) -> Dict[str, np.ndarray]:
        with self.lock:
            if self.size < batch_size:
                return None

            if self.size < self.capacity:
                idx = slice(self.size - batch_size, self.size)
                return {
                    "obs": self.observations[idx],
                    "obs_full": self.observations_full[idx],
                    "act": self.actions[idx],
                    "action_mean": self.action_means[idx],
                    "rew": self.rewards[idx],
                    "log_prob": self.log_probs[idx],
                    "value": self.values[idx],
                    "don": self.dones[idx],
                    "culture_ctx": self.culture_ctx[idx],
                    "epigenetic_ctx": self.epigenetic_ctx[idx],
                    "behavior_ctx": self.behavior_ctx[idx],
                }

            recent_idx = (np.arange(self.ptr - batch_size, self.ptr) % self.capacity).astype(np.intp, copy=False)
            return {
                "obs": self.observations[recent_idx],
                "obs_full": self.observations_full[recent_idx],
                "act": self.actions[recent_idx],
                "action_mean": self.action_means[recent_idx],
                "rew": self.rewards[recent_idx],
                "log_prob": self.log_probs[recent_idx],
                "value": self.values[recent_idx],
                "don": self.dones[recent_idx],
                "culture_ctx": self.culture_ctx[recent_idx],
                "epigenetic_ctx": self.epigenetic_ctx[recent_idx],
                "behavior_ctx": self.behavior_ctx[recent_idx],
            }

    def _get_ordered_data(self) -> Dict[str, np.ndarray]:
        if self.size < self.capacity:
            idx = slice(0, self.ptr)
            return {
                "obs": self.observations[idx],
                "obs_full": self.observations_full[idx],
                "act": self.actions[idx],
                "action_mean": self.action_means[idx],
                "rew": self.rewards[idx],
                "log_prob": self.log_probs[idx],
                "value": self.values[idx],
                "don": self.dones[idx],
                "culture_ctx": self.culture_ctx[idx],
                "epigenetic_ctx": self.epigenetic_ctx[idx],
                "behavior_ctx": self.behavior_ctx[idx],
            }

        return {
            "obs": np.concatenate([self.observations[self.ptr:], self.observations[:self.ptr]], axis=0),
            "obs_full": np.concatenate([self.observations_full[self.ptr:], self.observations_full[:self.ptr]], axis=0),
            "act": np.concatenate([self.actions[self.ptr:], self.actions[:self.ptr]], axis=0),
            "action_mean": np.concatenate([self.action_means[self.ptr:], self.action_means[:self.ptr]], axis=0),
            "rew": np.concatenate([self.rewards[self.ptr:], self.rewards[:self.ptr]], axis=0),
            "log_prob": np.concatenate([self.log_probs[self.ptr:], self.log_probs[:self.ptr]], axis=0),
            "value": np.concatenate([self.values[self.ptr:], self.values[:self.ptr]], axis=0),
            "don": np.concatenate([self.dones[self.ptr:], self.dones[:self.ptr]], axis=0),
            "culture_ctx": np.concatenate([self.culture_ctx[self.ptr:], self.culture_ctx[:self.ptr]], axis=0),
            "epigenetic_ctx": np.concatenate([self.epigenetic_ctx[self.ptr:], self.epigenetic_ctx[:self.ptr]], axis=0),
            "behavior_ctx": np.concatenate([self.behavior_ctx[self.ptr:], self.behavior_ctx[:self.ptr]], axis=0),
        }

    def sample_all(self) -> Dict[str, np.ndarray]:
        with self.lock:
            if self.size == 0:
                return None

            data = self._get_ordered_data()
            self.ptr = 0
            self.size = 0
            return data
