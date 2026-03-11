import numpy as np
import torch
from gymnasium import spaces
from pettingzoo import ParallelEnv

from primordial.core import config as cfg
from primordial.core.world import PrimordialWorld


class PrimordialPettingZooEnv(ParallelEnv):
    """
    CPU-first PettingZoo adapter aligned with the world data surface.
    """

    metadata = {"render_modes": ["headless"], "name": "primordial_v17"}

    def __init__(self, headless=True):
        self.world = PrimordialWorld(headless=headless)
        self.possible_agents = [f"agent_{i}" for i in range(cfg.MAX_AGENTS)]
        self.agents = self.possible_agents[:]
        self.device = torch.device("cpu")
        print(f"[API HONESTY] PettingZoo Adapter running on {self.device}")

        # Expanded Action Space: [Thrust, Steer, Metabolic, Signal, Mimicry, Altruism]
        action_low = np.array(cfg.ACTION_LOW, dtype=np.float32)
        action_high = np.array(cfg.ACTION_HIGH, dtype=np.float32)
        self.observation_spaces = {
            agent: spaces.Box(low=-np.inf, high=np.inf, shape=(cfg.OBSERVATION_DIM,), dtype=np.float32)
            for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: spaces.Box(low=action_low, high=action_high, dtype=np.float32)
            for agent in self.possible_agents
        }

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def _get_obs_torch(self):
        return self.world.get_observations_torch(device=self.device)

    def _get_rewards_torch(self):
        return self.world.get_rewards_torch(device=self.device)

    def reset(self, seed=None, options=None):
        self.world.reset(seed=seed)
        obs_torch = self._get_obs_torch()
        alive_mask = self.world.get_alive_mask_torch(device=self.device)

        observations = {}
        for i in range(cfg.MAX_AGENTS):
            if int(alive_mask[i].item()) == 1:
                observations[self.possible_agents[i]] = obs_torch[i]

        self.agents = list(observations.keys())
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions):
        if isinstance(actions, dict):
            actions_batched = torch.zeros((cfg.MAX_AGENTS, cfg.ACTION_DIM), device=self.device)
            for i, agent in enumerate(self.possible_agents):
                if agent in actions:
                    actions_batched[i] = torch.as_tensor(actions[agent], device=self.device, dtype=torch.float32)
            self.world.step(actions_batched)
        else:
            self.world.step(actions)

        obs_torch = self._get_obs_torch()
        reward_torch = self._get_rewards_torch()
        alive_mask = self.world.get_alive_mask_torch(device=self.device)

        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}
        current_alive_agents = []

        for i in range(cfg.MAX_AGENTS):
            agent = self.possible_agents[i]
            is_alive = int(alive_mask[i].item()) == 1
            observations[agent] = obs_torch[i]
            rewards[agent] = float(reward_torch[i].item())
            terminations[agent] = not is_alive
            truncations[agent] = False
            infos[agent] = {}
            if is_alive:
                current_alive_agents.append(agent)

        self.agents = current_alive_agents
        return observations, rewards, terminations, truncations, infos
