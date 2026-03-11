import numpy as np
import torch

from primordial.adapters.pettingzoo_env import PrimordialPettingZooEnv
from primordial.core import config as cfg


def test_pettingzoo_action_space():
    env = PrimordialPettingZooEnv()
    agent_id = env.possible_agents[0]
    action_space = env.action_space(agent_id)
    assert action_space.shape == (cfg.ACTION_DIM,)

    # Thrust, Steer, Metabolic, Signal, Mimicry, Altruism
    expected_low = np.array(cfg.ACTION_LOW, dtype=np.float32)
    expected_high = np.array(cfg.ACTION_HIGH, dtype=np.float32)

    assert np.allclose(action_space.low, expected_low)
    assert np.allclose(action_space.high, expected_high)


def test_pettingzoo_reset():
    env = PrimordialPettingZooEnv()
    observations, infos = env.reset()

    assert len(observations) > 0

    first_agent = list(observations.keys())[0]
    obs_val = observations[first_agent]

    if torch.is_tensor(obs_val):
        obs_val = obs_val.cpu().numpy()

    assert obs_val.shape == (cfg.OBSERVATION_DIM,)
    assert np.all(np.isfinite(obs_val))


def test_pettingzoo_step():
    env = PrimordialPettingZooEnv()
    env.reset()

    actions = {
        agent: env.action_space(agent).sample()
        for agent in env.agents
    }

    new_obs, rewards, terminations, truncations, infos = env.step(actions)

    assert len(new_obs) == len(rewards) == len(terminations) == len(truncations)
    for agent in rewards:
        assert np.isfinite(rewards[agent])


def test_thermodynamic_loop_carcass():
    env = PrimordialPettingZooEnv()
    env.reset()
    carcass_field = env.world.fields.carcasses.to_numpy()
    assert carcass_field.shape == cfg.FIELD_RES


def test_world_metrics_expose_extended_culture_surface():
    env = PrimordialPettingZooEnv()
    env.reset()

    metrics = env.world.get_metrics()

    for key in (
        "avg_energy",
        "signal_activity",
        "signal_density",
        "regional_marker_concentration",
        "dialect_entropy",
        "dialect_divergence",
        "mimic_attempts",
        "mimic_success",
        "mimic_success_rate",
        "altruism_events",
        "altruism_transfer_amount",
        "altruism_recipient_count",
    ):
        assert key in metrics


def test_world_ring_buffer_payload_contains_contexts():
    env = PrimordialPettingZooEnv()
    env.reset()
    env.world.step(np.zeros((cfg.MAX_AGENTS, cfg.ACTION_DIM), dtype=np.float32))

    payload = env.world.get_ring_buffer_data()

    assert set(payload.keys()) == {
        "obs",
        "obs_full",
        "act",
        "action_mean",
        "rew",
        "log_prob",
        "value",
        "don",
        "culture_ctx",
        "epigenetic_ctx",
        "behavior_ctx",
    }
    assert payload["obs"].shape == (cfg.MAX_AGENTS, cfg.BASE_OBSERVATION_DIM)
    assert payload["obs_full"].shape == (cfg.MAX_AGENTS, cfg.OBSERVATION_DIM)
    assert payload["culture_ctx"].shape == (cfg.MAX_AGENTS, cfg.CULTURE_CONTEXT_DIM)
    assert payload["epigenetic_ctx"].shape == (cfg.MAX_AGENTS, cfg.EPIGENETIC_DIM)
    assert payload["behavior_ctx"].shape == (cfg.MAX_AGENTS, cfg.BEHAVIOR_DIM)


def test_field_update_commits_diffused_buffers():
    env = PrimordialPettingZooEnv()
    env.reset()

    before_nutrients = env.world.fields.nutrients.to_numpy().copy()
    before_culture = env.world.fields.culture.to_numpy().copy()

    env.world.step(np.zeros((cfg.MAX_AGENTS, cfg.ACTION_DIM), dtype=np.float32))

    after_nutrients = env.world.fields.nutrients.to_numpy()
    after_culture = env.world.fields.culture.to_numpy()

    assert not np.allclose(before_nutrients, after_nutrients)
    assert after_culture.shape[-1] == cfg.CULTURE_CHANNELS
