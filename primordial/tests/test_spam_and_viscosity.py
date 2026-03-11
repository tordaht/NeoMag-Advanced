import numpy as np

from primordial.core import config as cfg
from primordial.core.world import PrimordialWorld


def _single_agent_world(agent_type=cfg.TYPE_PRED):
    world = PrimordialWorld(headless=True, capture_behavior_events=True)
    world.reset(seed=42)

    alive = np.zeros(cfg.MAX_AGENTS, dtype=np.int32)
    alive[0] = 1
    world.organisms.alive.from_numpy(alive)

    types = np.zeros(cfg.MAX_AGENTS, dtype=np.int32)
    types[0] = agent_type
    world.organisms.type.from_numpy(types)

    energy = np.zeros(cfg.MAX_AGENTS, dtype=np.float32)
    energy[0] = 100.0
    world.organisms.energy.from_numpy(energy)

    pos = world.organisms.pos.to_numpy()
    pos[:] = np.array([305.0, 305.0], dtype=np.float32)
    world.organisms.pos.from_numpy(pos)
    return world


def test_mimic_spam_gets_blocked_and_more_expensive():
    world = _single_agent_world(agent_type=cfg.TYPE_PRED)
    actions = np.zeros((cfg.MAX_AGENTS, cfg.ACTION_DIM), dtype=np.float32)
    actions[0, 4] = 1.0

    total_attempts = 0
    total_blocks = 0
    total_cost = 0.0
    total_penalty = 0.0
    for _ in range(20):
        world.step(actions)
        metrics = world.get_metrics()
        total_attempts += metrics["mimic_attempts"]
        total_blocks += metrics["mimic_cooldown_blocks"]
        total_cost += metrics["mimic_signal_cost_total"]
        total_penalty += metrics["mimic_spam_penalty_total"]

    assert total_attempts <= 2
    assert total_blocks >= 10
    assert total_cost > (cfg.MIMIC_BASE_COST + cfg.MIMIC_SIGNAL_COST) * 2.0
    assert total_penalty > 0.5
    assert any(event["event_type"] == "mimic" for event in world.get_behavior_events())


def test_alien_culture_drag_is_material():
    world = _single_agent_world(agent_type=cfg.TYPE_PREY)

    dialect = np.zeros((cfg.MAX_AGENTS, cfg.CULTURE_CHANNELS), dtype=np.float32)
    dialect[0] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    world.organisms.dialect_state.from_numpy(dialect)

    culture = np.zeros((cfg.FIELD_RES[0], cfg.FIELD_RES[1], cfg.CULTURE_CHANNELS), dtype=np.float32)
    culture[:, :, 1] = 60.0
    world.fields.culture.from_numpy(culture)

    actions = np.zeros((cfg.MAX_AGENTS, cfg.ACTION_DIM), dtype=np.float32)
    actions[0, 0] = 1.0
    world.step(actions)

    metrics = world.get_metrics()
    assert metrics["avg_alien_mismatch"] > 0.6
    assert metrics["avg_culture_drag"] > 0.9
    assert metrics["avg_territorial_pressure"] > 0.6
    assert metrics["territorial_pressure_energy_loss"] > 0.05


def test_predator_observation_contains_signal_anomaly_channel():
    world = _single_agent_world(agent_type=cfg.TYPE_PRED)

    dialect = np.zeros((cfg.MAX_AGENTS, cfg.CULTURE_CHANNELS), dtype=np.float32)
    dialect[0] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    world.organisms.dialect_state.from_numpy(dialect)

    field_x = int(305.0 * cfg.FIELD_RES[0] / cfg.WORLD_RES[0])
    field_y = int(305.0 * cfg.FIELD_RES[1] / cfg.WORLD_RES[1])

    culture = world.fields.culture.to_numpy()
    culture[field_x, field_y] = np.array([0.0, 40.0, 0.0], dtype=np.float32)
    world.fields.culture.from_numpy(culture)

    signal = world.fields.active_signal.to_numpy()
    signal[field_x, field_y] = 1.0
    world.fields.active_signal.from_numpy(signal)

    prey_quorum = world.fields.prey_quorum.to_numpy()
    prey_quorum[field_x, field_y] = 1.0
    world.fields.prey_quorum.from_numpy(prey_quorum)

    world.organisms.compute_observations(world.fields, world.organisms.observations)
    obs = world.organisms.observations.to_numpy()[0]

    assert obs[13] > 0.1
    assert 0.0 <= obs[14] <= 1.0
    assert obs[15] > 0.1
