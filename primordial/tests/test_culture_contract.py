import numpy as np

from primordial.core import config as cfg
from primordial.core.world import PrimordialWorld


def test_dialect_divergence_is_reported_for_isolated_agents():
    world = PrimordialWorld(headless=True)
    world.reset(seed=42)

    alive = np.zeros(cfg.MAX_AGENTS, dtype=np.int32)
    alive[:2] = 1
    world.organisms.alive.from_numpy(alive)

    dialect = np.zeros((cfg.MAX_AGENTS, cfg.CULTURE_CHANNELS), dtype=np.float32)
    dialect[0] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    dialect[1] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    world.organisms.dialect_state.from_numpy(dialect)
    world.organisms.local_marker_ctx.from_numpy(np.zeros_like(dialect))

    metrics = world.get_metrics()
    assert metrics["dialect_divergence"] > 0.2
    assert metrics["dialect_entropy"] > 0.5


def test_spawned_child_inherits_epigenetic_context():
    world = PrimordialWorld(headless=True)
    world.reset(seed=42)

    parent_idx = 0
    child_idx = cfg.INITIAL_PREY_COUNT + cfg.INITIAL_PRED_COUNT

    parent_epi = np.zeros((cfg.MAX_AGENTS, cfg.EPIGENETIC_DIM), dtype=np.float32)
    parent_epi[parent_idx] = np.linspace(0.2, 0.9, cfg.EPIGENETIC_DIM, dtype=np.float32)
    world.organisms.epigenetic_ctx.from_numpy(parent_epi)

    parents = np.zeros(cfg.MAX_AGENTS, dtype=np.int32)
    dead = np.zeros(cfg.MAX_AGENTS, dtype=np.int32)
    parents[0] = parent_idx
    dead[0] = child_idx
    world.organisms.parents_buf.from_numpy(parents)
    world.organisms.dead_buf.from_numpy(dead)
    world.organisms._spawn_kernel(world.organisms.parents_buf, world.organisms.dead_buf, 1)

    inherited = world.organisms.epigenetic_ctx.to_numpy()[child_idx]
    expected = parent_epi[parent_idx] * cfg.EPIGENETIC_INHERITANCE_STRENGTH

    assert int(world.organisms.alive.to_numpy()[child_idx]) == 1
    assert np.mean(np.abs(inherited - expected)) < 0.12
