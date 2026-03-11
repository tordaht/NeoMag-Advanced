import numpy as np

from primordial.core import config as cfg
from primordial.core.ring_buffer import ReplayRingBuffer


def make_payload(step: int, num_agents: int, obs_dim: int, act_dim: int):
    return {
        "obs": np.full((num_agents, obs_dim), step, dtype=np.float32),
        "act": np.full((num_agents, act_dim), step, dtype=np.float32),
        "rew": np.full(num_agents, step, dtype=np.float32),
        "don": np.zeros(num_agents, dtype=np.bool_),
        "culture_ctx": np.full((num_agents, cfg.CULTURE_CONTEXT_DIM), step, dtype=np.float32),
        "epigenetic_ctx": np.full((num_agents, cfg.EPIGENETIC_DIM), step, dtype=np.float32),
        "behavior_ctx": np.full((num_agents, cfg.BEHAVIOR_DIM), step, dtype=np.float32),
    }


def test_ring_buffer_overwrite_behavior():
    capacity = 5
    num_agents = 2
    obs_dim = 3
    act_dim = 2

    buffer = ReplayRingBuffer(capacity, num_agents, obs_dim, act_dim)

    for i in range(capacity):
        buffer.add(make_payload(i, num_agents, obs_dim, act_dim))

    assert buffer.size == capacity
    assert buffer.ptr == 0

    payload = make_payload(99, num_agents, obs_dim, act_dim)
    buffer.add(payload)

    assert buffer.size == capacity
    assert buffer.ptr == 1

    data = buffer.sample_all()
    assert data is not None
    assert len(data["obs"]) == capacity
    assert np.all(data["obs"][-1] == 99)
    assert np.all(data["obs"][0] == 1)
    assert np.all(data["culture_ctx"][-1] == 99)
    assert np.all(data["epigenetic_ctx"][-1] == 99)
    assert np.all(data["behavior_ctx"][-1] == 99)
    assert buffer.size == 0
    assert buffer.ptr == 0


def test_ring_buffer_thread_safety():
    buffer = ReplayRingBuffer(100, 2, 3, 2)
    assert buffer.lock is not None
    buffer.add(make_payload(0, 2, 3, 2))
    assert buffer.size == 1
    assert not buffer.lock.locked()


def test_ring_buffer_sample_batch_keeps_latest_chronological_window():
    buffer = ReplayRingBuffer(6, 1, 2, 1)

    for i in range(8):
        buffer.add(make_payload(i, 1, 2, 1))

    batch = buffer.sample_batch(3)
    assert batch is not None
    assert batch["obs"].shape[0] == 3
    assert np.all(batch["obs"][0] == 5)
    assert np.all(batch["obs"][1] == 6)
    assert np.all(batch["obs"][2] == 7)
    assert np.all(batch["culture_ctx"][2] == 7)
