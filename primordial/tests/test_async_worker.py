import time

import numpy as np
import torch

from primordial.core import config as cfg
from primordial.core.ring_buffer import ReplayRingBuffer
from primordial.training.async_worker import AsyncTrainingWorker
from primordial.training.policy import PrimordialPolicy


def _payload(idx: int):
    obs = np.full((8, cfg.BASE_OBSERVATION_DIM), idx + 1, dtype=np.float32)
    culture = np.full((8, cfg.CULTURE_CONTEXT_DIM), 0.2 + idx * 0.01, dtype=np.float32)
    epi = np.full((8, cfg.EPIGENETIC_DIM), 0.5, dtype=np.float32)
    beh = np.full((8, cfg.BEHAVIOR_DIM), 0.1, dtype=np.float32)
    obs_full = np.concatenate((obs, culture, epi, beh), axis=1).astype(np.float32)
    return {
        "obs": obs,
        "obs_full": obs_full,
        "act": np.full((8, cfg.ACTION_DIM), 0.25, dtype=np.float32),
        "action_mean": np.full((8, cfg.ACTION_DIM), 0.2, dtype=np.float32),
        "rew": np.full(8, 1.0 + idx * 0.1, dtype=np.float32),
        "log_prob": np.full(8, -0.8, dtype=np.float32),
        "value": np.full(8, 0.3, dtype=np.float32),
        "don": np.zeros(8, dtype=np.bool_),
        "culture_ctx": culture,
        "epigenetic_ctx": epi,
        "behavior_ctx": beh,
    }


def test_async_worker_runs_real_updates_and_syncs():
    model = PrimordialPolicy()
    buffer = ReplayRingBuffer(capacity=32, num_agents=8, obs_dim=cfg.BASE_OBSERVATION_DIM, act_dim=cfg.ACTION_DIM)

    for idx in range(20):
        buffer.add(_payload(idx))

    worker = AsyncTrainingWorker(model, buffer, torch.device("cpu"), batch_size=4, save_interval_steps=1000)
    worker.start()

    deadline = time.time() + 5.0
    while worker.train_step == 0 and time.time() < deadline:
        time.sleep(0.05)

    synced = worker.sync_to_main()
    worker.stop()

    assert worker.train_step > 0
    assert worker.last_loss > 0.0
    assert "policy_loss" in worker.last_stats
    assert "entropy" in worker.last_stats
    assert synced is True
