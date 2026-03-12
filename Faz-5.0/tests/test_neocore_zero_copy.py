from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neo_core import (
    NeoCoreAsyncBridge,
    NeoCorePolicy,
    NeoCoreWorld,
    advantage_alignment,
    gossip_trust_update,
    interaction_trust_update,
)


def test_trust_matrix_updates_from_social_events():
    world = NeoCoreWorld(prefer_cuda=False)
    source = 0
    target = 1

    world.pos[source] = torch.tensor([100.0, 100.0], device=world.device)
    world.pos[target] = torch.tensor([110.0, 100.0], device=world.device)
    world.energy[source] = 90.0
    world.energy[target] = 20.0
    world.actions.zero_()

    baseline = float(world.trust_matrix[target, source].item())
    world.actions[source, 5] = 1.0
    world.step()
    after_altruism = float(world.trust_matrix[target, source].item())

    world.actions.zero_()
    world.actions[source, 4] = 1.0
    world.step()
    after_mimic = float(world.trust_matrix[target, source].item())

    assert after_altruism > baseline
    assert after_mimic < after_altruism


def test_observations_expose_trust_channels():
    world = NeoCoreWorld(prefer_cuda=False)
    source = 0
    target = 1

    world.trust_matrix[target, source] = 1.5
    world.trust_matrix[source, target] = -1.0
    world.nearest_idx[target] = source
    world.nearest_dist[target] = 10.0
    world.interaction_delta[target] = 0.25
    world._rebuild_observations()

    obs = world.get_observation_tensor()
    assert obs[target, 12].item() > 0.5
    assert obs[target, 13].item() < 0.0
    assert obs[target, 15].item() > 0.0


def test_syntax_bits_flow_into_observation():
    world = NeoCoreWorld(prefer_cuda=False)
    source = 0
    target = 1

    world.pos[source] = torch.tensor([100.0, 100.0], device=world.device)
    world.pos[target] = torch.tensor([110.0, 100.0], device=world.device)
    world.actions.zero_()
    world.actions[source, 3] = 1.0
    world.actions[source, 6] = 1.0
    world.actions[source, 8] = 1.0
    world.step()

    obs = world.get_observation_tensor()
    assert obs[target, 16].item() == 1.0
    assert obs[target, 17].item() == 0.0
    assert obs[target, 18].item() == 1.0
    assert obs[target, 19].item() == 0.0
    assert int(world.syntax_code[target].item()) == 5


def test_async_bridge_reports_cpu_policy_and_extended_action_space():
    bridge = NeoCoreAsyncBridge()
    _, metrics = bridge.collect_step()
    report = bridge.bridge_report()

    assert report["bridge_mode"] == "async_cpu_bridge"
    assert report["policy_device"] == "cpu"
    assert report["action_dim"] == 10
    assert report["architecture"] == "ae_comm + trust_biased_gat"
    assert "syntax_active_codes" in metrics
    assert "ae_loss" in metrics
    assert "social_influence_reward" in metrics


def test_policy_exposes_message_head_decoder_and_attention():
    world = NeoCoreWorld(prefer_cuda=False)
    policy = NeoCorePolicy()
    obs = world.get_observation_tensor().to("cpu")
    social_context = world.get_social_context(device=torch.device("cpu"))

    sample = policy.sample_actions(obs, social_context)
    aux = policy.auxiliary_losses(obs, social_context)

    assert sample.sampled_action.shape[-1] == 10
    assert sample.sampled_message_bits.shape[-1] == 4
    assert sample.message_symbol.shape[-1] == 16
    assert sample.attention_weights.shape[-1] == world.config.fov_neighbors
    assert aux["ae_loss"].item() >= 0.0


def test_alignment_and_gossip_math_are_bounded():
    ego = torch.tensor([0.4, -0.2])
    neighbor = torch.tensor([0.5, 0.8])
    memory = torch.tensor([0.7, -0.6])
    aligned = advantage_alignment(ego, neighbor, memory, gamma=0.985, alignment_lambda=0.92)
    updated = interaction_trust_update(torch.tensor([0.2, -0.4]), aligned, eta=0.25)
    gossiped = gossip_trust_update(
        trust_ik=torch.tensor([0.1, -0.2]),
        trust_ij=torch.tensor([0.8, 0.4]),
        trust_jk=torch.tensor([0.6, -0.5]),
        beta=0.2,
    )

    assert aligned.shape == ego.shape
    assert torch.all(updated <= 1.0)
    assert torch.all(updated >= -1.0)
    assert gossiped.shape == ego.shape
