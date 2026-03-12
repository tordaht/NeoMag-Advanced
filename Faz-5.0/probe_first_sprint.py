from __future__ import annotations

import json
from pathlib import Path

import torch

from neo_core import NeoCoreAsyncBridge, NeoCorePolicy, NeoCoreWorld


def run_probe(steps: int = 64) -> dict:
    world = NeoCoreWorld(prefer_cuda=True)
    policy = NeoCorePolicy().to(world.device)
    bridge = NeoCoreAsyncBridge()
    bridge.collect_step()

    trust_trace = []
    reward_trace = []
    for _ in range(steps):
        social_context = world.get_social_context(device=world.device)
        sample = policy.sample_actions(world.get_observation_tensor(), social_context)
        world.step(sample.sampled_action)
        metrics = world.metrics()
        trust_trace.append(metrics["trust_mean"])
        reward_trace.append(float(world.rewards.mean().item()))

    target = 1
    source = 0
    world.actions.zero_()
    world.pos[source] = torch.tensor([100.0, 100.0], device=world.device)
    world.pos[target] = torch.tensor([110.0, 100.0], device=world.device)
    world.energy[source] = 90.0
    world.energy[target] = 30.0
    world.actions[source, 5] = 1.0
    world.actions[source, 3] = 1.0
    world.actions[source, 6] = 1.0
    world.actions[source, 8] = 1.0
    before_trust = float(world.trust_matrix[target, source].item())
    world.step()
    after_altruism = float(world.trust_matrix[target, source].item())

    world.actions.zero_()
    world.actions[source, 4] = 1.0
    world.actions[source, 3] = 1.0
    world.actions[source, 6] = 1.0
    world.actions[source, 8] = 1.0
    world.step()
    after_mimic = float(world.trust_matrix[target, source].item())

    obs = world.get_observation_tensor()
    social_context = world.get_social_context(device=world.device)
    aux = policy.auxiliary_losses(obs, social_context)
    sample = policy.sample_actions(obs, social_context)
    report = {
        "runtime": world.runtime_report(),
        "bridge": bridge.bridge_report(),
        "shapes": {
            "observations": list(obs.shape),
            "actions": list(world.actions.shape),
            "trust_matrix": list(world.trust_matrix.shape),
            "neighbor_state": list(social_context["neighbor_state"].shape),
            "neighbor_messages": list(social_context["neighbor_messages"].shape),
        },
        "zero_copy_alias": {
            "observations_ptr": int(obs.data_ptr()),
            "actions_ptr": int(world.actions.data_ptr()),
            "trust_ptr": int(world.trust_matrix.data_ptr()),
        },
        "trust_probe": {
            "before": before_trust,
            "after_altruism": after_altruism,
            "after_mimic": after_mimic,
            "observation_trust_out": float(obs[target, 12].item()),
            "observation_trust_in": float(obs[target, 13].item()),
            "received_syntax_bits": [float(obs[target, 16 + idx].item()) for idx in range(4)],
            "received_syntax_code": int(world.syntax_code[target].item()),
        },
        "architecture_probe": {
            "ae_loss": float(aux["ae_loss"].item()),
            "social_influence_reward": float(aux["social_influence_reward"].item()),
            "positive_listening_loss": float(aux["positive_listening_loss"].item()),
            "attention_shape": list(sample.attention_weights.shape),
            "message_symbol_shape": list(sample.message_symbol.shape),
        },
        "trace": {
            "trust_mean_tail": trust_trace[-8:],
            "reward_mean_tail": reward_trace[-8:],
        },
        "metrics": world.metrics(),
    }
    return report


def main():
    output = Path(__file__).resolve().parent / "runs" / "neo_core_first_sprint.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    report = run_probe()
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSaved report to {output}")


if __name__ == "__main__":
    main()
