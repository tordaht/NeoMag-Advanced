import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from primordial.adapters.pettingzoo_env import PrimordialPettingZooEnv
from primordial.core.behavior_event_logger import BehaviorEventLogger
from primordial.core import config as cfg
from primordial.core.ring_buffer import ReplayRingBuffer
from primordial.training.async_worker import prepare_training_batch, run_training_step
from primordial.training.policy import PrimordialPolicy


DEFAULT_SEEDS = (42, 43, 44)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def collect_rollout(env, model, replay, device):
    obs_tensor = env.world.get_observations_torch(device=device)
    sample = model.sample_actions(obs_tensor)
    env.world.step(sample.sampled_action)
    replay.add(
        env.world.get_ring_buffer_data(
            {
                "action_mean": sample.action_mean.cpu().numpy().astype(np.float32, copy=False),
                "log_prob": sample.log_prob.cpu().numpy().astype(np.float32, copy=False),
                "value": sample.value.cpu().numpy().astype(np.float32, copy=False),
            }
        )
    )


@torch.no_grad()
def evaluate_policy(model, seed: int, steps: int, event_log_path: Path | None = None):
    set_seed(seed)
    env = PrimordialPettingZooEnv(headless=True)
    env.world.capture_behavior_events = True
    env.reset(seed=seed)
    device = env.device
    event_logger = BehaviorEventLogger(filename=str(event_log_path), session_id=f"seed_{seed}") if event_log_path else None

    total_reward = 0.0
    total_mimic = 0
    total_mimic_success = 0
    total_altruism = 0
    total_transfer = 0.0
    total_cooldown_blocks = 0
    total_drag = 0.0
    total_anomaly = 0.0
    total_territorial_pressure = 0.0
    total_territorial_energy_loss = 0.0
    pop_trace = []
    reward_trace = []
    indigo_trace = []
    checkpoints = []
    checkpoint_interval = max(1, steps // 5)

    for step_idx in range(steps):
        obs_tensor = env.world.get_observations_torch(device=device)
        sample = model.sample_actions(obs_tensor)
        env.world.step(sample.sampled_action)
        if event_logger is not None:
            event_logger.log_events(env.world.get_behavior_events())
        metrics = env.world.get_metrics()
        step_reward = float(env.world.organisms.rewards.to_numpy().mean())
        total_reward += step_reward
        total_mimic += int(metrics.get("mimic_attempts", 0))
        total_mimic_success += int(metrics.get("mimic_success", 0))
        total_altruism += int(metrics.get("altruism_events", 0))
        total_transfer += float(metrics.get("altruism_transfer_amount", 0.0))
        total_cooldown_blocks += int(metrics.get("mimic_cooldown_blocks", 0))
        total_drag += float(metrics.get("avg_culture_drag", 0.0))
        total_anomaly += float(metrics.get("avg_signal_anomaly", 0.0))
        total_territorial_pressure += float(metrics.get("avg_territorial_pressure", 0.0))
        total_territorial_energy_loss += float(metrics.get("territorial_pressure_energy_loss", 0.0))
        pop_trace.append(int(metrics["alive_count"]))
        indigo_trace.append(int(metrics.get("indigo_count", 0)))
        reward_trace.append(step_reward)

        if (step_idx + 1) % checkpoint_interval == 0 or step_idx + 1 == steps:
            checkpoints.append(
                {
                    "step": step_idx + 1,
                    "alive_count": int(metrics["alive_count"]),
                    "prey_count": int(metrics["prey_count"]),
                    "pred_count": int(metrics["pred_count"]),
                    "avg_reward_window": float(np.mean(reward_trace[-checkpoint_interval:])),
                    "mimic_success_total": total_mimic_success,
                    "altruism_events_total": total_altruism,
                    "signal_activity": float(metrics.get("signal_activity", 0.0)),
                    "mimic_cooldown_blocks_total": total_cooldown_blocks,
                    "avg_energy": float(metrics.get("avg_energy", 0.0)),
                    "avg_culture_drag": total_drag / max(1, step_idx + 1),
                    "avg_signal_anomaly": total_anomaly / max(1, step_idx + 1),
                    "avg_territorial_pressure": total_territorial_pressure / max(1, step_idx + 1),
                    "territorial_pressure_energy_loss_total": total_territorial_energy_loss,
                    "indigo_count": int(metrics.get("indigo_count", 0)),
                }
            )

    final_alive = pop_trace[-1] if pop_trace else 0
    final_indigo = indigo_trace[-1] if indigo_trace else 0
    return {
        "seed": seed,
        "eval_steps": steps,
        "avg_reward": total_reward / max(1, steps),
        "mimic_attempts": total_mimic,
        "mimic_success": total_mimic_success,
        "mimic_success_rate": total_mimic_success / max(1, total_mimic),
        "altruism_events": total_altruism,
        "altruism_transfer_amount": total_transfer,
        "altruism_transfer_rate": total_transfer / max(1, total_altruism),
        "mimic_cooldown_blocks": total_cooldown_blocks,
        "avg_culture_drag": total_drag / max(1, steps),
        "avg_signal_anomaly": total_anomaly / max(1, steps),
        "avg_territorial_pressure": total_territorial_pressure / max(1, steps),
        "territorial_pressure_energy_loss": total_territorial_energy_loss,
        "alive_mean": float(np.mean(pop_trace)) if pop_trace else 0.0,
        "alive_std": float(np.std(pop_trace)) if pop_trace else 0.0,
        "alive_min": int(np.min(pop_trace)) if pop_trace else 0,
        "alive_max": int(np.max(pop_trace)) if pop_trace else 0,
        "alive_final": final_alive,
        "indigo_mean": float(np.mean(indigo_trace)) if indigo_trace else 0.0,
        "indigo_final": final_indigo,
        "indigo_final_share": float(final_indigo / max(1, final_alive)),
        "reward_min": float(np.min(reward_trace)) if reward_trace else 0.0,
        "reward_max": float(np.max(reward_trace)) if reward_trace else 0.0,
        "checkpoints": checkpoints,
    }


def train_seed(seed: int, train_steps: int):
    set_seed(seed)
    env = PrimordialPettingZooEnv(headless=True)
    env.reset(seed=seed)
    device = env.device

    model = PrimordialPolicy().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    replay = ReplayRingBuffer(
        capacity=512,
        num_agents=cfg.MAX_AGENTS,
        obs_dim=cfg.BASE_OBSERVATION_DIM,
        act_dim=cfg.ACTION_DIM,
    )

    last_stats = {}
    optimizer_steps = 0
    for _ in range(train_steps):
        collect_rollout(env, model, replay, device)
        batch_np = replay.sample_batch(batch_size=16)
        batch = prepare_training_batch(batch_np, device) if batch_np else None
        if batch is not None:
            last_stats = run_training_step(model, optimizer, batch)
            optimizer_steps += 1
            last_stats["optimizer_step"] = float(optimizer_steps)

    return model, last_stats


def summarize_variance(results):
    if not results:
        return {}

    variance = {}
    for key in (
        "avg_reward",
        "mimic_success",
        "mimic_success_rate",
        "altruism_events",
        "altruism_transfer_amount",
        "alive_mean",
        "alive_final",
        "avg_culture_drag",
        "avg_territorial_pressure",
        "territorial_pressure_energy_loss",
        "indigo_final_share",
    ):
        values = [float(item[key]) for item in results]
        variance[f"{key}_mean"] = float(np.mean(values))
        variance[f"{key}_std"] = float(np.std(values))
    return variance


def summarize_indigo_repeatability(results):
    if not results:
        return {}

    shares = [float(item.get("indigo_final_share", 0.0)) for item in results]
    pressure = [float(item.get("avg_territorial_pressure", 0.0)) for item in results]
    mimic_success = [float(item.get("mimic_success_rate", 0.0)) for item in results]
    survived = [int(item.get("indigo_final", 0)) > 0 for item in results]

    return {
        "survived_all_seeds": bool(all(survived)),
        "indigo_final_share_mean": float(np.mean(shares)),
        "indigo_final_share_std": float(np.std(shares)),
        "territorial_pressure_mean": float(np.mean(pressure)),
        "mimic_success_rate_mean": float(np.mean(mimic_success)),
        "repeatable_strategy": bool(all(survived) and np.mean(shares) > 0.05),
    }


def run_validation(train_steps: int, eval_steps: int, output_path: Path):
    baseline_model = PrimordialPolicy()
    report = {
        "config": {
            "train_steps": train_steps,
            "eval_steps": eval_steps,
            "device": "cpu bridge / taichi cuda fields",
            "seeds": list(DEFAULT_SEEDS),
        },
        "architecture": {
            "world_runtime": "Taichi fields and ecology run on CUDA when available, otherwise CPU.",
            "cpu_bridge": "Observations, actions, rewards, dones, log_prob, and value cross into NumPy before PPO updates.",
            "inference_loop": "Synchronous with each simulation step.",
            "training_loop": "Async PPO-lite worker/trainer consumes replay windows and performs optimizer steps on CPU tensors.",
            "zero_copy": "Not end-to-end zero-copy. Current learning path is explicit bridge, not pure GPU training.",
        },
        "baseline": [],
        "trained": [],
    }
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    baseline_results = []
    report["baseline"] = []
    for seed in DEFAULT_SEEDS:
        print(f"[BASELINE] seed={seed} evaluation starting...", flush=True)
        baseline_result = evaluate_policy(
            baseline_model,
            seed,
            eval_steps,
            event_log_path=output_path.with_name(f"behavior_events_baseline_seed_{seed}.csv"),
        )
        baseline_results.append(baseline_result)
        report["baseline"] = baseline_results
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[BASELINE] seed={seed} done | reward={baseline_result['avg_reward']:.4f} | mimic_success={baseline_result['mimic_success']}", flush=True)

    for seed in DEFAULT_SEEDS:
        print(f"[TRAIN] seed={seed} training starting...", flush=True)
        model, stats = train_seed(seed, train_steps)
        print(f"[TRAIN] seed={seed} training done | loss={stats.get('loss', 0.0):.4f} | entropy={stats.get('entropy', 0.0):.4f}", flush=True)
        print(f"[EVAL] seed={seed} trained evaluation starting...", flush=True)
        result = evaluate_policy(
            model,
            seed,
            eval_steps,
            event_log_path=output_path.with_name(f"behavior_events_trained_seed_{seed}.csv"),
        )
        result["training"] = stats
        report["trained"].append(result)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(
            f"[EVAL] seed={seed} done | reward={result['avg_reward']:.4f} | "
            f"mimic_success={result['mimic_success']} | altruism={result['altruism_events']}",
            flush=True,
        )

    report["variance"] = {
        "baseline": summarize_variance(report["baseline"]),
        "trained": summarize_variance(report["trained"]),
    }
    report["indigo_repeatability"] = {
        "baseline": summarize_indigo_repeatability(report["baseline"]),
        "trained": summarize_indigo_repeatability(report["trained"]),
    }

    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description="Three-seed honest learning validation harness.")
    parser.add_argument("--train-steps", type=int, default=240)
    parser.add_argument("--eval-steps", type=int, default=5000)
    parser.add_argument("--output", type=Path, default=Path("runs") / "honest_learning_eval.json")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = run_validation(args.train_steps, args.eval_steps, args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
