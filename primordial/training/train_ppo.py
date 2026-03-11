import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from primordial.adapters.pettingzoo_env import PrimordialPettingZooEnv
from primordial.adapters.tensorboard_logger import PrimordialLogger
from primordial.core import config as cfg
from primordial.core.ring_buffer import ReplayRingBuffer
from primordial.training.async_worker import (
    DEFAULT_CHECKPOINT_PATH,
    build_linear_decay_scheduler,
    prepare_training_batch,
    run_training_step,
)
from primordial.training.policy import PrimordialPolicy


def _collect_step(env, model, device):
    obs_tensor = env.world.get_observations_torch(device=device)
    sample = model.sample_actions(obs_tensor)
    env.world.step(sample.sampled_action)
    rollout = {
        "action_mean": sample.action_mean.cpu().numpy().astype(np.float32, copy=False),
        "log_prob": sample.log_prob.cpu().numpy().astype(np.float32, copy=False),
        "value": sample.value.cpu().numpy().astype(np.float32, copy=False),
    }
    return env.world.get_ring_buffer_data(rollout), sample


def _moving_average(values, window=10):
    if not values:
        return 0.0
    tail = values[-min(window, len(values)) :]
    return float(np.mean(tail))


def _safe_corr(a, b):
    if len(a) < 3 or len(b) < 3:
        return 0.0
    if np.std(a) < 1e-6 or np.std(b) < 1e-6:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _lotka_volterra_report(trace, max_lag=120):
    prey = np.asarray([item["prey_count"] for item in trace], dtype=np.float64)
    pred = np.asarray([item["pred_count"] for item in trace], dtype=np.float64)
    if prey.size < 8 or pred.size < 8:
        return {"best_lag": 0, "best_corr": 0.0}

    best_lag = 0
    best_corr = -1.0
    max_lag = min(max_lag, len(trace) // 2)
    for lag in range(1, max_lag + 1):
        corr = _safe_corr(prey[:-lag], pred[lag:])
        if corr > best_corr:
            best_corr = corr
            best_lag = lag

    return {
        "best_lag": int(best_lag),
        "best_corr": float(best_corr),
        "predator_extinct": bool(np.max(pred) <= 0.0),
        "predator_min": int(np.min(pred)),
        "predator_max": int(np.max(pred)),
        "prey_min": int(np.min(prey)),
        "prey_max": int(np.max(prey)),
    }


def _training_evidence(trace, loss_trace):
    energy = np.asarray([item["avg_energy"] for item in trace], dtype=np.float64)
    loss_values = [item["loss"] for item in loss_trace if np.isfinite(item["loss"])]
    entropy_values = [item["entropy"] for item in loss_trace if np.isfinite(item["entropy"])]

    early_loss = float(np.mean(loss_values[: min(10, len(loss_values))])) if loss_values else 0.0
    late_loss = _moving_average(loss_values, window=10)
    early_entropy = float(np.mean(entropy_values[: min(10, len(entropy_values))])) if entropy_values else 0.0
    late_entropy = _moving_average(entropy_values, window=10)

    return {
        "loss_trend": {
            "early_mean": early_loss,
            "late_mean": late_loss,
            "delta": late_loss - early_loss,
            "optimizer_steps": len(loss_values),
        },
        "entropy_trend": {
            "early_mean": early_entropy,
            "late_mean": late_entropy,
            "delta": late_entropy - early_entropy,
        },
        "energy_profile": {
            "mean": float(np.mean(energy)) if energy.size else 0.0,
            "std": float(np.std(energy)) if energy.size else 0.0,
            "min": float(np.min(energy)) if energy.size else 0.0,
            "max": float(np.max(energy)) if energy.size else 0.0,
            "late_mean": float(np.mean(energy[-min(100, len(energy)) :])) if energy.size else 0.0,
        },
        "lotka_volterra": _lotka_volterra_report(trace),
    }


@torch.no_grad()
def evaluate_policy(model, seed: int, steps: int = 180):
    env = PrimordialPettingZooEnv(headless=True)
    env.reset(seed=seed)
    device = env.device
    torch.manual_seed(seed)

    total_reward = 0.0
    total_mimic = 0
    total_altruism = 0
    total_mimic_success = 0
    total_cooldown_blocks = 0
    total_drag = 0.0
    total_anomaly = 0.0
    action_sum = np.zeros(cfg.ACTION_DIM, dtype=np.float64)
    action_steps = 0

    for _ in range(steps):
        obs_tensor = env.world.get_observations_torch(device=device)
        sample = model.sample_actions(obs_tensor)
        env.world.step(sample.sampled_action)
        metrics = env.world.get_metrics()
        total_reward += float(env.world.organisms.rewards.to_numpy().mean())
        total_mimic += int(metrics.get("mimic_attempts", 0))
        total_altruism += int(metrics.get("altruism_events", 0))
        total_mimic_success += int(metrics.get("mimic_success", 0))
        total_cooldown_blocks += int(metrics.get("mimic_cooldown_blocks", 0))
        total_drag += float(metrics.get("avg_culture_drag", 0.0))
        total_anomaly += float(metrics.get("avg_signal_anomaly", 0.0))
        alive = env.world.organisms.alive.to_numpy() == 1
        if alive.any():
            action_sum += env.world.organisms.actions.to_numpy()[alive].mean(axis=0)
            action_steps += 1

    metrics = env.world.get_metrics()
    return {
        "seed": seed,
        "avg_reward": total_reward / max(1, steps),
        "mimic_attempts": total_mimic,
        "altruism_events": total_altruism,
        "mimic_success": total_mimic_success,
        "mimic_success_rate": total_mimic_success / max(1, total_mimic),
        "alive_count": metrics["alive_count"],
        "avg_energy": float(metrics.get("avg_energy", 0.0)),
        "avg_visibility": float(metrics.get("avg_visibility", 0.0)),
        "avg_culture_drag": total_drag / max(1, steps),
        "avg_signal_anomaly": total_anomaly / max(1, steps),
        "mimic_cooldown_blocks": total_cooldown_blocks,
        "altruism_thermo_gap": float(metrics.get("altruism_thermo_gap", 0.0)),
        "action_activation": (action_sum / max(1, action_steps)).tolist(),
    }


def train(
    total_steps: int = 1000,
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    save_report: bool = True,
    seed: int = 42,
    eval_seeds: tuple[int, ...] = (42, 43, 44),
    log_dir: str | None = None,
):
    print("=" * 50)
    print("PRIMORDIAL MARL FRAMEWORK v17.3 (EMERGENCE PPO-LITE)")
    print("API: constrained sensory world + honest rollout + ecology evidence")
    print("=" * 50)

    env = PrimordialPettingZooEnv(headless=True)
    logger = PrimordialLogger(log_dir=log_dir or f"runs/v17_honest_learning_seed_{seed}")

    device = env.device
    model = PrimordialPolicy().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.PPO_LR)
    scheduler = build_linear_decay_scheduler(optimizer, total_steps)
    replay = ReplayRingBuffer(
        capacity=512,
        num_agents=cfg.MAX_AGENTS,
        obs_dim=cfg.BASE_OBSERVATION_DIM,
        act_dim=cfg.ACTION_DIM,
    )

    env.reset(seed=seed)
    print(f"Hardware: {device} | Bridge Pipeline: Explicit CPU")

    start_time = time.time()
    training_info = {}
    optimizer_steps = 0
    trace = []
    loss_trace = []

    for step in range(total_steps):
        if not env.agents:
            env.reset(seed=seed + step)
            continue

        payload, _ = _collect_step(env, model, device)
        replay.add(payload)

        batch_np = replay.sample_batch(batch_size=16)
        batch = prepare_training_batch(batch_np, device) if batch_np else None
        if batch is not None:
            training_info = run_training_step(model, optimizer, batch, scheduler=scheduler)
            optimizer_steps += 1
            training_info["optimizer_step"] = float(optimizer_steps)
            loss_trace.append(
                {
                    "step": step,
                    "loss": float(training_info.get("loss", 0.0)),
                    "entropy": float(training_info.get("entropy", 0.0)),
                    "policy_loss": float(training_info.get("policy_loss", 0.0)),
                    "value_loss": float(training_info.get("value_loss", 0.0)),
                    "lr": float(training_info.get("lr", cfg.PPO_LR)),
                }
            )

        metrics = env.world.get_metrics()
        logger.log_step(metrics, training_info)
        trace.append(
            {
                "step": step,
                "prey_count": int(metrics["prey_count"]),
                "pred_count": int(metrics["pred_count"]),
                "alive_count": int(metrics["alive_count"]),
                "avg_energy": float(metrics.get("avg_energy", 0.0)),
                "avg_visibility": float(metrics.get("avg_visibility", 0.0)),
                "avg_culture_drag": float(metrics.get("avg_culture_drag", 0.0)),
                "altruism_thermo_gap": float(metrics.get("altruism_thermo_gap", 0.0)),
                "mimic_signal_cost_total": float(metrics.get("mimic_signal_cost_total", 0.0)),
                "lr": float(training_info.get("lr", cfg.PPO_LR)),
            }
        )

        if step % 100 == 0:
            fps = (step + 1) / max(1e-6, (time.time() - start_time))
            print(
                f"[Step {step:04d}] FPS: {fps:.1f} | Prey/Pred: {metrics['prey_count']}/{metrics['pred_count']} | "
                f"Energy: {metrics.get('avg_energy', 0.0):.2f} | Loss: {training_info.get('loss', 0.0):.3f} | "
                f"Visibility: {metrics.get('avg_visibility', 0.0):.3f} | ThermoGap: {metrics.get('altruism_thermo_gap', 0.0):.3f}"
            )

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)

    evidence = _training_evidence(trace, loss_trace)
    eval_report = {
        "seed": seed,
        "baseline": evaluate_policy(PrimordialPolicy().to(device), seed=max(1, seed - 1), steps=total_steps),
        "runs": [evaluate_policy(model, seed=eval_seed, steps=total_steps) for eval_seed in eval_seeds],
        "training": training_info,
        "evidence": evidence,
        "trace_tail": trace[-20:],
        "loss_tail": loss_trace[-20:],
    }

    if save_report:
        report_path = checkpoint_path.with_suffix(".eval.json")
        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(eval_report, handle, indent=2)
        print(f"Evaluation report saved to {report_path}")

    logger.close()
    print(
        "[Evidence] "
        f"loss delta={evidence['loss_trend']['delta']:.4f} | "
        f"late energy={evidence['energy_profile']['late_mean']:.2f} | "
        f"lag corr={evidence['lotka_volterra']['best_corr']:.3f} @ lag {evidence['lotka_volterra']['best_lag']}"
    )
    print(f"\nTraining run complete. Checkpoint saved to {checkpoint_path}.")
    return eval_report


def main():
    parser = argparse.ArgumentParser(description="Train the emergence-focused Primordial PPO-lite stack.")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(total_steps=args.steps, checkpoint_path=args.checkpoint, save_report=not args.no_report, seed=args.seed)


if __name__ == "__main__":
    main()
