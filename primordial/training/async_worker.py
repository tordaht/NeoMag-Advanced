import copy
import threading
import time
import traceback
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from primordial.core import config as cfg
from primordial.core.ring_buffer import ReplayRingBuffer
from primordial.training.policy import PrimordialPolicy


DEFAULT_CHECKPOINT_PATH = Path(__file__).resolve().parents[2] / "primordial_ppo_v17.pt"


def _compose_obs_full(batch: Dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    obs_full_np = batch.get("obs_full")
    if obs_full_np is not None and getattr(obs_full_np, "shape", ())[-1] == cfg.OBSERVATION_DIM:
        return torch.from_numpy(obs_full_np).to(device=device, dtype=torch.float32)

    obs = torch.from_numpy(batch["obs"]).to(device=device, dtype=torch.float32)
    culture_ctx = torch.from_numpy(batch["culture_ctx"]).to(device=device, dtype=torch.float32)
    epigenetic_ctx = torch.from_numpy(batch["epigenetic_ctx"]).to(device=device, dtype=torch.float32)
    behavior_ctx = torch.from_numpy(batch["behavior_ctx"]).to(device=device, dtype=torch.float32)
    return torch.cat((obs, culture_ctx, epigenetic_ctx, behavior_ctx), dim=-1)


def _scale_rewards(rew: torch.Tensor) -> torch.Tensor:
    scaled = rew * cfg.PPO_REWARD_SCALE
    return torch.clamp(scaled, -cfg.PPO_REWARD_CLIP, cfg.PPO_REWARD_CLIP)


def build_linear_decay_scheduler(optimizer: optim.Optimizer, total_steps: int):
    total_steps = max(1, int(total_steps))

    def lr_lambda(step: int) -> float:
        progress = min(1.0, step / total_steps)
        return max(cfg.PPO_LR_MIN_FACTOR, 1.0 - (1.0 - cfg.PPO_LR_MIN_FACTOR) * progress)

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def prepare_training_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Optional[Dict[str, torch.Tensor]]:
    if not batch or len(batch["obs"]) == 0:
        return None

    obs_full = _compose_obs_full(batch, device)
    act = torch.from_numpy(batch["act"]).to(device=device, dtype=torch.float32)
    raw_rew = torch.from_numpy(batch["rew"]).to(device=device, dtype=torch.float32)
    rew = _scale_rewards(raw_rew)
    don = torch.from_numpy(batch["don"]).to(device=device, dtype=torch.bool)
    old_log_prob_np = np.asarray(batch.get("log_prob", np.zeros_like(batch["rew"])), dtype=np.float32)
    value_np = np.asarray(batch.get("value", np.zeros_like(batch["rew"])), dtype=np.float32)
    old_log_prob = torch.from_numpy(old_log_prob_np).to(device=device, dtype=torch.float32)
    value = torch.from_numpy(value_np).to(device=device, dtype=torch.float32)

    valid_mask = torch.isfinite(obs_full).all(dim=-1)
    valid_mask &= torch.isfinite(act).all(dim=-1)
    valid_mask &= torch.isfinite(rew)
    valid_mask &= obs_full.abs().sum(dim=-1) > 0

    if int(valid_mask.sum().item()) < 32:
        return None

    returns = torch.zeros_like(rew)
    advantages = torch.zeros_like(rew)
    next_return = torch.zeros(rew.shape[1], device=device, dtype=torch.float32)
    next_value = torch.zeros(rew.shape[1], device=device, dtype=torch.float32)
    next_advantage = torch.zeros(rew.shape[1], device=device, dtype=torch.float32)

    gamma = cfg.PPO_GAMMA
    gae_lambda = cfg.PPO_GAE_LAMBDA

    for t in range(rew.shape[0] - 1, -1, -1):
        mask = (~don[t]).float()
        returns[t] = rew[t] + gamma * next_return * mask
        td_error = rew[t] + gamma * next_value * mask - value[t]
        advantages[t] = td_error + gamma * gae_lambda * next_advantage * mask
        next_return = returns[t]
        next_value = value[t]
        next_advantage = advantages[t]

    obs_flat = obs_full.view(-1, obs_full.shape[-1])
    act_flat = act.view(-1, act.shape[-1])
    log_prob_flat = old_log_prob.view(-1)
    value_flat = value.view(-1)
    return_flat = returns.view(-1)
    adv_flat = advantages.view(-1)
    done_flat = don.view(-1)
    valid_flat = valid_mask.view(-1) & torch.isfinite(log_prob_flat) & (~done_flat)

    if int(valid_flat.sum().item()) < 32:
        return None

    adv_selected = adv_flat[valid_flat]
    adv_selected = (adv_selected - adv_selected.mean()) / (adv_selected.std(unbiased=False) + 1e-6)

    return {
        "obs": obs_flat[valid_flat],
        "act": act_flat[valid_flat],
        "old_log_prob": log_prob_flat[valid_flat],
        "old_value": value_flat[valid_flat],
        "returns": return_flat[valid_flat],
        "advantages": adv_selected,
        "reward_mean": float(raw_rew[valid_mask].mean().item()),
        "scaled_reward_mean": float(rew[valid_mask].mean().item()),
    }


def run_training_step(
    model: nn.Module,
    optimizer: optim.Optimizer,
    batch: Dict[str, torch.Tensor],
    scheduler: Optional[optim.lr_scheduler.LRScheduler] = None,
) -> Dict[str, float]:
    eval_info = model.evaluate_actions(batch["obs"], batch["act"])

    log_ratio = eval_info["log_prob"] - batch["old_log_prob"]
    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(ratio, 1.0 - cfg.PPO_CLIP_EPS, 1.0 + cfg.PPO_CLIP_EPS)

    surrogate_a = ratio * batch["advantages"]
    surrogate_b = clipped_ratio * batch["advantages"]
    policy_loss = -torch.min(surrogate_a, surrogate_b).mean()
    value_loss = nn.functional.mse_loss(eval_info["value"], batch["returns"])
    entropy_bonus = eval_info["entropy"].mean()
    total_loss = policy_loss + cfg.PPO_VALUE_COEF * value_loss - cfg.PPO_ENTROPY_COEF * entropy_bonus

    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), cfg.PPO_MAX_GRAD_NORM)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    approx_kl = 0.5 * ((eval_info["log_prob"] - batch["old_log_prob"]) ** 2).mean()
    clip_frac = (torch.abs(ratio - 1.0) > cfg.PPO_CLIP_EPS).float().mean()

    return {
        "loss": float(total_loss.item()),
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "entropy": float(entropy_bonus.item()),
        "approx_kl": float(approx_kl.item()),
        "clip_frac": float(clip_frac.item()),
        "reward_mean": float(batch["reward_mean"]),
        "scaled_reward_mean": float(batch.get("scaled_reward_mean", batch["reward_mean"])),
        "lr": float(optimizer.param_groups[0]["lr"]),
    }


class AsyncTrainingWorker:
    """
    Async PPO-lite worker with snapshot-based model sync.
    """

    def __init__(
        self,
        actor_critic: nn.Module,
        ring_buffer: ReplayRingBuffer,
        device: torch.device,
        batch_size: int = 16,
        checkpoint_path: Optional[str] = None,
        save_interval_steps: int = 50,
        scheduler_steps: int = cfg.PPO_SCHEDULER_STEPS,
    ):
        self.main_model = actor_critic
        self.ring_buffer = ring_buffer
        self.device = device
        self.batch_size = batch_size
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT_PATH
        self.save_interval_steps = save_interval_steps

        self.internal_model = copy.deepcopy(actor_critic).to(device)
        self.optimizer = optim.Adam(self.internal_model.parameters(), lr=cfg.PPO_LR)
        self.scheduler = build_linear_decay_scheduler(self.optimizer, scheduler_steps)

        self.is_running = False
        self.thread = None
        self.crash_count = 0
        self.train_step = 0
        self.last_loss = 0.0
        self.last_stats: Dict[str, float] = {}

        self.update_lock = threading.Lock()
        self.has_new_weights = False
        self.pending_state_dict = None

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self.thread = threading.Thread(target=self._training_loop, daemon=True, name="PPO_Training_Worker")
        self.thread.start()
        print("[v17.1 AWAKENING] Async Training Worker Online (PPO-lite).")

    def stop(self):
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2.0)
            print("[v17.1 AWAKENING] Async Training Worker Offline.")

    def sync_to_main(self) -> bool:
        with self.update_lock:
            if not self.has_new_weights or self.pending_state_dict is None:
                return False
            snapshot = self.pending_state_dict
            self.pending_state_dict = None
            self.has_new_weights = False

        self.main_model.load_state_dict(snapshot)
        return True

    def _snapshot_weights(self):
        with self.update_lock:
            self.pending_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in self.internal_model.state_dict().items()
            }
            self.has_new_weights = True

    def _save_checkpoint(self):
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.internal_model.state_dict(), self.checkpoint_path)

    def _training_loop(self):
        while self.is_running:
            try:
                batch_np = self.ring_buffer.sample_batch(self.batch_size)
                batch = prepare_training_batch(batch_np, self.device) if batch_np else None

                if batch is None:
                    time.sleep(0.1)
                    continue

                loss_info = run_training_step(self.internal_model, self.optimizer, batch, scheduler=self.scheduler)
                self.train_step += 1
                loss_info["optimizer_step"] = float(self.train_step)
                self.last_loss = loss_info["loss"]
                self.last_stats = loss_info
                self._snapshot_weights()

                if self.train_step % self.save_interval_steps == 0:
                    self._save_checkpoint()

                if self.train_step % 10 == 0:
                    print(
                        f"[AI WORKER] Step {self.train_step} | "
                        f"Loss: {loss_info['loss']:.4f} | "
                        f"Policy: {loss_info['policy_loss']:.4f} | "
                        f"Value: {loss_info['value_loss']:.4f} | "
                        f"Entropy: {loss_info['entropy']:.4f}"
                    )

            except Exception as exc:
                self.crash_count += 1
                print(f"\n[AI WORKER CRASH] Step: {self.train_step} | Error: {exc}")
                print(traceback.format_exc())
                time.sleep(1.0)
