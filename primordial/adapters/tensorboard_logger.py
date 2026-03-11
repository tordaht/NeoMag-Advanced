import os

from torch.utils.tensorboard import SummaryWriter


class PrimordialLogger:
    """
    Lightweight logger for ecology, training, and social behavior metrics.
    """

    def __init__(self, log_dir="runs/v17_honest_learning"):
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.step = 0

    def log_step(self, metrics, training_info=None):
        training_info = training_info or {}

        for key in (
            "alive_count",
            "prey_count",
            "pred_count",
            "avg_energy",
            "signal_activity",
            "mimic_attempts",
            "mimic_success",
            "mimic_success_rate",
            "mimic_cooldown_blocks",
            "mimic_signal_cost_total",
            "mimic_spam_penalty_total",
            "altruism_events",
            "altruism_transfer_amount",
            "altruism_recipient_count",
            "altruism_transfer_rate",
            "altruism_donor_loss",
            "altruism_kin_reward",
            "altruism_thermo_gap",
            "avg_visibility",
            "avg_alien_mismatch",
            "avg_culture_drag",
            "avg_signal_anomaly",
            "avg_territorial_pressure",
            "territorial_pressure_energy_loss",
            "avg_mimic_failure_streak",
            "dialect_entropy",
            "dialect_divergence",
            "emerald_signal_density",
            "amber_signal_density",
            "indigo_signal_density",
            "regional_marker_density",
            "territorial_overlap",
            "active_signal_density",
        ):
            if key in metrics:
                self.writer.add_scalar(f"world/{key}", metrics[key], self.step)

        for key in (
            "loss",
            "policy_loss",
            "value_loss",
            "entropy",
            "approx_kl",
            "clip_frac",
            "reward_mean",
            "scaled_reward_mean",
            "lr",
            "optimizer_step",
        ):
            if key in training_info:
                self.writer.add_scalar(f"train/{key}", training_info[key], self.step)

        self.step += 1

    def close(self):
        self.writer.close()
