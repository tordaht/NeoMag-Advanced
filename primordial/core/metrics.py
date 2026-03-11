import csv
import os
import time

from . import config as cfg


class MetricsCore:
    """
    CSV logger for the observatory surface. Reinitializes automatically when the
    schema changes so the file cannot silently drift from the code.
    """

    def __init__(self, filename=cfg.METRICS_LOG_FILE):
        self.filename = filename
        self.session_id = time.strftime("%Y%m%d_%H%M%S")
        self.log_index = 0
        self.headers = [
            "session_id",
            "log_index",
            "step",
            "prey_count",
            "pred_count",
            "alive_count",
            "avg_energy",
            "signal_activity",
            "emerald_count",
            "amber_count",
            "indigo_count",
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
            "emerald_density",
            "amber_density",
            "indigo_density",
            "emerald_signal_density",
            "amber_signal_density",
            "indigo_signal_density",
            "regional_marker_density",
            "territorial_overlap",
            "active_signal_density",
        ]

        should_init = True
        if os.path.exists(self.filename):
            try:
                with open(self.filename, "r", encoding="utf-8") as handle:
                    if handle.readline().strip() == ",".join(self.headers):
                        should_init = False
            except Exception:
                pass

        if should_init:
            with open(self.filename, "w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerow(self.headers)

    def log(self, metrics_dict):
        try:
            with open(self.filename, "a", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                self.log_index += 1
                writer.writerow([
                    self.session_id,
                    self.log_index,
                    metrics_dict.get("step", 0),
                    metrics_dict.get("prey_count", 0),
                    metrics_dict.get("pred_count", 0),
                    metrics_dict.get("alive_count", 0),
                    metrics_dict.get("avg_energy", 0.0),
                    metrics_dict.get("signal_activity", 0.0),
                    metrics_dict.get("emerald_count", 0),
                    metrics_dict.get("amber_count", 0),
                    metrics_dict.get("indigo_count", 0),
                    metrics_dict.get("mimic_attempts", 0),
                    metrics_dict.get("mimic_success", 0),
                    metrics_dict.get("mimic_success_rate", 0.0),
                    metrics_dict.get("mimic_cooldown_blocks", 0),
                    metrics_dict.get("mimic_signal_cost_total", 0.0),
                    metrics_dict.get("mimic_spam_penalty_total", 0.0),
                    metrics_dict.get("altruism_events", 0),
                    metrics_dict.get("altruism_transfer_amount", 0.0),
                    metrics_dict.get("altruism_recipient_count", 0),
                    metrics_dict.get("altruism_transfer_rate", 0.0),
                    metrics_dict.get("altruism_donor_loss", 0.0),
                    metrics_dict.get("altruism_kin_reward", 0.0),
                    metrics_dict.get("altruism_thermo_gap", 0.0),
                    metrics_dict.get("avg_visibility", 0.0),
                    metrics_dict.get("avg_alien_mismatch", 0.0),
                    metrics_dict.get("avg_culture_drag", 0.0),
                    metrics_dict.get("avg_signal_anomaly", 0.0),
                    metrics_dict.get("avg_territorial_pressure", 0.0),
                    metrics_dict.get("territorial_pressure_energy_loss", 0.0),
                    metrics_dict.get("avg_mimic_failure_streak", 0.0),
                    metrics_dict.get("emerald_density", 0.0),
                    metrics_dict.get("amber_density", 0.0),
                    metrics_dict.get("indigo_density", 0.0),
                    metrics_dict.get("emerald_signal_density", 0.0),
                    metrics_dict.get("amber_signal_density", 0.0),
                    metrics_dict.get("indigo_signal_density", 0.0),
                    metrics_dict.get("regional_marker_density", metrics_dict.get("regional_marker_concentration", 0.0)),
                    metrics_dict.get("territorial_overlap", 0.0),
                    metrics_dict.get("active_signal_density", 0.0),
                ])
        except Exception as exc:
            print(f"[METRICS ERROR] CSV write failed: {exc}")
