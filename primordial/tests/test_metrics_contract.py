import csv

from primordial.core.metrics import MetricsCore
from evaluate_honest_learning import summarize_indigo_repeatability, summarize_variance


def test_metrics_core_logs_real_social_columns(tmp_path):
    metrics_path = tmp_path / "observatory_metrics.csv"
    logger = MetricsCore(filename=str(metrics_path))
    logger.log(
        {
            "step": 12,
            "prey_count": 100,
            "pred_count": 7,
            "alive_count": 107,
            "avg_energy": 9.5,
            "signal_activity": 0.2,
            "emerald_count": 50,
            "amber_count": 30,
            "indigo_count": 27,
            "mimic_attempts": 8,
            "mimic_success": 3,
            "mimic_success_rate": 0.375,
            "avg_territorial_pressure": 0.12,
            "territorial_pressure_energy_loss": 1.75,
            "altruism_events": 2,
            "altruism_transfer_amount": 5.0,
            "altruism_recipient_count": 4,
            "altruism_transfer_rate": 2.5,
            "emerald_density": 0.1,
            "amber_density": 0.2,
            "indigo_density": 0.3,
            "regional_marker_density": 0.4,
            "active_signal_density": 0.5,
        }
    )

    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))

    header_index = {name: idx for idx, name in enumerate(rows[0])}

    assert rows[0][header_index["mimic_success"]] == "mimic_success"
    assert rows[0][header_index["avg_territorial_pressure"]] == "avg_territorial_pressure"
    assert rows[0][header_index["altruism_transfer_rate"]] == "altruism_transfer_rate"
    assert rows[0][header_index["active_signal_density"]] == "active_signal_density"
    assert rows[1][header_index["mimic_success"]] == "3"
    assert rows[1][header_index["territorial_pressure_energy_loss"]] == "1.75"
    assert rows[1][header_index["altruism_transfer_rate"]] == "2.5"
    assert rows[1][header_index["active_signal_density"]] == "0.5"


def test_summarize_variance_reports_multiple_cto_metrics():
    variance = summarize_variance(
        [
            {
                "avg_reward": 1.0,
                "mimic_success": 2,
                "mimic_success_rate": 0.2,
                "altruism_events": 3,
                "altruism_transfer_amount": 4.0,
                "alive_mean": 100.0,
                "alive_final": 90.0,
                "avg_culture_drag": 0.2,
                "avg_territorial_pressure": 0.3,
                "territorial_pressure_energy_loss": 2.0,
                "indigo_final_share": 0.25,
            },
            {
                "avg_reward": 3.0,
                "mimic_success": 6,
                "mimic_success_rate": 0.6,
                "altruism_events": 1,
                "altruism_transfer_amount": 8.0,
                "alive_mean": 120.0,
                "alive_final": 110.0,
                "avg_culture_drag": 0.4,
                "avg_territorial_pressure": 0.5,
                "territorial_pressure_energy_loss": 4.0,
                "indigo_final_share": 0.15,
            },
        ]
    )

    assert variance["avg_reward_mean"] == 2.0
    assert variance["mimic_success_std"] > 0.0
    assert variance["alive_final_mean"] == 100.0
    assert variance["avg_territorial_pressure_mean"] == 0.4


def test_indigo_repeatability_summary_requires_multi_seed_survival():
    summary = summarize_indigo_repeatability(
        [
            {"indigo_final": 20, "indigo_final_share": 0.2, "avg_territorial_pressure": 0.3, "mimic_success_rate": 0.08},
            {"indigo_final": 15, "indigo_final_share": 0.1, "avg_territorial_pressure": 0.4, "mimic_success_rate": 0.07},
            {"indigo_final": 18, "indigo_final_share": 0.12, "avg_territorial_pressure": 0.5, "mimic_success_rate": 0.06},
        ]
    )

    assert summary["survived_all_seeds"] is True
    assert summary["repeatable_strategy"] is True
