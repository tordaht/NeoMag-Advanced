import json
from pathlib import Path

import numpy as np

from primordial.training.train_ppo import train


SEEDS = (42, 43, 44)
STEPS = 5000


def _variance(values):
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)) if arr.size else 0.0,
        "std": float(np.std(arr)) if arr.size else 0.0,
        "min": float(np.min(arr)) if arr.size else 0.0,
        "max": float(np.max(arr)) if arr.size else 0.0,
    }


def main():
    runs_dir = Path("runs") / "acceptance_emergence"
    runs_dir.mkdir(parents=True, exist_ok=True)

    aggregate = {
        "config": {
            "steps": STEPS,
            "seeds": list(SEEDS),
        },
        "runs": [],
    }

    for seed in SEEDS:
        checkpoint = runs_dir / f"acceptance_seed_{seed}.pt"
        result = train(
            total_steps=STEPS,
            checkpoint_path=checkpoint,
            save_report=True,
            seed=seed,
            eval_seeds=(seed,),
            log_dir=str(runs_dir / f"tensorboard_seed_{seed}"),
        )
        aggregate["runs"].append(
            {
                "seed": seed,
                "checkpoint": str(checkpoint),
                "report": str(checkpoint.with_suffix(".eval.json")),
                "loss_delta": float(result["evidence"]["loss_trend"]["delta"]),
                "late_energy": float(result["evidence"]["energy_profile"]["late_mean"]),
                "lag_corr": float(result["evidence"]["lotka_volterra"]["best_corr"]),
                "lag": int(result["evidence"]["lotka_volterra"]["best_lag"]),
                "predator_extinct": bool(result["evidence"]["lotka_volterra"]["predator_extinct"]),
                "trained_eval": result["runs"][0],
            }
        )

        (runs_dir / "acceptance_aggregate.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

    aggregate["variance"] = {
        "loss_delta": _variance([item["loss_delta"] for item in aggregate["runs"]]),
        "late_energy": _variance([item["late_energy"] for item in aggregate["runs"]]),
        "lag_corr": _variance([item["lag_corr"] for item in aggregate["runs"]]),
        "eval_avg_energy": _variance([item["trained_eval"]["avg_energy"] for item in aggregate["runs"]]),
        "eval_avg_reward": _variance([item["trained_eval"]["avg_reward"] for item in aggregate["runs"]]),
    }
    aggregate["acceptance"] = {
        "all_predators_survived": all(not item["predator_extinct"] for item in aggregate["runs"]),
        "all_loss_improved": all(item["loss_delta"] < 0.0 for item in aggregate["runs"]),
        "energy_band_ok": all(70.0 <= item["late_energy"] <= 120.0 for item in aggregate["runs"]),
        "lag_correlation_ok": all(item["lag_corr"] >= 0.25 for item in aggregate["runs"]),
    }

    output_path = runs_dir / "acceptance_aggregate.json"
    output_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

    markdown = [
        "# Acceptance Emergence Report",
        "",
        f"- Steps per seed: `{STEPS}`",
        f"- Seeds: `{', '.join(str(seed) for seed in SEEDS)}`",
        "",
        "## Run Summary",
    ]
    for item in aggregate["runs"]:
        markdown.append(
            f"- Seed `{item['seed']}`: loss delta `{item['loss_delta']:.4f}`, late energy `{item['late_energy']:.2f}`, "
            f"lag corr `{item['lag_corr']:.4f}` @ lag `{item['lag']}`, predator extinct `{item['predator_extinct']}`"
        )
    markdown.extend(
        [
            "",
            "## Acceptance Gates",
            f"- All predators survived: `{aggregate['acceptance']['all_predators_survived']}`",
            f"- All loss deltas improved: `{aggregate['acceptance']['all_loss_improved']}`",
            f"- Late energy band OK: `{aggregate['acceptance']['energy_band_ok']}`",
            f"- Lag correlation OK: `{aggregate['acceptance']['lag_correlation_ok']}`",
        ]
    )
    (runs_dir / "acceptance_report.md").write_text("\n".join(markdown), encoding="utf-8")

    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
