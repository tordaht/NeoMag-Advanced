import argparse
import csv
import json
from pathlib import Path

import numpy as np


TRIBES = ("emerald", "amber", "indigo")


def load_rows(metrics_path: Path):
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def canonicalize_rows(rows):
    canonical = {}
    for order, row in enumerate(rows):
        session_id = row.get("session_id") or "legacy"
        log_index = int(row.get("log_index") or order + 1)
        try:
            step = int(float(row.get("step") or 0))
        except ValueError:
            continue
        key = (session_id, step)
        row_copy = dict(row)
        row_copy["session_id"] = session_id
        row_copy["log_index"] = log_index
        canonical[key] = row_copy
    return sorted(canonical.values(), key=lambda item: (item["session_id"], int(item["log_index"])))


def write_canonical_csv(rows, output_path: Path):
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def dominance_windows(rows):
    windows = []
    current = None
    for row in rows:
        counts = {tribe: int(float(row.get(f"{tribe}_count", 0) or 0)) for tribe in TRIBES}
        tribe = max(counts, key=counts.get)
        step = int(float(row.get("step", 0) or 0))
        if current is None or current["tribe"] != tribe:
            if current is not None:
                current["end_step"] = step
                windows.append(current)
            current = {"tribe": tribe, "start_step": step, "end_step": step}
        else:
            current["end_step"] = step
    if current is not None:
        windows.append(current)
    return windows


def crossover_events(windows):
    events = []
    for previous, current in zip(windows, windows[1:]):
        events.append(
            {
                "step": current["start_step"],
                "from": previous["tribe"],
                "to": current["tribe"],
            }
        )
    return events


def tribe_report(rows):
    if not rows:
        return {}

    trend = [
        {
            "session_id": row.get("session_id", "legacy"),
            "step": int(float(row.get("step", 0) or 0)),
            "emerald_count": int(float(row.get("emerald_count", 0) or 0)),
            "amber_count": int(float(row.get("amber_count", 0) or 0)),
            "indigo_count": int(float(row.get("indigo_count", 0) or 0)),
        }
        for row in rows
    ]

    count_matrix = np.asarray(
        [[float(row.get(f"{tribe}_count", 0) or 0) for row in rows] for tribe in TRIBES],
        dtype=np.float64,
    )
    corr = np.corrcoef(count_matrix) if count_matrix.shape[1] > 1 else np.eye(len(TRIBES))
    windows = dominance_windows(rows)

    return {
        "tribe_population_trend": trend,
        "tribe_correlation_matrix": {
            "tribes": list(TRIBES),
            "matrix": corr.tolist(),
        },
        "dominance_windows": windows,
        "crossover_events": crossover_events(windows),
        "territorial_overlap": {
            "mean": float(np.mean([float(row.get("territorial_overlap", 0.0) or 0.0) for row in rows])),
            "max": float(np.max([float(row.get("territorial_overlap", 0.0) or 0.0) for row in rows])),
        },
        "signal_density_by_tribe": {
            tribe: float(np.mean([float(row.get(f"{tribe}_signal_density", 0.0) or 0.0) for row in rows]))
            for tribe in TRIBES
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Build canonical metrics CSV and tribe analysis report.")
    parser.add_argument("--metrics", type=Path, default=Path("observatory_metrics.csv"))
    parser.add_argument("--canonical-output", type=Path, default=Path("runs") / "observatory_metrics_canonical.csv")
    parser.add_argument("--tribe-output", type=Path, default=Path("runs") / "tribe_analysis_report.json")
    args = parser.parse_args()

    args.canonical_output.parent.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.metrics)
    canonical_rows = canonicalize_rows(rows)
    write_canonical_csv(canonical_rows, args.canonical_output)
    report = tribe_report(canonical_rows)
    args.tribe_output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"canonical_rows": len(canonical_rows), "tribe_report": args.tribe_output.as_posix()}, indent=2))


if __name__ == "__main__":
    main()
