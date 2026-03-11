import argparse
import json
import os
import platform
import threading
import time
from pathlib import Path

import numpy as np
import torch

from evaluate_honest_learning import run_validation
from primordial.core import config as cfg
from primordial.core.world import PrimordialWorld
from primordial.training.policy import PrimordialPolicy


def hardware_report():
    gpu_name = "cpu"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "gpu": gpu_name,
    }


@torch.no_grad()
def _sample_action(actor, world):
    obs = world.get_observations_torch(device="cpu")
    sample = actor.sample_actions(obs)
    return sample.sampled_action


def benchmark_headless_tps(duration_s: float):
    world = PrimordialWorld(headless=True)
    world.reset(seed=42)
    actor = PrimordialPolicy().to("cpu")

    steps = 0
    start = time.perf_counter()
    while time.perf_counter() - start < duration_s:
        world.step(_sample_action(actor, world))
        steps += 1
    elapsed = max(time.perf_counter() - start, 1e-6)
    return {
        "mode": "headless",
        "measurement_seconds": elapsed,
        "sample_count": steps,
        "headless_tps": steps / elapsed,
    }


def benchmark_ui_open(duration_s: float, target_render_fps: float):
    world = PrimordialWorld(headless=False)
    world.reset(seed=42)
    actor = PrimordialPolicy().to("cpu")
    world_lock = threading.Lock()
    stop_event = threading.Event()
    counters = {"steps": 0, "frames": 0}

    def sim_loop():
        while not stop_event.is_set():
            with world_lock:
                world.step(_sample_action(actor, world))
            counters["steps"] += 1

    def render_loop():
        frame_interval = 1.0 / max(target_render_fps, 1.0)
        next_frame = time.perf_counter()
        while not stop_event.is_set():
            now = time.perf_counter()
            if now < next_frame:
                time.sleep(next_frame - now)
                continue
            with world_lock:
                raw = world.render(1, float(cfg.WORLD_RES[0] / 2), float(cfg.WORLD_RES[1] / 2), 1.0)
            if raw is not None:
                _ = raw.transpose(1, 0, 2)
            counters["frames"] += 1
            next_frame += frame_interval

    sim_thread = threading.Thread(target=sim_loop, daemon=True, name="BenchmarkSim")
    render_thread = threading.Thread(target=render_loop, daemon=True, name="BenchmarkRender")

    start = time.perf_counter()
    sim_thread.start()
    render_thread.start()
    time.sleep(duration_s)
    stop_event.set()
    sim_thread.join(timeout=2.0)
    render_thread.join(timeout=2.0)
    elapsed = max(time.perf_counter() - start, 1e-6)

    return {
        "mode": "ui_open",
        "measurement_seconds": elapsed,
        "sample_count": counters["steps"],
        "frame_samples": counters["frames"],
        "ui_open_tps": counters["steps"] / elapsed,
        "render_fps": counters["frames"] / elapsed,
        "ui_surface": "Taichi renderer active with RGB copy path; DearPyGui event pump omitted for deterministic offline bench.",
    }


def write_markdown_report(report: dict, output_md: Path):
    bench = report["benchmark"]
    validation = report["validation"]
    trained = validation.get("trained", [])
    repeatability = validation.get("indigo_repeatability", {}).get("trained", {})

    lines = [
        "# CTO Methodology Report",
        "",
        "## 1. Gercek Hiz",
        f"- Headless TPS: {bench['headless']['headless_tps']:.2f}",
        f"- UI Acik TPS: {bench['ui_open']['ui_open_tps']:.2f}",
        f"- Render FPS: {bench['ui_open']['render_fps']:.2f}",
        f"- Olcum Suresi: {bench['headless']['measurement_seconds']:.2f}s / {bench['ui_open']['measurement_seconds']:.2f}s",
        f"- Sample Count: {bench['headless']['sample_count']} / {bench['ui_open']['sample_count']}",
        f"- Donanim: CPU={bench['hardware']['cpu']} | GPU={bench['hardware']['gpu']}",
        "",
        "## 2. Gercek Ogrenme",
    ]
    for item in trained:
        lines.append(
            f"- Seed {item['seed']}: mimic_success_rate={item['mimic_success_rate']:.4f}, "
            f"mimic_cooldown_blocks={item['mimic_cooldown_blocks']}, mimic_attempts={item['mimic_attempts']}"
        )
    lines.extend(
        [
            "",
            "## 3. Gercek Kultur Surtunmesi",
        ]
    )
    for item in trained:
        lines.append(
            f"- Seed {item['seed']}: avg_territorial_pressure={item['avg_territorial_pressure']:.6f}, "
            f"territorial_pressure_energy_loss={item['territorial_pressure_energy_loss']:.4f}, avg_culture_drag={item['avg_culture_drag']:.6f}"
        )
    lines.extend(
        [
            "",
            "## 4. Indigo Tekrarlanabilirligi",
            f"- Survived All Seeds: {repeatability.get('survived_all_seeds', False)}",
            f"- Repeatable Strategy: {repeatability.get('repeatable_strategy', False)}",
            f"- Indigo Final Share Mean: {repeatability.get('indigo_final_share_mean', 0.0):.6f}",
            f"- Indigo Final Share Std: {repeatability.get('indigo_final_share_std', 0.0):.6f}",
        ]
    )
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Run CTO methodology benchmark and 3-seed validation.")
    parser.add_argument("--benchmark-seconds", type=float, default=5.0)
    parser.add_argument("--render-fps", type=float, default=30.0)
    parser.add_argument("--train-steps", type=int, default=240)
    parser.add_argument("--eval-steps", type=int, default=5000)
    parser.add_argument("--output", type=Path, default=Path("runs") / "cto_methodology_report.json")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    validation_path = args.output.with_name(args.output.stem + "_validation.json")
    output_md = args.output.with_suffix(".md")

    report = {
        "benchmark": {
            "hardware": hardware_report(),
            "headless": benchmark_headless_tps(args.benchmark_seconds),
            "ui_open": benchmark_ui_open(args.benchmark_seconds, args.render_fps),
        },
        "validation": run_validation(args.train_steps, args.eval_steps, validation_path),
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown_report(report, output_md)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
