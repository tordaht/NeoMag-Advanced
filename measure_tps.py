import json
import os
import subprocess
import sys

import numpy as np


def run_isolated_sample(seed: int, run_length: int) -> float:
    code = f"""
import json
import time
import numpy as np
from primordial.core.world import PrimordialWorld
from primordial.core import config as cfg

world = PrimordialWorld(headless=True, seed={seed})
world.reset(seed={seed})
dummy_actions = np.zeros((cfg.MAX_AGENTS, cfg.ACTION_DIM), dtype=np.float32)

for _ in range(10):
    world.step(dummy_actions)

start_time = time.perf_counter()
for _ in range({run_length}):
    world.step(dummy_actions)
elapsed = time.perf_counter() - start_time
print(json.dumps({{"tps": {run_length} / elapsed}}))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())

    for line in reversed(result.stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            return float(json.loads(line)["tps"])
        except Exception:
            continue
    raise RuntimeError("TPS sample output could not be parsed.")


def measure_tps(run_length=500, sample_count=3):
    """Honest TPS methodology for the explicit-bridge runtime."""
    print("--- [PERF] DURUST TPS METODOLOJISI ---")
    print("Baslatiliyor (Headless Mode)...\n")

    tps_results = []

    for i in range(sample_count):
        tps = run_isolated_sample(seed=42 + i, run_length=run_length)
        tps_results.append(tps)
        print(f"Sample {i + 1}: {tps:.2f} TPS")

    avg_tps = float(np.mean(tps_results))
    peak_tps = float(np.max(tps_results))

    print("\n--- [NIHAI PERFORMANS RAPORU] ---")
    print("Mode               : Headless (Explicit CPU Bridge Active)")
    print("Environment        : RTX 5080 (sm_120) / CUDA 11+")
    print(f"Sample Count       : {sample_count}")
    print(f"Run Length         : {run_length} Steps")
    print("Measurement Method : isolated subprocess + perf_counter()")
    print("-----------------------------------")
    print(f"Average TPS        : {avg_tps:.2f}")
    print(f"Peak TPS           : {peak_tps:.2f}")
    print("-----------------------------------")

    assert avg_tps > 280.0, f"TPS 280'in altina dustu! (Olculen: {avg_tps:.2f})"
    print("[MUHUR] Performans kriteri (TPS >= 280) durustce karsilandi.")


if __name__ == "__main__":
    measure_tps()
