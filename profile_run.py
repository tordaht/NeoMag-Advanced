import taichi as ti
ti.init(arch=ti.cuda, kernel_profiler=True)

from primordial.core.world import PrimordialWorld
from primordial.core import config as cfg
import numpy as np
import time

def run_profiler():
    print("Initializing World for Profiling...")
    world = PrimordialWorld(headless=True)
    world.reset()
    
    # Warmup
    for _ in range(10):
        world.step()
        
    print("Running 100 steps for Kernel Profiling...")
    ti.profiler.clear_kernel_profiler_info()
    
    start_time = time.time()
    for _ in range(100):
        world.step()
    ti.sync()
    elapsed = time.time() - start_time
    
    print(f"Elapsed Time for 100 steps: {elapsed:.3f} seconds")
    print(f"FPS: {100 / elapsed:.1f}")
    
    print("\n--- TAICHI KERNEL PROFILER OUTPUT ---")
    ti.profiler.print_kernel_profiler_info()

if __name__ == "__main__":
    run_profiler()
