import numpy as np
import time
from primordial.core.world import PrimordialWorld
from primordial.core import config as cfg

def capture_metrics(steps=1000):
    world = PrimordialWorld(headless=True)
    world.reset(seed=42)
    
    prey_counts = []
    pred_counts = []
    avg_speeds = []
    
    dummy_actions = np.zeros((cfg.MAX_AGENTS, 4), dtype=np.float32)
    
    print(f"Ölçüm başlatılıyor ({steps} adım)...")
    for s in range(steps):
        world.step(dummy_actions)
        m = world.get_metrics()
        prey_counts.append(m['prey_count'])
        pred_counts.append(m['pred_count'])
        
        # Ajan hızlarını çek (GPU -> CPU)
        vels = world.organisms.vel.to_numpy()
        alive_mask = world.organisms.alive.to_numpy()
        active_vels = vels[alive_mask == 1]
        if len(active_vels) > 0:
            speeds = np.linalg.norm(active_vels, axis=1)
            avg_speeds.append(np.mean(speeds))
            
    # Analiz
    prey_std = np.std(prey_counts)
    pred_std = np.std(pred_counts)
    overall_avg_speed = np.mean(avg_speeds)
    
    print("\n--- SMOOTHING ÖNCESİ METRİKLER ---")
    print(f"Otçul Oynaklığı (StdDev): {prey_std:.2f}")
    print(f"Etçil Oynaklığı (StdDev): {pred_std:.2f}")
    print(f"Ortalama Ajan Hızı: {overall_avg_speed:.4f}")
    print("----------------------------------\n")
    
    return {
        "prey_std": prey_std,
        "pred_std": pred_std,
        "avg_speed": overall_avg_speed
    }

if __name__ == "__main__":
    capture_metrics()
