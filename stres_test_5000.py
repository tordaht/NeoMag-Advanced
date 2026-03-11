import taichi as ti
import numpy as np
from primordial.core.world import PrimordialWorld
from primordial.core.metrics import MetricsCore
from primordial.core import config as cfg

def run_stress_test():
    print("--- TASK-001: 5000 ADIM STRES TESTİ BAŞLATILDI ---")
    world = PrimordialWorld(headless=True)
    world.reset()
    metrics_engine = MetricsCore()
    
    for s in range(5001):
        world.step()
        
        if s % 100 == 0:
            m = world.get_metrics()
            metrics_engine.log(m)
            print(f"Adım {s:04d} | Otçul: {m['prey_count']:4d} | Etçil: {m['pred_count']:4d} | Toplam: {m['alive_count']:4d}")
            
            if m['prey_count'] == 0 or m['pred_count'] == 0:
                print("[HATA] Homeostaz 5000 adımdan önce çöktü!")
                return False
                
    print("--- TEST BAŞARILI: 5000 Adım Tamamlandı. Veriler CSV'ye mühürlendi. ---")
    return True

if __name__ == "__main__":
    run_stress_test()
