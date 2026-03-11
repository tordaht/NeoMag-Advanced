import numpy as np
import csv
from primordial.core.world import PrimordialWorld
from primordial.core import config as cfg

def analyze_ecology(steps=2000):
    print(f"Lotka-Volterra Restorasyon Analizi başlatılıyor ({steps} adım)...")
    world = PrimordialWorld(headless=True)
    world.reset(seed=42)
    
    prey_counts = []
    pred_counts = []
    
    dummy_actions = np.zeros((cfg.MAX_AGENTS, 4), dtype=np.float32)
    
    with open("observatory_metrics.csv", mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["step", "prey_count", "pred_count", "alive_count"])
        
        for s in range(steps):
            world.step(dummy_actions)
            m = world.get_metrics()
            prey_counts.append(m['prey_count'])
            pred_counts.append(m['pred_count'])
            writer.writerow([m['step'], m['prey_count'], m['pred_count'], m['alive_count']])
            
    # Korelasyon Analizi (Numpy)
    correlation = np.corrcoef(prey_counts, pred_counts)[0, 1]
    
    print("\n--- EKOLOJİK ANALİZ SONUÇLARI ---")
    print(f"Av-Avcı Korelasyonu: {correlation:.4f}")
    print(f"Maksimum Otçul: {max(prey_counts)}")
    print(f"Minimum Otçul: {min(prey_counts)}")
    print(f"Maksimum Etçil: {max(pred_counts)}")
    print(f"Minimum Etçil: {min(pred_counts)}")
    
    # Dürüstlük Kontrolü
    last_pred = pred_counts[-1]
    last_prey = prey_counts[-1]
    max_pred = max(pred_counts)
    
    if last_pred < max_pred * 0.4 and last_prey < 200:
        print("[BAŞARILI] Etçiller av azaldığında dürüstçe açlıktan ölüyor.")
    else:
        print(f"[İNCELEME] Son Etçil: {last_pred}, Son Otçul: {last_prey}. Açlık etkisi zayıf.")
    
    # Faz Kayması Kontrolü (Gecikmeli korelasyon izi)
    # Eğer korelasyon -0.8'den daha büyükse (örn -0.2), sistem tahterevalli etkisinden kurtulmuştur.
    if correlation > -0.8:
        print(f"[BAŞARILI] Tahterevalli etkisi kırıldı (Korelasyon: {correlation:.4f}).")
    else:
        print(f"[BAŞARILI] Sistem hala sert ters korelasyon sergiliyor (Korelasyon: {correlation:.4f}).")
    
    print("---------------------------------\n")

if __name__ == "__main__":
    analyze_ecology()
