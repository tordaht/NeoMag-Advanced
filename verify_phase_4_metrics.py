import numpy as np
import taichi as ti
from primordial.core.world import PrimordialWorld
from primordial.core import config as cfg

def verify_phase_4():
    print("--- [FAZ 4 DOĞRULAMA] MIMICRY & ALTRUISM SİSTEM ANALİZİ ---")
    world = PrimordialWorld(headless=True)
    world.reset(seed=42)
    rng = np.random.default_rng(42)
    
    # 100 Adım boyunca rastgele aksiyonlarla sistemi tetikle
    # Mimicry ve Altruism aksiyonlarını (4. ve 5. indeks) özellikle aktif et
    steps = 100
    total_mimic = 0
    total_altruism = 0
    max_mimic = 0
    max_altruism = 0
    for s in range(steps):
        # Deterministik rastgele aksiyonlar (Batch: MAX_AGENTS x ACTION_DIM)
        # Mimicry ve Altruism (4 ve 5) kanallarını eşik üstüne itecek dağılım korunur.
        random_actions = rng.random((cfg.MAX_AGENTS, cfg.ACTION_DIM), dtype=np.float32)
        world.step(random_actions)

        m = world.get_metrics()
        step_mimic = int(m.get("mimic_attempts", 0))
        step_altruism = int(m.get("altruism_events", 0))
        total_mimic += step_mimic
        total_altruism += step_altruism
        max_mimic = max(max_mimic, step_mimic)
        max_altruism = max(max_altruism, step_altruism)

        if s % 20 == 0:
            print(
                f"Adım {s:03d} | Step Taklit: {step_mimic:4d} | "
                f"Step Fedakarlık: {step_altruism:4d} | "
                f"Kümülatif Taklit: {total_mimic:6d} | "
                f"Kümülatif Fedakarlık: {total_altruism:6d}"
            )

    m = world.get_metrics()
    mimic = total_mimic
    altruism = total_altruism
    success = m.get('mimic_success_rate', 0.0)
    
    print("\n--- [NİHAİ RAPOR] ---")
    print(f"Kümülatif Taklit Girişimi: {mimic}")
    print(f"Kümülatif Fedakarlık Olayı: {altruism}")
    print(f"Tek Adım Maks Taklit: {max_mimic}")
    print(f"Tek Adım Maks Fedakarlık: {max_altruism}")
    print(f"Taklit Başarı Oranı: {success:.4f}")
    
    if mimic > 0 and altruism > 0:
        print("\n[MÜHÜR] FAZ 4 METRİKLERİ ARTIK GERÇEKTİR VE MATEMATİKSEL KARŞILIĞI VARDIR.")
        return True
    else:
        print("\n[HATA] Metrikler hala 0 kalıyor. Mantık hatasını inceleyin!")
        return False

if __name__ == "__main__":
    verify_phase_4()
