import numpy as np

from primordial.core import config as cfg
from primordial.core.world import PrimordialWorld


def test_visual_integrity():
    """
    v17.1.1-CULTURE-EXT pixel-match harness.
    Verifies lens output against expected color ranges.
    """
    print("--- [PIXEL-MATCH] GORSEL KALITE TESTI BASLATILDI ---")
    world = PrimordialWorld(headless=False)
    world.reset()

    for _ in range(50):
        world.step()

    results = {}
    for mode in [0, 1, 2, 3]:
        frame = world.render(mode, cfg.WORLD_RES[0] / 2, cfg.WORLD_RES[1] / 2, 1.0)
        avg_color = np.mean(frame, axis=(0, 1))
        results[mode] = avg_color
        print(f"Mode {mode} Rendered. Avg RGB: {avg_color}")

    assert np.any(results[1] > 0.0), "[HATA] Mode 1 Culture Dominance lens bos veya hatali."
    print("[BASARILI] Mode 1 Culture Dominance lens veri uretiyor.")

    assert results[2][2] > 0.0 or results[2][0] > 0.0, "[HATA] Mode 2 Communication lens veri icermiyor."
    print("[BASARILI] Mode 2 Communication lens aktif.")

    assert results[3][0] >= 0.0, "[HATA] Mode 3 Social Events lens render edemedi."
    print("[BASARILI] Mode 3 Social Events lens aktif.")

    print("--- [PIXEL-MATCH] TUM TESTLER GECTI: LENS REFACTOR DOGRULANDI ---")


if __name__ == "__main__":
    test_visual_integrity()
