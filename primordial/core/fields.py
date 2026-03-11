import taichi as ti
from . import config as cfg

@ti.data_oriented
class EnvironmentalFields:
    """
    v17.1.0-CULTURE Core Substrate: Dual Quorum + Cultural Markers.
    Supports 3-channel symbolic families for collective memory.
    """
    def __init__(self):
        self.nutrients = ti.field(dtype=ti.f32, shape=cfg.FIELD_RES)
        self.prey_quorum = ti.field(dtype=ti.f32, shape=cfg.FIELD_RES)
        self.pred_quorum = ti.field(dtype=ti.f32, shape=cfg.FIELD_RES)
        self.active_signal = ti.field(dtype=ti.f32, shape=cfg.FIELD_RES)
        self.culture = ti.Vector.field(cfg.CULTURE_CHANNELS, dtype=ti.f32, shape=cfg.FIELD_RES)
        self.thermal = ti.field(dtype=ti.f32, shape=cfg.FIELD_RES)
        self.carcasses = ti.field(dtype=ti.f32, shape=cfg.FIELD_RES)
        self.hazard = ti.field(dtype=ti.f32, shape=cfg.FIELD_RES) 
        
        self._new_nutrients = ti.field(dtype=ti.f32, shape=cfg.FIELD_RES)
        self._new_prey_q = ti.field(dtype=ti.f32, shape=cfg.FIELD_RES)
        self._new_pred_q = ti.field(dtype=ti.f32, shape=cfg.FIELD_RES)
        self._new_signal = ti.field(dtype=ti.f32, shape=cfg.FIELD_RES)
        self._new_culture = ti.Vector.field(cfg.CULTURE_CHANNELS, dtype=ti.f32, shape=cfg.FIELD_RES)

    @ti.func
    def random_noise(self, p):
        return ti.math.fract(ti.sin(p.dot(ti.Vector([127.1, 311.7]))) * 43758.5453)

    @ti.func
    def fbm(self, p):
        v = 0.0
        a = 0.5
        for _ in range(3):
            i = ti.math.floor(p)
            f = ti.math.fract(p)
            u = f * f * (3.0 - 2.0 * f)
            n = ti.math.mix(ti.math.mix(self.random_noise(i), self.random_noise(i + ti.Vector([1, 0])), u.x),
                           ti.math.mix(self.random_noise(i + ti.Vector([0, 1])), self.random_noise(i + ti.Vector([1, 1])), u.x), u.y)
            v += a * n
            p *= 2.0
            a *= 0.5
        return v

    @ti.kernel
    def seed_resources(self):
        self.nutrients.fill(0.0)
        self.prey_quorum.fill(0.0)
        self.pred_quorum.fill(0.0)
        self.active_signal.fill(0.0)
        self.culture.fill(0.0)
        self.carcasses.fill(0.0)
        self.hazard.fill(0.0)
        for i, j in self.nutrients:
            p = ti.Vector([float(i), float(j)]) * 0.1
            if self.fbm(p) > 0.6:
                self.nutrients[i, j] = 5.0
            self.thermal[i, j] = self.fbm(p * 0.5)

    @ti.kernel
    def update(self, t: float):
        for i, j in self.nutrients:
            # 1. Diffusion Orchestration (Hat 3 + Culture)
            t_nut, t_prey, t_pred, t_sig = 0.0, 0.0, 0.0, 0.0
            t_cult = ti.Vector([0.0, 0.0, 0.0])
            
            for ni, nj in ti.static(ti.ndrange((-1, 2), (-1, 2))):
                idx = ((i + ni) % cfg.FIELD_RES[0], (j + nj) % cfg.FIELD_RES[1])
                t_nut += self.nutrients[idx]
                t_prey += self.prey_quorum[idx]
                t_pred += self.pred_quorum[idx]
                t_sig += self.active_signal[idx]
                t_cult += self.culture[idx]
            
            p_frac = ti.Vector([float(i), float(j)]) * 0.08 + t * 0.1
            regrow = 0.0
            if self.fbm(p_frac) > 0.65: regrow = cfg.NUTRIENT_REGROWTH_RATE
                
            self._new_nutrients[i, j] = (t_nut / 9.0) * 0.98 + regrow
            self._new_prey_q[i, j] = (t_prey / 9.0) * (1.0 - cfg.QUORUM_DECAY)
            self._new_pred_q[i, j] = (t_pred / 9.0) * (1.0 - cfg.QUORUM_DECAY)
            self._new_signal[i, j] = (t_sig / 9.0) * (1.0 - cfg.ACTIVE_SIGNAL_DECAY)
            
            # Cultural Diffusion & Decay (Phase 3)
            self._new_culture[i, j] = (t_cult / 9.0) * (1.0 - cfg.CULTURE_DECAY)
            
            if self.carcasses[i, j] > 0: self.carcasses[i, j] *= 0.98
            self.hazard[i, j] *= 0.95

        for i, j in self.nutrients:
            self.nutrients[i, j] = self._new_nutrients[i, j]
            self.prey_quorum[i, j] = self._new_prey_q[i, j]
            self.pred_quorum[i, j] = self._new_pred_q[i, j]
            self.active_signal[i, j] = self._new_signal[i, j]
            self.culture[i, j] = self._new_culture[i, j]

    @ti.kernel
    def _get_culture_density_kernel(self) -> ti.types.vector(3, ti.f32):
        total = ti.Vector([0.0, 0.0, 0.0])
        for i, j in self.culture:
            total += self.culture[i, j]
        return total / float(cfg.FIELD_RES[0] * cfg.FIELD_RES[1])

    def get_culture_metrics(self) -> dict:
        """Returns average cultural density for each channel."""
        density = self._get_culture_density_kernel()
        return {
            "emerald_density": float(density[0]),
            "amber_density": float(density[1]),
            "indigo_density": float(density[2]),
            "active_signal_density": float(self.active_signal.to_numpy().mean())
        }
