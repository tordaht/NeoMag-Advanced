import taichi as ti
import numpy as np
from . import config as cfg

@ti.data_oriented
class SpecimenManager:
    """
    v17.1.1-CULTURE-EXT: Extended Cultural & Epigenetic Engine.
    Supports Local Dialects, 8-dim Epigenetic Instincts, and Composite Behaviors.
    Aligned with 36-dim Observation Contract.
    """
    def __init__(self):
        self.alive = ti.field(dtype=ti.i32, shape=cfg.MAX_AGENTS)
        self.type = ti.field(dtype=ti.i32, shape=cfg.MAX_AGENTS)
        self.pos = ti.Vector.field(2, dtype=ti.f32, shape=cfg.MAX_AGENTS)
        self.vel = ti.Vector.field(2, dtype=ti.f32, shape=cfg.MAX_AGENTS)
        self.angle = ti.field(dtype=ti.f32, shape=cfg.MAX_AGENTS) 
        self.energy = ti.field(dtype=ti.f32, shape=cfg.MAX_AGENTS)
        self.age = ti.field(dtype=ti.f32, shape=cfg.MAX_AGENTS)
        self.traits = ti.Vector.field(cfg.TRAIT_COUNT, dtype=ti.f32, shape=cfg.MAX_AGENTS)
        self.signal = ti.field(dtype=ti.f32, shape=cfg.MAX_AGENTS)
        self.glow = ti.field(dtype=ti.f32, shape=cfg.MAX_AGENTS)
        
        # Phase 3.1: Extended Contexts (Operatör Şef & Kodcu Entegrasyonu)
        self.dialect_state = ti.Vector.field(cfg.CULTURE_CHANNELS, dtype=ti.f32, shape=cfg.MAX_AGENTS)
        self.local_marker_ctx = ti.Vector.field(cfg.CULTURE_CHANNELS, dtype=ti.f32, shape=cfg.MAX_AGENTS)
        self.epigenetic_ctx = ti.Vector.field(cfg.EPIGENETIC_DIM, dtype=ti.f32, shape=cfg.MAX_AGENTS)
        self.behavior_ctx = ti.Vector.field(cfg.BEHAVIOR_DIM, dtype=ti.f32, shape=cfg.MAX_AGENTS)
        
        # Behavioral Counters & Timers
        self.mimic_timer = ti.field(dtype=ti.f32, shape=cfg.MAX_AGENTS)
        self.altruism_timer = ti.field(dtype=ti.f32, shape=cfg.MAX_AGENTS)
        self.mimic_attempts = ti.field(dtype=ti.i32, shape=())
        self.mimic_success = ti.field(dtype=ti.i32, shape=())
        self.altruism_events = ti.field(dtype=ti.i32, shape=())
        self.altruism_transfer_amount = ti.field(dtype=ti.f32, shape=())
        self.altruism_recipient_count = ti.field(dtype=ti.i32, shape=())
        self.altruism_donor_loss = ti.field(dtype=ti.f32, shape=())
        self.altruism_kin_reward = ti.field(dtype=ti.f32, shape=())
        self.mimic_signal_cost_total = ti.field(dtype=ti.f32, shape=())
        self.visibility_accum = ti.field(dtype=ti.f32, shape=())
        self.visibility_samples = ti.field(dtype=ti.i32, shape=())
        self.alien_mismatch_accum = ti.field(dtype=ti.f32, shape=())
        self.viscosity_drag_accum = ti.field(dtype=ti.f32, shape=())
        
        # RL Tensors (ti.ndarray for explicit bridge)
        self.observations = ti.ndarray(dtype=ti.f32, shape=(cfg.MAX_AGENTS, cfg.OBSERVATION_DIM)) 
        self.actions = ti.ndarray(dtype=ti.f32, shape=(cfg.MAX_AGENTS, cfg.ACTION_DIM))
        self.last_actions = ti.ndarray(dtype=ti.f32, shape=(cfg.MAX_AGENTS, cfg.ACTION_DIM))
        self.rewards = ti.ndarray(dtype=ti.f32, shape=cfg.MAX_AGENTS)
        
        self.grid_count = ti.field(dtype=ti.i32, shape=(cfg.WORLD_RES[0]//cfg.SPATIAL_GRID_CELL_SIZE, cfg.WORLD_RES[1]//cfg.SPATIAL_GRID_CELL_SIZE))
        self.grid_agents = ti.field(dtype=ti.i32, shape=(cfg.WORLD_RES[0]//cfg.SPATIAL_GRID_CELL_SIZE, cfg.WORLD_RES[1]//cfg.SPATIAL_GRID_CELL_SIZE, 64))
        self.repro_counts = ti.field(dtype=ti.i32, shape=3) 
        self.parents_buf = ti.field(dtype=ti.i32, shape=cfg.MAX_AGENTS)
        self.dead_buf = ti.field(dtype=ti.i32, shape=cfg.MAX_AGENTS)
        self.cultural_stats = ti.field(dtype=ti.i32, shape=cfg.CULTURE_CHANNELS)
        
        self._zero_obs_np = np.zeros((cfg.MAX_AGENTS, cfg.OBSERVATION_DIM), dtype=np.float32)
        self._zero_actions_np = np.zeros((cfg.MAX_AGENTS, cfg.ACTION_DIM), dtype=np.float32)
        self._zero_rewards_np = np.zeros(cfg.MAX_AGENTS, dtype=np.float32)

    @ti.kernel
    def reset_all(self):
        self.alive.fill(0)
        self.signal.fill(0.0)
        self.glow.fill(0.0)
        self.age.fill(0.0)
        self.mimic_timer.fill(0.0)
        self.altruism_timer.fill(0.0)
        self.mimic_attempts[None] = 0
        self.mimic_success[None] = 0
        self.altruism_events[None] = 0
        self.altruism_transfer_amount[None] = 0.0
        self.altruism_recipient_count[None] = 0
        self.altruism_donor_loss[None] = 0.0
        self.altruism_kin_reward[None] = 0.0
        self.mimic_signal_cost_total[None] = 0.0
        self.visibility_accum[None] = 0.0
        self.visibility_samples[None] = 0
        self.alien_mismatch_accum[None] = 0.0
        self.viscosity_drag_accum[None] = 0.0
        
        for i in range(cfg.INITIAL_PREY_COUNT + cfg.INITIAL_PRED_COUNT):
            self.alive[i] = 1
            self.type[i] = cfg.TYPE_PREY if i < cfg.INITIAL_PREY_COUNT else cfg.TYPE_PRED
            self.pos[i] = ti.Vector([ti.random()*cfg.WORLD_RES[0], ti.random()*cfg.WORLD_RES[1]])
            self.vel[i] = ti.Vector([0.0, 0.0])
            self.angle[i] = ti.random() * 6.28
            self.energy[i] = cfg.INITIAL_ENERGY
            self.traits[i] = ti.Vector([ti.random(), ti.random(), ti.random()]).normalized()
            self.dialect_state[i] = self.traits[i]
            
            for j in ti.static(range(cfg.EPIGENETIC_DIM)):
                self.epigenetic_ctx[i][j] = ti.random()
            
            self.behavior_ctx[i] = ti.Vector([ti.random(), ti.random(), 0.0, 0.0])

    def reset_step_counters(self):
        """Resets periodic metrics counters. Expected by world orchestrator."""
        self.mimic_attempts[None] = 0
        self.mimic_success[None] = 0
        self.altruism_events[None] = 0
        self.altruism_transfer_amount[None] = 0.0
        self.altruism_recipient_count[None] = 0
        self.altruism_donor_loss[None] = 0.0
        self.altruism_kin_reward[None] = 0.0
        self.mimic_signal_cost_total[None] = 0.0
        self.visibility_accum[None] = 0.0
        self.visibility_samples[None] = 0
        self.alien_mismatch_accum[None] = 0.0
        self.viscosity_drag_accum[None] = 0.0

    def get_culture_metrics(self) -> dict:
        visibility_samples = max(1, int(self.visibility_samples[None]))
        counts = self._get_cultural_counts_kernel()
        return {
            "emerald_count": int(counts[0]),
            "amber_count": int(counts[1]),
            "indigo_count": int(counts[2]),
            "mimic_attempts": int(self.mimic_attempts[None]),
            "mimic_success": int(self.mimic_success[None]),
            "mimic_success_rate": float(self.mimic_success[None] / max(1, self.mimic_attempts[None])),
            "mimic_signal_cost_total": float(self.mimic_signal_cost_total[None]),
            "altruism_events": int(self.altruism_events[None]),
            "altruism_transfer_amount": float(self.altruism_transfer_amount[None]),
            "altruism_recipient_count": int(self.altruism_recipient_count[None]),
            "altruism_transfer_rate": float(self.altruism_transfer_amount[None] / max(1, self.altruism_events[None])),
            "altruism_donor_loss": float(self.altruism_donor_loss[None]),
            "altruism_kin_reward": float(self.altruism_kin_reward[None]),
            "altruism_thermo_gap": float(self.altruism_donor_loss[None] - self.altruism_transfer_amount[None]),
            "avg_visibility": float(self.visibility_accum[None] / visibility_samples),
            "avg_alien_mismatch": float(self.alien_mismatch_accum[None] / visibility_samples),
            "avg_culture_drag": float(self.viscosity_drag_accum[None] / visibility_samples),
        }

    @ti.func
    def _safe_norm3(self, vec):
        return ti.sqrt(vec.dot(vec) + 1e-6)

    @ti.func
    def _culture_similarity(self, idx_a: int, idx_b: int) -> float:
        a = self.dialect_state[idx_a]
        b = self.dialect_state[idx_b]
        return ti.math.clamp(a.dot(b) / (self._safe_norm3(a) * self._safe_norm3(b)), 0.0, 1.0)

    @ti.func
    def _epigenetic_similarity(self, idx_a: int, idx_b: int) -> float:
        dot_val = 0.0
        norm_a = 0.0
        norm_b = 0.0
        for j in ti.static(range(cfg.EPIGENETIC_DIM)):
            va = self.epigenetic_ctx[idx_a][j]
            vb = self.epigenetic_ctx[idx_b][j]
            dot_val += va * vb
            norm_a += va * va
            norm_b += vb * vb
        return ti.math.clamp(dot_val / ti.sqrt((norm_a + 1e-6) * (norm_b + 1e-6)), 0.0, 1.0)

    @ti.func
    def _kin_similarity(self, idx_a: int, idx_b: int) -> float:
        epi = self._epigenetic_similarity(idx_a, idx_b)
        culture = self._culture_similarity(idx_a, idx_b)
        return ti.math.clamp(epi * cfg.KIN_EPIGENETIC_WEIGHT + culture * cfg.KIN_CULTURE_WEIGHT, 0.0, 1.0)

    @ti.func
    def _visibility_decay(self, rho: float, dist: float) -> float:
        return ti.exp(-cfg.VISION_DECAY_K * rho * dist)

    @ti.kernel
    def _get_cultural_counts_kernel(self) -> ti.types.vector(3, ti.i32):
        self.cultural_stats.fill(0)
        for i in range(cfg.MAX_AGENTS):
            if self.alive[i]:
                t = self.traits[i]
                if t[0] > t[1] and t[0] > t[2]: ti.atomic_add(self.cultural_stats[0], 1)
                elif t[1] > t[0] and t[1] > t[2]: ti.atomic_add(self.cultural_stats[1], 1)
                else: ti.atomic_add(self.cultural_stats[2], 1)
        return ti.Vector([self.cultural_stats[0], self.cultural_stats[1], self.cultural_stats[2]])

    def reset_rl_buffers(self):
        self.observations.from_numpy(self._zero_obs_np)
        self.actions.from_numpy(self._zero_actions_np)
        self.last_actions.from_numpy(self._zero_actions_np)
        self.rewards.from_numpy(self._zero_rewards_np)

    def clear_rewards(self):
        self.rewards.from_numpy(self._zero_rewards_np)

    @ti.kernel
    def update_spatial_index(self):
        self.grid_count.fill(0)
        for i in range(cfg.MAX_AGENTS):
            if self.alive[i]:
                gx, gy = int(self.pos[i].x // cfg.SPATIAL_GRID_CELL_SIZE), int(self.pos[i].y // cfg.SPATIAL_GRID_CELL_SIZE)
                if 0 <= gx < self.grid_count.shape[0] and 0 <= gy < self.grid_count.shape[1]:
                    idx = ti.atomic_add(self.grid_count[gx, gy], 1)
                    if idx < 64: self.grid_agents[gx, gy, idx] = i

    @ti.kernel
    def apply_repulsion(self):
        for i in range(cfg.MAX_AGENTS):
            if self.alive[i]:
                gx, gy = int(self.pos[i].x // cfg.SPATIAL_GRID_CELL_SIZE), int(self.pos[i].y // cfg.SPATIAL_GRID_CELL_SIZE)
                repel = ti.Vector([0.0, 0.0])
                for ox, oy in ti.static(ti.ndrange((-1, 2), (-1, 2))):
                    nx, ny = gx+ox, gy+oy
                    if 0 <= nx < self.grid_count.shape[0] and 0 <= ny < self.grid_count.shape[1]:
                        for k in range(self.grid_count[nx, ny]):
                            other = self.grid_agents[nx, ny, k]
                            if other != i and self.alive[other]:
                                diff = self.pos[i] - self.pos[other]
                                dist = diff.norm()
                                if 0.001 < dist < cfg.REPULSION_RADIUS:
                                    repel += (diff / dist) * (cfg.REPULSION_RADIUS - dist) * cfg.REPULSION_STRENGTH
                self.vel[i] += repel

    @ti.kernel
    def apply_physics_and_metabolism(self, 
        fields: ti.template(), 
        alive_count: int, 
        prey_count: int, 
        pred_count: int, 
        time: float,
        actions: ti.types.ndarray(),
        last_actions: ti.types.ndarray(),
        rewards: ti.types.ndarray()
    ):
        for i in range(cfg.MAX_AGENTS):
            if self.alive[i]:
                self.age[i] += 0.001
                
                thrust = ti.math.clamp(actions[i, 0], 0.0, 1.0)
                steer = ti.math.clamp(actions[i, 1], -1.0, 1.0)
                metabolic_shift = ti.math.clamp(actions[i, 2], 0.0, 1.0) 
                self.signal[i] = ti.math.clamp(actions[i, 3], 0.0, 1.0) 
                self.glow[i] *= 0.94 
                
                jitter_penalty = 0.0
                for j in ti.static(range(cfg.ACTION_DIM)):
                    diff = actions[i, j] - last_actions[i, j]
                    jitter_penalty += diff * diff
                rewards[i] -= jitter_penalty * 0.5 
                
                self.vel[i] *= cfg.DAMPING
                self.angle[i] += steer * 0.12 
                
                drift_x = ti.math.sin(time + self.pos[i].y * 0.01) * cfg.MICRO_CURRENT_STRENGTH
                drift_y = ti.math.cos(time * 0.7 + self.pos[i].x * 0.01) * cfg.MICRO_CURRENT_STRENGTH
                self.vel[i] += ti.Vector([drift_x, drift_y])

                intent_mag = thrust * (1.0 + metabolic_shift * 0.4)
                intent_vec = ti.Vector([ti.cos(self.angle[i]), ti.sin(self.angle[i])]) * intent_mag
                
                accel = cfg.ACCEL_SMOOTHING
                v_limit = cfg.MAX_VELOCITY
                if self.type[i] == cfg.TYPE_PRED:
                    intent_vec *= 1.4
                    accel *= 1.2 
                    v_limit *= 1.15
                else:
                    intent_vec *= 0.90
                    accel *= 0.80
                
                self.vel[i] += intent_vec * accel
                v_norm = self.vel[i].norm()
                if v_norm > v_limit: self.vel[i] = (self.vel[i] / v_norm) * v_limit
                self.pos[i] = (self.pos[i] + self.vel[i] * 2.5) % ti.Vector([cfg.WORLD_RES[0], cfg.WORLD_RES[1]])
                
                ix, iy = int(self.pos[i].x * cfg.FIELD_RES[0]/cfg.WORLD_RES[0]), int(self.pos[i].y * cfg.FIELD_RES[1]/cfg.WORLD_RES[1])
                
                # Active signal pulse for quorum
                if self.signal[i] > 0.05:
                    ti.atomic_add(fields.active_signal[ix, iy], cfg.ACTIVE_SIGNAL_DEPOSIT_RATE * self.signal[i])
                    if self.type[i] == cfg.TYPE_PREY:
                        ti.atomic_add(fields.prey_quorum[ix, iy], cfg.QUORUM_DEPOSIT_RATE * self.signal[i])
                    else:
                        ti.atomic_add(fields.pred_quorum[ix, iy], cfg.QUORUM_DEPOSIT_RATE * self.signal[i])
                
                # Cultural trail: passive identity (weak) + active pulse (stronger)
                passive_deposit = cfg.CULTURE_PASSIVE_DEPOSIT_RATE
                active_deposit = cfg.CULTURE_ACTIVE_SIGNAL_BOOST * self.signal[i]
                fields.culture[ix, iy] += self.dialect_state[i] * (passive_deposit + active_deposit)
                
                local_culture = fields.culture[ix, iy]
                local_mean = local_culture.normalized() if local_culture.norm() > 0.01 else self.dialect_state[i]
                self.local_marker_ctx[i] = local_culture
                local_culture_strength = ti.math.tanh(local_culture.norm() * cfg.ALIEN_VISCOSITY_FIELD_WEIGHT)
                
                adopt_rate = cfg.DIALECT_ADOPT_RATE
                gx, gy = int(self.pos[i].x // cfg.SPATIAL_GRID_CELL_SIZE), int(self.pos[i].y // cfg.SPATIAL_GRID_CELL_SIZE)
                
                # Count neighbors for various checks
                neighbors_count = 0
                if 0 <= gx < self.grid_count.shape[0] and 0 <= gy < self.grid_count.shape[1]:
                    density = float(self.grid_count[gx, gy]) / 64.0
                    adopt_rate *= (1.0 + density * 2.0)
                    neighbors_count = self.grid_count[gx, gy]

                alien_mismatch = 0.0
                if local_culture.norm() > 0.01:
                    alien_mismatch = 1.0 - ti.math.clamp(
                        self.dialect_state[i].dot(local_mean) / (self._safe_norm3(self.dialect_state[i]) * self._safe_norm3(local_mean)),
                        0.0,
                        1.0,
                    )
                m_alien = alien_mismatch * local_culture_strength
                velocity_scale = 1.0 / (1.0 + cfg.ALIEN_VISCOSITY_GAMMA * m_alien)
                self.vel[i] *= velocity_scale
                ti.atomic_add(self.alien_mismatch_accum[None], alien_mismatch)
                ti.atomic_add(self.viscosity_drag_accum[None], 1.0 - velocity_scale)
                
                # Apply FALSE_SIGNAL_PENALTY if signaling into the void
                if self.signal[i] > 0.5 and neighbors_count <= 1:
                    rewards[i] -= cfg.FALSE_SIGNAL_PENALTY

                mutation_rate = cfg.DIALECT_MUTATION_BASE
                if local_culture.norm() < 0.05: mutation_rate *= 3.0
                
                noise = ti.Vector([ti.random()-0.5, ti.random()-0.5, ti.random()-0.5]) * 0.1
                self.dialect_state[i] = ( (1.0 - adopt_rate - mutation_rate) * self.dialect_state[i] 
                                         + adopt_rate * local_mean 
                                         + mutation_rate * noise ).normalized()

                self.behavior_ctx[i][0] = self.signal[i]
                self.behavior_ctx[i][1] = ti.math.clamp(self.mimic_timer[i], 0.0, 1.0)
                self.behavior_ctx[i][2] = ti.math.clamp(self.age[i] / 10.0, 0.0, 1.0) 
                self.behavior_ctx[i][3] = ti.math.clamp(1.0 - self.energy[i]/100.0, 0.0, 1.0) 

                tax_mul = cfg.PREDATOR_TAX_MUL if self.type[i] == cfg.TYPE_PRED else 1.0
                density_tax = (float(alive_count)**2) * cfg.DENSITY_STRESS_FACTOR / 1000.0
                
                # Phase 4: Mimicry & Altruism logic
                mimic_action = ti.math.clamp(actions[i, 4], 0.0, 1.0)
                altruism_action = ti.math.clamp(actions[i, 5], 0.0, 1.0)
                
                self.mimic_timer[i] = ti.math.max(0.0, self.mimic_timer[i] - (1.0 / cfg.MIMIC_WINDOW_STEPS))
                self.altruism_timer[i] = ti.math.max(0.0, self.altruism_timer[i] - (1.0 / 18.0))
                
                if mimic_action > cfg.MIMIC_THRESHOLD:
                    ti.atomic_add(self.mimic_attempts[None], 1)
                    self.mimic_timer[i] = 1.0
                    adopt_rate_mimic = cfg.MIMIC_BLEND_STRENGTH
                    self.dialect_state[i] = ( (1.0 - adopt_rate_mimic) * self.dialect_state[i] + adopt_rate_mimic * local_mean ).normalized()
                    signal_cost = cfg.MIMIC_BASE_COST + cfg.MIMIC_SIGNAL_COST
                    self.energy[i] -= signal_cost
                    rewards[i] -= cfg.MIMIC_COST_BETA * signal_cost
                    ti.atomic_add(self.mimic_signal_cost_total[None], signal_cost)
                
                if altruism_action > cfg.ALTRUISM_THRESHOLD:
                    donor_loss = cfg.ALTRUISM_GIVE_AMOUNT * cfg.ALTRUISM_TAX_MUL
                    if self.energy[i] > donor_loss + cfg.ALTRUISM_RESCUE_THRESHOLD:
                        ti.atomic_add(self.altruism_events[None], 1)
                        self.altruism_timer[i] = 1.0 # 50 steps visual pulse window
                        self.energy[i] -= donor_loss
                        ti.atomic_add(self.altruism_donor_loss[None], donor_loss)

                        recipients = 0
                        for ox, oy in ti.static(ti.ndrange((-1, 2), (-1, 2))):
                            nx, ny = gx+ox, gy+oy
                            if 0 <= nx < self.grid_count.shape[0] and 0 <= ny < self.grid_count.shape[1]:
                                for k in range(self.grid_count[nx, ny]):
                                    other = self.grid_agents[nx, ny, k]
                                    if other != i and self.alive[other] and self.type[other] == self.type[i]:
                                        dist = (self.pos[i] - self.pos[other]).norm()
                                        if dist < cfg.INTERACTION_RADIUS and self.energy[other] < cfg.ALTRUISM_RESCUE_THRESHOLD:
                                            recipients += 1

                        effective_recipients = ti.min(recipients, cfg.ALTRUISM_MAX_RECIPIENTS)
                        ti.atomic_add(self.altruism_recipient_count[None], effective_recipients)
                        if effective_recipients > 0:
                            transfer_budget = donor_loss * cfg.ALTRUISM_TRANSFER_EFFICIENCY
                            transfer_each = transfer_budget / float(effective_recipients)
                            distributed = 0
                            kin_reward = 0.0
                            for ox, oy in ti.static(ti.ndrange((-1, 2), (-1, 2))):
                                nx, ny = gx+ox, gy+oy
                                if 0 <= nx < self.grid_count.shape[0] and 0 <= ny < self.grid_count.shape[1]:
                                    for k in range(self.grid_count[nx, ny]):
                                        other = self.grid_agents[nx, ny, k]
                                        if other != i and self.alive[other] and self.type[other] == self.type[i]:
                                            dist = (self.pos[i] - self.pos[other]).norm()
                                            if dist < cfg.INTERACTION_RADIUS and self.energy[other] < cfg.ALTRUISM_RESCUE_THRESHOLD and distributed < effective_recipients:
                                                self.energy[other] += transfer_each
                                                kin_reward += transfer_each * self._kin_similarity(i, other)
                                                distributed += 1
                            ti.atomic_add(self.altruism_transfer_amount[None], transfer_budget)
                            ti.atomic_add(self.altruism_kin_reward[None], kin_reward)
                            rewards[i] += kin_reward * cfg.ALTRUISM_REWARD_SCALE - donor_loss * cfg.ALTRUISM_SACRIFICE_COST
                        else:
                            rewards[i] -= cfg.SELF_SACRIFICE_OVERUSE_PENALTY * 0.5
                    else:
                        # Overuse penalty: trying to use altruism when energy is too low
                        rewards[i] -= cfg.SELF_SACRIFICE_OVERUSE_PENALTY

                energy_loss = (cfg.BASE_METABOLIC_RATE * tax_mul * (1.0 + metabolic_shift) + density_tax + self.vel[i].norm() * cfg.VELOCITY_ENERGY_TAX)
                self.energy[i] -= energy_loss
                rewards[i] -= energy_loss * 0.1 
                
                if self.type[i] == cfg.TYPE_PREY:
                    if fields.nutrients[ix, iy] > 0.5:
                        fields.nutrients[ix, iy] -= 0.5
                        self.energy[i] += cfg.PREY_FEED_ENERGY
                        rewards[i] += 2.0 # balanced
                        self.glow[i] = ti.math.clamp(self.glow[i] + 0.15, 0.0, 1.0)
                else: # Pred
                    for ox, oy in ti.static(ti.ndrange((-1, 2), (-1, 2))):
                        nx, ny = gx+ox, gy+oy
                        if 0 <= nx < self.grid_count.shape[0] and 0 <= ny < self.grid_count.shape[1]:
                            for k in range(self.grid_count[nx, ny]):
                                other = self.grid_agents[nx, ny, k]
                                if self.alive[other] and self.type[other] == cfg.TYPE_PREY:
                                    if (self.pos[other]-self.pos[i]).norm() < cfg.INTERACTION_RADIUS:
                                        self.alive[other] = 0 
                                        self.energy[i] += cfg.PRED_FEED_ENERGY
                                        rewards[i] += 15.0 # balanced
                                        if self.mimic_timer[i] > 0.0:
                                            ti.atomic_add(self.mimic_success[None], 1)
                                            rewards[i] += cfg.MIMIC_REWARD_ALPHA * cfg.PRED_FEED_ENERGY
                                        fields.carcasses[ix, iy] += 10.0 
                                        ti.atomic_add(fields.hazard[ix, iy], 0.8) 
                                        self.glow[i] = ti.math.clamp(self.glow[i] + 0.40, 0.0, 1.0)

                if self.energy[i] <= 0:
                    self.alive[i] = 0
                    rewards[i] -= 100.0 
                    ti.atomic_add(fields.hazard[ix, iy], 0.5)
                    if self.type[i] == cfg.TYPE_PRED:
                        ti.atomic_add(fields.nutrients[ix, iy], cfg.PRED_DEATH_NUTRIENT_BURST)
                
                for j in ti.static(range(cfg.ACTION_DIM)):
                    last_actions[i, j] = actions[i, j]

    @ti.kernel
    def compute_observations(self, fields: ti.template(), observations: ti.types.ndarray()):
        for i in range(cfg.MAX_AGENTS):
            if self.alive[i]:
                gx, gy = int(self.pos[i].x // cfg.SPATIAL_GRID_CELL_SIZE), int(self.pos[i].y // cfg.SPATIAL_GRID_CELL_SIZE)
                local_density = 0.0
                if 0 <= gx < self.grid_count.shape[0] and 0 <= gy < self.grid_count.shape[1]:
                    local_density = float(self.grid_count[gx, gy]) / 64.0

                field_x = int(self.pos[i].x * cfg.FIELD_RES[0]/cfg.WORLD_RES[0])
                field_y = int(self.pos[i].y * cfg.FIELD_RES[1]/cfg.WORLD_RES[1])
                local_signal_density = fields.active_signal[field_x, field_y]
                rho = local_density * cfg.VISION_AGENT_DENSITY_WEIGHT + local_signal_density * cfg.VISION_SIGNAL_DENSITY_WEIGHT

                vis_40 = self._visibility_decay(rho, 40.0)
                vis_50 = self._visibility_decay(rho, 50.0)
                vis_60 = self._visibility_decay(rho, 60.0)
                vis_100 = self._visibility_decay(rho, 100.0)
                ti.atomic_add(self.visibility_accum[None], (vis_40 + vis_50 + vis_60 + vis_100) * 0.25)
                ti.atomic_add(self.visibility_samples[None], 1)

                observations[i, 0] = self.sense(i, 0.5, 50.0, fields.nutrients) * vis_50
                observations[i, 1] = self.sense(i, 0.0, 60.0, fields.nutrients) * vis_60
                observations[i, 2] = self.sense(i, -0.5, 50.0, fields.nutrients) * vis_50
                observations[i, 3] = self.sense_pop(i, 0.6, 40.0) * vis_40
                observations[i, 4] = self.sense_pop(i, 0.0, 50.0) * vis_50
                observations[i, 5] = self.sense_pop(i, -0.6, 40.0) * vis_40
                observations[i, 6] = self.sense_signal(i, 0.0, 60.0) * vis_60
                observations[i, 7] = self.sense_signal(i, 0.0, 100.0) * vis_100
                observations[i, 8] = fields.hazard[field_x, field_y]
                observations[i, 9] = fields.thermal[field_x, field_y]
                
                cult_val = fields.culture[field_x, field_y]
                observations[i, 10] = cult_val[0]
                observations[i, 11] = cult_val[1]
                observations[i, 12] = cult_val[2]
                observations[i, 13] = 0.0 # Pad
                observations[i, 14] = 0.0 # Pad
                observations[i, 15] = 0.0 # Pad
                
                obs_idx = 16
                for j in ti.static(range(cfg.EPIGENETIC_DIM)):
                    observations[i, obs_idx+j] = self.epigenetic_ctx[i][j]
                
                obs_idx = 24
                for j in ti.static(range(cfg.BEHAVIOR_DIM)):
                    observations[i, obs_idx+j] = self.behavior_ctx[i][j]
                
                obs_idx = 28
                observations[i, obs_idx] = self.energy[i]/100.0
                observations[i, obs_idx+1] = float(self.type[i])
                observations[i, obs_idx+2] = self.vel[i].x
                observations[i, obs_idx+3] = self.vel[i].y
                observations[i, obs_idx+4] = self.angle[i] / 6.28
                observations[i, obs_idx+5] = self.signal[i]
                observations[i, obs_idx+6] = self.glow[i]
                observations[i, obs_idx+7] = self.age[i] / 10.0
            else:
                for j in range(cfg.OBSERVATION_DIM):
                    observations[i, j] = 0.0

    @ti.func
    def sense(self, idx, off, dist, field: ti.template()):
        p = self.pos[idx] + ti.Vector([ti.cos(self.angle[idx]+off), ti.sin(self.angle[idx]+off)]) * dist
        return field[int(p.x % cfg.WORLD_RES[0] * cfg.FIELD_RES[0]/cfg.WORLD_RES[0]), int(p.y % cfg.WORLD_RES[1] * cfg.FIELD_RES[1]/cfg.WORLD_RES[1])]

    @ti.func
    def sense_pop(self, idx, off, dist):
        ray = ti.Vector([ti.cos(self.angle[idx]+off), ti.sin(self.angle[idx]+off)])
        hit = 0.0
        gx, gy = int(self.pos[idx].x // cfg.SPATIAL_GRID_CELL_SIZE), int(self.pos[idx].y // cfg.SPATIAL_GRID_CELL_SIZE)
        for ox, oy in ti.static(ti.ndrange((-1, 2), (-1, 2))):
            nx, ny = gx+ox, gy+oy
            if 0 <= nx < self.grid_count.shape[0] and 0 <= ny < self.grid_count.shape[1]:
                for k in range(self.grid_count[nx, ny]):
                    other = self.grid_agents[nx, ny, k]
                    if other != idx and self.alive[other]:
                        diff = self.pos[other] - self.pos[idx]
                        if diff.norm() < dist and diff.dot(ray) > 0.8 * diff.norm(): hit += (1.0 - diff.norm()/dist)
        return ti.math.clamp(hit, 0.0, 1.0)

    @ti.func
    def sense_signal(self, idx, off, dist):
        total = 0.0
        gx, gy = int(self.pos[idx].x // cfg.SPATIAL_GRID_CELL_SIZE), int(self.pos[idx].y // cfg.SPATIAL_GRID_CELL_SIZE)
        for ox, oy in ti.static(ti.ndrange((-1, 2), (-1, 2))):
            nx, ny = gx+ox, gy+oy
            if 0 <= nx < self.grid_count.shape[0] and 0 <= ny < self.grid_count.shape[1]:
                for k in range(self.grid_count[nx, ny]):
                    other = self.grid_agents[nx, ny, k]
                    if other != idx and self.alive[other]:
                        d_val = (self.pos[other] - self.pos[idx]).norm()
                        if d_val < dist: total += self.signal[other] * (1.0 - d_val/dist)
        return ti.math.clamp(total, 0.0, 1.0)

    @ti.kernel
    def handle_reproduction(self, prey_count: int, pred_count: int):
        self.repro_counts.fill(0)
        for i in range(cfg.MAX_AGENTS):
            if self.alive[i]:
                repro_thresh = cfg.ASE_REPRO_THRESH if self.type[i] == cfg.TYPE_PREY else cfg.PRED_REPRO_THRESH
                if self.energy[i] > repro_thresh:
                    idx = ti.atomic_add(self.repro_counts[0], 1)
                    self.parents_buf[idx] = i
            else:
                idx = ti.atomic_add(self.repro_counts[2], 1)
                self.dead_buf[idx] = i

    def execute_reproduction(self, prey_count: int, pred_count: int):
        self.handle_reproduction(prey_count, pred_count)
        c = self.repro_counts.to_numpy()
        num_parents, num_dead = c[0], c[2]
        if num_dead == 0 or num_parents == 0: return
        spawn_count = min(num_parents, num_dead)
        self._spawn_kernel(self.parents_buf, self.dead_buf, spawn_count)

    @ti.kernel
    def _spawn_kernel(self, p_idx_field: ti.template(), dead_idx_field: ti.template(), count: int):
        for i in range(count):
            p_idx = p_idx_field[i]
            c_idx = dead_idx_field[i]
            self.alive[c_idx] = 1
            self.type[c_idx] = self.type[p_idx]
            self.pos[c_idx] = self.pos[p_idx] + ti.Vector([ti.random()-0.5, ti.random()-0.5])*5.0
            self.vel[c_idx] = ti.Vector([0.0, 0.0]) 
            self.energy[p_idx] *= 0.5
            self.energy[c_idx] = self.energy[p_idx]
            self.traits[c_idx] = self.traits[p_idx]
            self.dialect_state[c_idx] = self.dialect_state[p_idx]
            for j in ti.static(range(cfg.EPIGENETIC_DIM)):
                mutation = (ti.random() - 0.5) * 2.0 * cfg.EPIGENETIC_MUTATION_SIGMA
                self.epigenetic_ctx[c_idx][j] = ti.math.clamp(self.epigenetic_ctx[p_idx][j] * cfg.EPIGENETIC_INHERITANCE_STRENGTH + mutation, 0.0, 1.0)
            self.behavior_ctx[c_idx] = self.behavior_ctx[p_idx]

    @ti.kernel
    def get_alive_count(self) -> int:
        count = 0
        for i in range(cfg.MAX_AGENTS):
            if self.alive[i]: ti.atomic_add(count, 1)
        return count

    @ti.kernel
    def get_type_counts(self) -> ti.types.vector(2, ti.i32):
        prey, pred = 0, 0
        for i in range(cfg.MAX_AGENTS):
            if self.alive[i]:
                if self.type[i] == cfg.TYPE_PREY: ti.atomic_add(prey, 1)
                else: ti.atomic_add(pred, 1)
        return ti.Vector([prey, pred])
