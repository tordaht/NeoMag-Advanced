import taichi as ti
from . import config as cfg

@ti.data_oriented
class CoreRenderer:
    """
    v17.1.1-CULTURE-EXT Renderer.
    Visualizes 3-channel Cultural Markers and Agent Dialects.
    Maintains Mineral Noir aesthetics with tribal bioluminescence.
    """
    def __init__(self):
        self.display_buffer = ti.Vector.field(3, dtype=ti.f32, shape=cfg.WORLD_RES)
        self.bg_color = ti.Vector([cfg.VIEWPORT_BG[0], cfg.VIEWPORT_BG[1], cfg.VIEWPORT_BG[2]])

    @ti.func
    def _smooth01(self, value: float) -> float:
        x = ti.math.clamp(value, 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)

    @ti.func
    def _pulse(self, phase: float) -> float:
        return self._smooth01(0.5 + 0.5 * ti.sin(phase))

    @ti.func
    def _dominant_color(self, culture):
        v0, v1, v2 = culture[0], culture[1], culture[2]
        max_v = ti.max(v0, ti.max(v1, v2))
        dom_col = ti.Vector([0.0, 0.0, 0.0])
        if max_v == v0:
            dom_col = ti.Vector(cfg.CULTURE_PALETTE[0])
        elif max_v == v1:
            dom_col = ti.Vector(cfg.CULTURE_PALETTE[1])
        else:
            dom_col = ti.Vector(cfg.CULTURE_PALETTE[2])
        return dom_col

    @ti.func
    def _culture_depth(self, culture):
        sum_v = culture[0] + culture[1] + culture[2]
        overlap = 0.0
        if culture[0] > 0.12:
            overlap += 1.0
        if culture[1] > 0.12:
            overlap += 1.0
        if culture[2] > 0.12:
            overlap += 1.0
        overlap = ti.math.clamp((overlap - 1.0) * 0.5, 0.0, 1.0)
        density = ti.math.tanh(sum_v * 0.12)
        return ti.math.clamp(density * (0.7 + 0.6 * overlap), 0.0, 1.0)

    @ti.kernel
    def render(self, fields: ti.template(), organisms: ti.template(), show_mode: int, cam_x: float, cam_y: float, zoom: float, current_time: float):
        # 1. Background Layer (Cultural Memory Integration)
        breath = 0.82 + 0.18 * self._pulse(current_time * 6.28 * (cfg.RESTING_PULSE_BPM / 60.0))
        alpha_glow = 0.92 + 0.08 * self._pulse(current_time * 6.28 * cfg.ALPHA_GLOW_HZ)
        
        for i, j in self.display_buffer:
            wx = (i - cfg.WORLD_RES[0]/2) / zoom + cam_x
            wy = (j - cfg.WORLD_RES[1]/2) / zoom + cam_y
            ix = int(wx % cfg.WORLD_RES[0] * cfg.FIELD_RES[0] / cfg.WORLD_RES[0])
            iy = int(wy % cfg.WORLD_RES[1] * cfg.FIELD_RES[1] / cfg.WORLD_RES[1])
            
            color = self.bg_color
            cult = fields.culture[ix, iy]
            dom_col = self._dominant_color(cult)
            depth = self._culture_depth(cult)
            matte_shadow = self.bg_color * (1.0 - depth * cfg.TERRITORIAL_DEPTH_STRENGTH)
            matte_tint = dom_col * depth * cfg.TERRITORIAL_MATTE_STRENGTH
            
            if show_mode == 0:
                # Visible: Cultural Haze + Quorum activity
                color = matte_shadow + matte_tint
                color += ti.Vector(cfg.CULTURE_PALETTE[0]) * ti.math.tanh(cult[0] * 0.08)
                color += ti.Vector(cfg.CULTURE_PALETTE[1]) * ti.math.tanh(cult[1] * 0.08)
                color += ti.Vector(cfg.CULTURE_PALETTE[2]) * ti.math.tanh(cult[2] * 0.08)
                q_prey = fields.prey_quorum[ix, iy]
                q_pred = fields.pred_quorum[ix, iy]
                sig = fields.active_signal[ix, iy]
                color += ti.Vector(cfg.COLOR_PREY_QUORUM) * ti.math.tanh(q_prey * 0.06)
                color += ti.Vector(cfg.COLOR_PRED_QUORUM) * ti.math.tanh(q_pred * 0.05)
                color += ti.Vector(cfg.COLOR_SIGNAL) * ti.math.tanh(sig * 0.7) * 0.16 * cfg.FOREGROUND_GLASS_GAIN * alpha_glow
            elif show_mode == 1:
                # Culture Dominance Lens: Hue (Dominant), Saturation (Confidence), Brightness (Density)
                v0, v1, v2 = cult[0], cult[1], cult[2]
                max_v = ti.max(v0, ti.max(v1, v2))
                sum_v = v0 + v1 + v2
                if max_v > 0.001:
                    second_max = v0 + v1 + v2 - max_v - ti.min(v0, ti.min(v1, v2))
                    confidence = (max_v - second_max) / max_v
                    brightness = ti.math.tanh(sum_v * 1.5)
                    color = ti.math.mix(matte_shadow, dom_col, confidence) * brightness * breath
                    color += dom_col * depth * 0.08 * alpha_glow
            elif show_mode == 2:
                # Communication Lens: Prey quorum, Pred quorum, and active signal
                q_prey = fields.prey_quorum[ix, iy]
                q_pred = fields.pred_quorum[ix, iy]
                sig = fields.active_signal[ix, iy]
                color = matte_shadow * 0.9 + matte_tint * 0.3
                color += ti.Vector(cfg.COLOR_PREY_QUORUM) * ti.math.tanh(q_prey * 2.1) * (0.85 + 0.15 * breath)
                color += ti.Vector(cfg.COLOR_PRED_QUORUM) * ti.math.tanh(q_pred * 2.1) * (0.85 + 0.15 * breath)
                color += ti.Vector(cfg.COLOR_SIGNAL) * ti.math.tanh(sig * 2.6) * cfg.FOREGROUND_GLASS_GAIN * alpha_glow
            elif show_mode == 3:
                # Social Events Lens: Minimal bg, highlight social events
                color = matte_shadow * 0.75 + matte_tint * 0.18
            
            self.display_buffer[i, j] = color

        # 2. Specimen Layer (Neural Cells + Cultural Identity)
        for i in range(cfg.MAX_AGENTS):
            if organisms.alive[i]:
                sx = (organisms.pos[i].x - cam_x) * zoom + cfg.WORLD_RES[0] / 2
                sy = (organisms.pos[i].y - cam_y) * zoom + cfg.WORLD_RES[1] / 2
                
                if -60 <= sx < cfg.WORLD_RES[0] + 60 and -60 <= sy < cfg.WORLD_RES[1] + 60:
                    energy = organisms.energy[i]
                    e_norm = ti.math.clamp(energy / 100.0, 0.0, 1.0)
                    sig = organisms.signal[i]
                    glow_val = organisms.glow[i]
                    traits = organisms.traits[i]
                    dialect = organisms.dialect_state[i]
                    
                    agent_col = ti.Vector(cfg.CULTURE_PALETTE[0]) * traits[0] + \
                                ti.Vector(cfg.CULTURE_PALETTE[1]) * traits[1] + \
                                ti.Vector(cfg.CULTURE_PALETTE[2]) * traits[2]
                    
                    agent_col = ti.math.mix(agent_col, dialect, 0.2)
                    
                    mimic_timer_val = organisms.mimic_timer[i]
                    altruism_timer_val = organisms.altruism_timer[i]
                    
                    is_mimic = 1.0 if mimic_timer_val > 0.0 else 0.0
                    is_altru = 1.0 if altruism_timer_val > 0.0 else 0.0
                    event_phase = current_time * 6.28 * (cfg.RESTING_PULSE_BPM / 60.0) + float(i) * 0.17
                    event_bloom = self._pulse(event_phase)
                    alpha_bloom = self._pulse(current_time * 6.28 * cfg.ALPHA_GLOW_HZ + float(i) * 0.11)
                    mimic_glow = is_mimic * (0.45 + 0.55 * event_bloom)
                    altru_glow = is_altru * (0.40 + 0.60 * event_bloom)
                    
                    if organisms.type[i] == cfg.TYPE_PRED:
                        agent_col = ti.math.mix(agent_col, ti.Vector(cfg.COLOR_PRED_QUORUM), 0.55)
                    else:
                        agent_col = ti.math.mix(agent_col, ti.Vector(cfg.COLOR_PREY_QUORUM), 0.18)
                    
                    if is_mimic > 0.5:
                        agent_col = ti.math.mix(agent_col, ti.Vector(cfg.COLOR_OVERLAP), 0.75 * mimic_timer_val)
                    if is_altru > 0.5:
                        agent_col = ti.math.mix(agent_col, ti.Vector(cfg.COLOR_SIGNAL), 0.35 * altruism_timer_val)
                    
                    agent_col *= (0.68 + 0.28 * breath + glow_val * 0.35 + mimic_glow * 0.28 + altru_glow * 0.24)
                    
                    r_body = (cfg.AGENT_RADIUS + e_norm * 1.5) * zoom
                    aura_radius = r_body + (6.0 * glow_val * zoom) + (5.0 * mimic_glow + 4.0 * altru_glow) * zoom
                    
                    wave_progress = self._pulse(current_time * 6.28 * cfg.SIGNAL_WAVE_SPEED + float(i) * 0.1)
                    ripple_radius = (cfg.AGENT_RADIUS * 2.5 + (sig + 0.2) * (10.0 + 26.0 * wave_progress)) * zoom
                    
                    altru_pulse_rad = 0.0
                    if is_altru > 0.5:
                        pulse_prog = self._smooth01(1.0 - altruism_timer_val)
                        altru_pulse_rad = r_body + cfg.INTERACTION_RADIUS * zoom * pulse_prog

                    scan_r = int(ti.max(ti.max(ti.max(r_body, ripple_radius), aura_radius), altru_pulse_rad) + 1.5)
                    
                    # Layering visual modes
                    render_aura = 1
                    render_body = 1
                    render_ripple = 1
                    render_altru = 1
                    
                    if show_mode == 1: # Culture Dominance
                        render_body = 1
                        render_aura = 0
                        render_ripple = 0
                        render_altru = 0
                    elif show_mode == 2: # Communication Lens
                        render_body = 1
                        render_aura = 0
                        render_ripple = 1 # Active signal pulse
                        render_altru = 0
                        if organisms.type[i] == cfg.TYPE_PRED:
                            agent_col = ti.Vector(cfg.COLOR_PRED_QUORUM)
                        else:
                            agent_col = ti.Vector(cfg.COLOR_PREY_QUORUM)
                    elif show_mode == 3: # Social Events Lens
                        render_body = 0
                        render_aura = 0
                        render_ripple = 0
                        render_altru = 1
                        if is_mimic > 0.5:
                            render_body = 1
                            agent_col = ti.Vector(cfg.COLOR_OVERLAP) * 1.6 * mimic_timer_val

                    for ox, oy in ti.ndrange((-scan_r, scan_r + 1), (-scan_r, scan_r + 1)):
                        px, py = int(sx + ox), int(sy + oy)
                        if 0 <= px < cfg.WORLD_RES[0] and 0 <= py < cfg.WORLD_RES[1]:
                            dist = ti.sqrt(float(ox*ox + oy*oy))
                            
                            if render_aura == 1 and dist < aura_radius:
                                aura_falloff = ti.math.clamp(1.0 - dist/aura_radius, 0.0, 1.0)
                                aura_mul = 0.12 + glow_val * 0.05
                                aura_mul += mimic_glow * 0.18 * cfg.BIO_GLOW_INTENSITY
                                aura_mul += altru_glow * 0.15 * cfg.BIO_GLOW_INTENSITY
                                aura_shape = ti.pow(aura_falloff, cfg.BIO_GLOW_SOFTNESS)
                                self.display_buffer[px, py] += agent_col * aura_shape * aura_mul * breath * (0.92 + 0.08 * alpha_bloom)

                            if render_altru == 1 and is_altru > 0.5 and ti.abs(dist - altru_pulse_rad) < 2.4 * zoom:
                                pulse_col = ti.Vector([1.0, 0.88, 0.10])
                                pulse_strength = (1.0 - ti.abs(dist - altru_pulse_rad) / (2.4 * zoom)) * altru_glow * cfg.BIO_GLOW_INTENSITY
                                self.display_buffer[px, py] += pulse_col * pulse_strength * breath

                            if render_body == 1 and dist < r_body:
                                opacity = ti.math.clamp(0.62 + e_norm * 0.28 + mimic_glow * 0.06 + altru_glow * 0.05, 0.62, 0.96)
                                body_fill = ti.exp(-2.5 * (dist / r_body)**2)
                                pixel_col = agent_col * (0.82 + 0.24 * body_fill) * cfg.FOREGROUND_GLASS_GAIN
                                self.display_buffer[px, py] = self.display_buffer[px, py] * (1.0 - opacity * 0.34) + pixel_col * opacity * 0.56
                            
                            if render_ripple == 1 and ripple_radius > r_body:
                                ripple_dist = ti.abs(dist - ripple_radius)
                                if ripple_dist < 1.8 * zoom:
                                    ripple_fade = (0.35 + 0.65 * wave_progress) * 0.10 * breath
                                    if show_mode == 2:
                                        ripple_fade *= 4.4
                                    self.display_buffer[px, py] += agent_col * ripple_fade * (1.0 - ripple_dist/(1.8*zoom))
