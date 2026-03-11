import os
import random
import sys
import time
from collections import deque

import dearpygui.dearpygui as dpg
import numpy as np
import torch


def set_determinism(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


set_determinism()

if getattr(sys, "frozen", False):
    project_root = os.path.dirname(os.path.abspath(sys.executable))
else:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
    if project_root not in sys.path:
        sys.path.append(project_root)

from primordial.core import config as cfg
from primordial.core.metrics import MetricsCore
from primordial.core.ring_buffer import ReplayRingBuffer
from primordial.core.world import PrimordialWorld
from primordial.training.async_worker import AsyncTrainingWorker
from primordial.training.policy import PrimordialPolicy


LENS_ITEMS = [
    "Bioluminescent",
    "Culture Dominance",
    "Communication",
    "Social Events",
]


class ObservatoryApp:
    def __init__(self):
        self.world = PrimordialWorld(headless=False)
        self.world.reset(seed=42)
        self.metrics_engine = MetricsCore()

        self.device = torch.device("cpu")
        self.checkpoint_path = os.path.join(project_root, "primordial_ppo_v17.pt")
        self.actor = PrimordialPolicy().to(self.device)
        self.policy_status = "Warm Start / Honest PPO-lite"
        self.load_checkpoint()

        self.ring_buffer = ReplayRingBuffer(
            capacity=4096,
            num_agents=cfg.MAX_AGENTS,
            obs_dim=cfg.BASE_OBSERVATION_DIM,
            act_dim=cfg.ACTION_DIM,
        )
        self.training_worker = AsyncTrainingWorker(self.actor, self.ring_buffer, self.device, batch_size=16)
        self.last_sync_log_step = -1

        self.is_running = False
        self.show_mode = 1
        self.cam_pos = np.array([cfg.WORLD_RES[0] / 2, cfg.WORLD_RES[1] / 2], dtype=np.float32)
        self.cam_zoom = 1.0
        self.last_interaction_time = time.perf_counter()
        self.hud_auto_hide = cfg.HUD_AUTO_HIDE
        self.hud_visible = True

        self.history_len = 600
        self.step_history = deque(maxlen=self.history_len)
        self.pop_history = deque(maxlen=self.history_len)
        self.reward_history = deque(maxlen=self.history_len)
        self.signal_history = deque(maxlen=self.history_len)
        self.log_messages = deque(maxlen=16)
        self.pixel_buffer = np.ones((cfg.WORLD_RES[1], cfg.WORLD_RES[0], 4), dtype=np.float32)

        self.setup_ui()

    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_path):
            try:
                state_dict = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
                self.actor.load_state_dict(state_dict)
                self.policy_status = "Checkpoint Loaded"
            except Exception as exc:
                self.policy_status = f"Checkpoint Error: {exc}"

    def save_checkpoint(self, sender=None, app_data=None, user_data=None):
        torch.save(self.actor.state_dict(), self.checkpoint_path)
        self.add_log(f"Checkpoint kaydedildi: {self.checkpoint_path}")

    def add_log(self, message: str):
        self.log_messages.appendleft(f"[{self.world.step_count:05d}] {message}")
        try:
            dpg.set_value("diagnostic_log", "\n".join(self.log_messages))
        except Exception:
            pass

    def setup_ui(self):
        dpg.create_context()
        with dpg.font_registry():
            font_path = "C:\\Windows\\Fonts\\arial.ttf"
            with dpg.font(font_path, 18) as default_font:
                dpg.add_font_range_hint(dpg.mvFontRangeHint_Default)
                dpg.add_font_chars([0x011E, 0x011F, 0x0130, 0x0131, 0x015E, 0x015F, 0x00C7, 0x00E7, 0x00D6, 0x00F6, 0x00DC, 0x00FC])
        dpg.bind_font(default_font)

        dpg.create_viewport(title="Primordial Observatory v17.2", width=1720, height=980)
        with dpg.texture_registry():
            dpg.add_raw_texture(cfg.WORLD_RES[0], cfg.WORLD_RES[1], self.pixel_buffer, format=dpg.mvFormat_Float_rgba, tag="main_viewport")

        with dpg.window(tag="Main", no_title_bar=True):
            with dpg.group(horizontal=True, tag="HUD_Header"):
                dpg.add_button(label="Play / Pause", callback=self.toggle_sim)
                dpg.add_button(label="Reset", callback=self.reset_sim)
                dpg.add_button(label="Save Checkpoint", callback=self.save_checkpoint)
                dpg.add_spacer(width=18)
                dpg.add_text("Lens")
                dpg.add_combo(LENS_ITEMS, default_value=LENS_ITEMS[1], width=190, callback=self.set_lens)
                dpg.add_spacer(width=18)
                dpg.add_text("Policy", color=(164, 205, 180))
                dpg.add_text(self.policy_status, tag="policy_status_txt", color=(214, 231, 220))
                dpg.add_spacer(width=18)
                dpg.add_text("FPS: 0", tag="fps_display", color=(227, 201, 130))
                dpg.add_text("TPS: 0", tag="tps_display", color=(113, 210, 176))

            dpg.add_separator()

            with dpg.group(horizontal=True):
                with dpg.child_window(width=1220, height=880, border=False):
                    dpg.add_image("main_viewport", width=1220, height=705)
                    with dpg.collapsing_header(label="Diagnostics", default_open=False):
                        dpg.add_text("", tag="diagnostic_log", wrap=1180)
                        with dpg.plot(height=145, width=-1):
                            dpg.add_plot_axis(dpg.mvXAxis, no_tick_labels=True, tag="trend_x_axis")
                            with dpg.plot_axis(dpg.mvYAxis, tag="trend_y_axis"):
                                dpg.add_line_series([], [], label="Population", tag="pop_total")
                                dpg.add_line_series([], [], label="Reward", tag="reward_series")
                                dpg.add_line_series([], [], label="Signal", tag="signal_series")
                            dpg.add_plot_legend()

                with dpg.child_window(width=420, height=880, border=True):
                    dpg.add_text("Core State", color=(152, 214, 191))
                    dpg.add_text("Adim: 0", tag="step_counter")
                    dpg.add_text("Alive: 0", tag="alive_txt")
                    dpg.add_text("Prey / Pred: 0 / 0", tag="species_txt")
                    dpg.add_text("Avg Energy: 0.00", tag="avg_energy_txt")
                    dpg.add_text("Signal Activity: 0.00", tag="signal_activity_txt")
                    dpg.add_text("Mimic Success: 0 / 0", tag="mimic_txt")
                    dpg.add_text("Altruism Rate: 0.00", tag="altruism_txt")
                    dpg.add_separator()
                    dpg.add_text("Training", color=(227, 201, 130))
                    dpg.add_text("Inference: Sync Policy Step", tag="inference_txt")
                    dpg.add_text("Training Loop: Async PPO-lite", tag="training_mode_txt")
                    dpg.add_text("Bridge: Taichi World -> CPU Replay", tag="bridge_txt")
                    dpg.add_text("Worker Step: 0", tag="worker_step_txt")
                    dpg.add_text("Optimizer Step: 0", tag="optimizer_step_txt")
                    dpg.add_text("Loss: 0.000", tag="loss_txt")
                    dpg.add_text("Policy / Value: 0.000 / 0.000", tag="policy_value_txt")
                    dpg.add_text("Entropy: 0.000", tag="entropy_txt")
                    dpg.add_text("Reward Trend: 0.000", tag="reward_txt")

        dpg.set_primary_window("Main", True)
        dpg.setup_dearpygui()
        dpg.show_viewport()

    def toggle_sim(self, sender=None, app_data=None, user_data=None):
        self.is_running = not self.is_running
        self.last_interaction_time = time.perf_counter()

    def reset_sim(self, sender=None, app_data=None, user_data=None):
        self.world.reset(seed=42)
        self.add_log("Ekosistem sifirlandi.")
        self.last_interaction_time = time.perf_counter()

    def set_lens(self, sender=None, app_data=None, user_data=None):
        choice = app_data if isinstance(app_data, str) else LENS_ITEMS[0]
        self.show_mode = LENS_ITEMS.index(choice)

    def handle_input(self):
        activity = False
        if dpg.is_key_down(dpg.mvKey_Up):
            self.cam_zoom *= 1.02
            activity = True
        if dpg.is_key_down(dpg.mvKey_Down):
            self.cam_zoom /= 1.02
            activity = True
        if dpg.is_mouse_button_down(dpg.mvMouseButton_Right):
            delta = dpg.get_mouse_drag_delta()
            self.cam_pos[0] -= (delta[0] * 0.1) / self.cam_zoom
            self.cam_pos[1] -= (delta[1] * 0.1) / self.cam_zoom
            activity = True
        if activity:
            self.last_interaction_time = time.perf_counter()

    def push_metrics(self, metrics):
        self.step_history.append(metrics["step"])
        self.pop_history.append(metrics["alive_count"])
        self.reward_history.append(float(self.training_worker.last_stats.get("reward_mean", 0.0)))
        self.signal_history.append(float(metrics.get("signal_activity", 0.0)))

        mimic_attempts = int(metrics.get("mimic_attempts", 0))
        mimic_success = int(metrics.get("mimic_success", round(metrics.get("mimic_success_rate", 0.0) * max(1, mimic_attempts))))
        altruism_events = float(metrics.get("altruism_events", 0))
        transfer_amount = float(metrics.get("altruism_transfer_amount", 0.0))
        transfer_rate = transfer_amount / max(1.0, altruism_events)

        dpg.set_value("step_counter", f"Adim: {metrics['step']}")
        dpg.set_value("alive_txt", f"Alive: {metrics['alive_count']}")
        dpg.set_value("species_txt", f"Prey / Pred: {metrics['prey_count']} / {metrics['pred_count']}")
        dpg.set_value("avg_energy_txt", f"Avg Energy: {metrics.get('avg_energy', 0.0):.2f}")
        dpg.set_value("signal_activity_txt", f"Signal Activity: {metrics.get('signal_activity', 0.0):.3f}")
        if mimic_attempts == 0 and mimic_success == 0:
            dpg.set_value("mimic_txt", "Mimic: Experimental / Inactive")
        else:
            dpg.set_value("mimic_txt", f"Mimic Success: {mimic_success} / {mimic_attempts}")

        if altruism_events <= 0 and transfer_amount <= 0.0:
            dpg.set_value("altruism_txt", "Altruism: Experimental / Inactive")
        else:
            dpg.set_value("altruism_txt", f"Altruism Rate: {transfer_rate:.2f}")

        stats = self.training_worker.last_stats
        dpg.set_value("inference_txt", "Inference: Sync Policy Step")
        dpg.set_value("training_mode_txt", "Training Loop: Async PPO-lite")
        dpg.set_value("bridge_txt", "Bridge: Taichi World -> CPU Replay")
        dpg.set_value("worker_step_txt", f"Worker Step: {self.training_worker.train_step}")
        dpg.set_value("optimizer_step_txt", f"Optimizer Step: {int(stats.get('optimizer_step', 0.0))}")
        dpg.set_value("loss_txt", f"Loss: {stats.get('loss', 0.0):.3f}")
        dpg.set_value("policy_value_txt", f"Policy / Value: {stats.get('policy_loss', 0.0):.3f} / {stats.get('value_loss', 0.0):.3f}")
        dpg.set_value("entropy_txt", f"Entropy: {stats.get('entropy', 0.0):.3f}")
        dpg.set_value("reward_txt", f"Reward Trend: {stats.get('reward_mean', 0.0):.3f}")

        dpg.set_value("pop_total", [list(self.step_history), list(self.pop_history)])
        dpg.set_value("reward_series", [list(self.step_history), list(self.reward_history)])
        dpg.set_value("signal_series", [list(self.step_history), list(self.signal_history)])
        dpg.fit_axis_data("trend_x_axis")
        dpg.fit_axis_data("trend_y_axis")

    def collect_rollout(self):
        obs_tensor = self.world.get_observations_torch(device=self.device)
        sample = self.actor.sample_actions(obs_tensor)
        self.world.step(sample.sampled_action)
        rollout = {
            "action_mean": sample.action_mean.cpu().numpy().astype(np.float32, copy=False),
            "log_prob": sample.log_prob.cpu().numpy().astype(np.float32, copy=False),
            "value": sample.value.cpu().numpy().astype(np.float32, copy=False),
        }
        self.ring_buffer.add(self.world.get_ring_buffer_data(rollout))

    def run(self):
        last_render_time = last_metrics_time = last_pacing_time = last_sim_time = time.perf_counter()
        render_interval = 1.0 / 60.0
        sim_interval = 1.0 / 18.0
        fps_counter = 0
        tps_counter = 0

        self.training_worker.start()
        self.add_log("Observatory online. Honest PPO-lite worker active.")

        try:
            while dpg.is_dearpygui_running():
                current_time = time.perf_counter()
                self.handle_input()

                if self.training_worker.sync_to_main() and self.training_worker.train_step - self.last_sync_log_step >= 20:
                    self.add_log("AI agirliklari ana politikaya senkronize edildi.")
                    self.last_sync_log_step = self.training_worker.train_step

                if self.is_running:
                    while current_time - last_sim_time >= sim_interval:
                        self.collect_rollout()
                        tps_counter += 1
                        last_sim_time += sim_interval

                    if current_time - last_metrics_time >= 0.5:
                        metrics = self.world.get_metrics()
                        self.metrics_engine.log(metrics)
                        self.push_metrics(metrics)
                        last_metrics_time = current_time
                        dpg.set_value(
                            "policy_status_txt",
                            f"Train {self.training_worker.train_step} | KL {self.training_worker.last_stats.get('approx_kl', 0.0):.4f}",
                        )

                if current_time - last_render_time >= render_interval:
                    raw_rgb = self.world.render(self.show_mode, float(self.cam_pos[0]), float(self.cam_pos[1]), float(self.cam_zoom))
                    if raw_rgb is not None:
                        self.pixel_buffer[:, :, :3] = raw_rgb.transpose(1, 0, 2)
                        dpg.set_value("main_viewport", self.pixel_buffer)
                    fps_counter += 1
                    last_render_time = current_time

                if current_time - last_pacing_time >= 1.0:
                    dpg.set_value("fps_display", f"FPS: {fps_counter}")
                    dpg.set_value("tps_display", f"TPS: {tps_counter}")
                    fps_counter = 0
                    tps_counter = 0
                    last_pacing_time = current_time

                dpg.render_dearpygui_frame()
        finally:
            self.training_worker.stop()
            self.metrics_engine.log({"step": "SHUTDOWN", "prey_count": 0, "pred_count": 0, "alive_count": 0})
            dpg.destroy_context()


if __name__ == "__main__":
    ObservatoryApp().run()
