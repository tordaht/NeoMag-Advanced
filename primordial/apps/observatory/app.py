import os
import random
import sys
import threading
import time
import ctypes
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
from primordial.core.behavior_event_logger import BehaviorEventLogger
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
    CHECKPOINT_IDLE_COLOR = (180, 205, 190)
    CHECKPOINT_BUSY_COLOR = (235, 195, 96)
    CHECKPOINT_OK_COLOR = (125, 216, 170)
    CHECKPOINT_ERROR_COLOR = (236, 114, 114)
    SIDEBAR_WIDTH = 360
    OUTER_PADDING = 8
    HEADER_HEIGHT = 44
    TREND_HEIGHT = 180
    RIGHT_TOP_MIN_HEIGHT = 360
    RIGHT_LOG_MIN_HEIGHT = 220

    def __init__(self):
        self.world = PrimordialWorld(headless=False, capture_behavior_events=True)
        self.world.reset(seed=42)
        self.metrics_engine = MetricsCore()
        self.event_logger = BehaviorEventLogger(filename=os.path.join(project_root, "behavior_events.csv"))

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
        self.actor_lock = threading.Lock()
        self.world_lock = threading.Lock()
        self.metrics_lock = threading.Lock()
        self.frame_lock = threading.Lock()
        self.render_state_lock = threading.Lock()
        self.pending_logs_lock = threading.Lock()
        self.pending_logs = deque()
        self.shutdown_event = threading.Event()
        self.sim_thread = None
        self.checkpoint_lock = threading.RLock()
        self.checkpoint_thread = None

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
        self.latest_metrics = self.world.get_metrics()
        self.metrics_version = 0
        self.latest_frame = None
        self.frame_version = 0
        self.render_request_pending = False
        self.render_request = (self.show_mode, float(self.cam_pos[0]), float(self.cam_pos[1]), float(self.cam_zoom))
        self.latest_tps = 0
        self.latest_fps = 0
        self.target_render_fps = 30.0
        self.target_ui_fps = 60.0
        self.metrics_interval = 0.5
        self.checkpoint_status_text = "Checkpoint: Hazir"
        self.checkpoint_status_color = self.CHECKPOINT_IDLE_COLOR
        self.checkpoint_busy = False
        self.viewport_size = (1720, 980)

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
        with self.checkpoint_lock:
            if self.checkpoint_busy:
                self.checkpoint_status_text = "Checkpoint: Kayit zaten suruyor..."
                self.checkpoint_status_color = self.CHECKPOINT_BUSY_COLOR
                self.queue_log("Checkpoint istegi geldi ama mevcut kayit hala suruyor.")
                return
            self.checkpoint_busy = True
            self.checkpoint_status_text = "Checkpoint: Kaydediliyor..."
            self.checkpoint_status_color = self.CHECKPOINT_BUSY_COLOR

        self.queue_log("Checkpoint kaydi baslatildi.")
        self.checkpoint_thread = threading.Thread(target=self._save_checkpoint_worker, daemon=True, name="CheckpointSaveWorker")
        self.checkpoint_thread.start()

    def _snapshot_actor_state(self):
        with self.actor_lock:
            return {
                key: value.detach().cpu().clone()
                for key, value in self.actor.state_dict().items()
            }

    def _save_checkpoint_worker(self):
        checkpoint_path = self.checkpoint_path
        temp_path = f"{checkpoint_path}.tmp"
        try:
            state_dict = self._snapshot_actor_state()
            torch.save(state_dict, temp_path)
            os.replace(temp_path, checkpoint_path)
            self.set_checkpoint_status("Checkpoint: Kaydedildi", self.CHECKPOINT_OK_COLOR)
            self.queue_log(f"Checkpoint kaydedildi: {checkpoint_path}")
        except Exception as exc:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except OSError:
                pass
            self.set_checkpoint_status(f"Checkpoint Hatasi: {exc}", self.CHECKPOINT_ERROR_COLOR)
            self.queue_log(f"Checkpoint kaydi basarisiz: {exc}")
        finally:
            with self.checkpoint_lock:
                self.checkpoint_busy = False

    def set_checkpoint_status(self, text: str, color):
        with self.checkpoint_lock:
            self.checkpoint_status_text = text
            self.checkpoint_status_color = color

    def refresh_checkpoint_status(self):
        with self.checkpoint_lock:
            text = self.checkpoint_status_text
            color = self.checkpoint_status_color
            busy = self.checkpoint_busy

        try:
            dpg.set_value("checkpoint_state_txt", text)
            dpg.configure_item("checkpoint_state_txt", color=color)
            dpg.configure_item("save_checkpoint_btn", enabled=not busy)
        except Exception:
            pass

    def queue_log(self, message: str):
        with self.pending_logs_lock:
            self.pending_logs.append(message)

    def flush_logs(self):
        with self.pending_logs_lock:
            pending = list(self.pending_logs)
            self.pending_logs.clear()

        if not pending:
            return

        current_step = int(self.get_latest_metrics().get("step", 0))
        for message in pending:
            self.log_messages.appendleft(f"[{current_step:05d}] {message}")

        try:
            dpg.set_value("diagnostic_log", "\n".join(self.log_messages))
        except Exception:
            pass

    def add_log(self, message: str):
        self.queue_log(message)
        self.flush_logs()

    def get_latest_metrics(self):
        with self.metrics_lock:
            return dict(self.latest_metrics)

    def update_latest_metrics(self, metrics):
        with self.metrics_lock:
            self.latest_metrics = dict(metrics)
            self.metrics_version += 1

    def get_metrics_snapshot(self):
        with self.metrics_lock:
            return self.metrics_version, dict(self.latest_metrics), int(self.latest_tps)

    def set_latest_tps(self, tps_value: int):
        with self.metrics_lock:
            self.latest_tps = int(tps_value)

    def request_render_frame(self):
        with self.render_state_lock:
            self.render_request = (self.show_mode, float(self.cam_pos[0]), float(self.cam_pos[1]), float(self.cam_zoom))
            self.render_request_pending = True

    def get_pending_render_request(self):
        with self.render_state_lock:
            if not self.render_request_pending:
                return None
            self.render_request_pending = False
            return self.render_request

    def publish_frame(self, raw_rgb):
        if raw_rgb is None:
            return
        frame = raw_rgb.transpose(1, 0, 2).astype(np.float32, copy=False)
        with self.frame_lock:
            self.latest_frame = np.ascontiguousarray(frame)
            self.frame_version += 1

    def consume_latest_frame(self):
        with self.frame_lock:
            if self.latest_frame is None:
                return self.frame_version, None
            return self.frame_version, np.copy(self.latest_frame)

    def setup_ui(self):
        dpg.create_context()
        with dpg.font_registry():
            font_path = "C:\\Windows\\Fonts\\arial.ttf"
            with dpg.font(font_path, 18) as default_font:
                dpg.add_font_range_hint(dpg.mvFontRangeHint_Default)
                dpg.add_font_chars([0x011E, 0x011F, 0x0130, 0x0131, 0x015E, 0x015F, 0x00C7, 0x00E7, 0x00D6, 0x00F6, 0x00DC, 0x00FC])
        dpg.bind_font(default_font)

        screen_width, screen_height = self.get_screen_size()
        self.viewport_size = (screen_width, screen_height)
        dpg.create_viewport(
            title="Primordial Observatory v17.2",
            width=screen_width,
            height=screen_height,
            x_pos=0,
            y_pos=0,
            resizable=True,
        )
        with dpg.texture_registry():
            dpg.add_raw_texture(cfg.WORLD_RES[0], cfg.WORLD_RES[1], self.pixel_buffer, format=dpg.mvFormat_Float_rgba, tag="main_viewport")

        with dpg.window(tag="Main", no_title_bar=True, no_move=True, no_resize=True):
            with dpg.group(horizontal=True, tag="HUD_Header"):
                dpg.add_button(label="Play / Pause", callback=self.toggle_sim)
                dpg.add_button(label="Reset", callback=self.reset_sim)
                dpg.add_button(label="Save Checkpoint", callback=self.save_checkpoint, tag="save_checkpoint_btn")
                dpg.add_text(self.checkpoint_status_text, tag="checkpoint_state_txt", color=self.checkpoint_status_color)
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

            with dpg.group(horizontal=True, tag="content_row"):
                with dpg.child_window(width=1220, height=880, border=False, tag="left_panel"):
                    dpg.add_image("main_viewport", width=1220, height=705, tag="viewport_image")
                    with dpg.child_window(height=170, border=True, tag="trend_panel"):
                        with dpg.plot(height=-1, width=-1, tag="trend_plot"):
                            dpg.add_plot_axis(dpg.mvXAxis, no_tick_labels=True, tag="trend_x_axis")
                            with dpg.plot_axis(dpg.mvYAxis, tag="trend_y_axis"):
                                dpg.add_line_series([], [], label="Population", tag="pop_total")
                                dpg.add_line_series([], [], label="Reward", tag="reward_series")
                                dpg.add_line_series([], [], label="Signal", tag="signal_series")
                            dpg.add_plot_legend()

                with dpg.child_window(width=self.SIDEBAR_WIDTH, height=880, border=False, tag="right_panel"):
                    with dpg.child_window(width=-1, height=540, border=True, tag="core_panel"):
                        dpg.add_text("Core State", color=(152, 214, 191))
                        dpg.add_text("Adim: 0", tag="step_counter")
                        dpg.add_text("Alive: 0", tag="alive_txt")
                        dpg.add_text("Prey / Pred: 0 / 0", tag="species_txt")
                        dpg.add_text("Avg Energy: 0.00", tag="avg_energy_txt")
                        dpg.add_text("Signal Activity: 0.00", tag="signal_activity_txt")
                        dpg.add_text("Mimic Success: 0 / 0", tag="mimic_txt")
                        dpg.add_text("Altruism Rate: 0.00", tag="altruism_txt")
                        dpg.add_text("Territorial Pressure: 0.000", tag="territorial_pressure_txt")
                        dpg.add_text("Culture Drag: 0.000", tag="culture_drag_txt")
                        dpg.add_text("Signal Anomaly: 0.000", tag="signal_anomaly_txt")
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

                    with dpg.child_window(width=-1, height=300, border=True, tag="diagnostics_panel"):
                        dpg.add_text("Diagnostics", color=(214, 214, 214))
                        dpg.add_text("", tag="diagnostic_log", wrap=340)

        dpg.set_primary_window("Main", True)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        self.maximize_viewport()
        self.apply_responsive_layout(force=True)
        self.refresh_checkpoint_status()

    def get_screen_size(self):
        try:
            user32 = ctypes.windll.user32
            return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
        except Exception:
            return 1720, 980

    def maximize_viewport(self):
        try:
            if hasattr(dpg, "maximize_viewport"):
                dpg.maximize_viewport()
        except Exception:
            pass

    def get_viewport_size(self):
        width = self.viewport_size[0]
        height = self.viewport_size[1]
        try:
            width = int(dpg.get_viewport_client_width())
            height = int(dpg.get_viewport_client_height())
        except Exception:
            pass
        return max(width, 1280), max(height, 720)

    def apply_responsive_layout(self, force=False):
        width, height = self.get_viewport_size()
        if not force and (width, height) == self.viewport_size:
            return

        self.viewport_size = (width, height)
        content_height = max(480, height - self.HEADER_HEIGHT - self.OUTER_PADDING * 3)
        right_width = min(max(self.SIDEBAR_WIDTH, int(width * 0.19)), 420)
        left_width = max(700, width - right_width - self.OUTER_PADDING * 4)
        image_height = max(360, content_height - self.TREND_HEIGHT - self.OUTER_PADDING)
        right_top_height = max(self.RIGHT_TOP_MIN_HEIGHT, int(content_height * 0.64))
        right_log_height = max(self.RIGHT_LOG_MIN_HEIGHT, content_height - right_top_height - self.OUTER_PADDING)

        try:
            dpg.configure_item("Main", width=width, height=height, pos=(0, 0))
            dpg.configure_item("left_panel", width=left_width, height=content_height)
            dpg.configure_item("right_panel", width=right_width, height=content_height)
            dpg.configure_item("viewport_image", width=left_width, height=image_height)
            dpg.configure_item("trend_panel", width=left_width, height=self.TREND_HEIGHT)
            dpg.configure_item("trend_plot", width=-1, height=self.TREND_HEIGHT - 12)
            dpg.configure_item("core_panel", width=right_width, height=right_top_height)
            dpg.configure_item("diagnostics_panel", width=right_width, height=right_log_height)
            dpg.configure_item("diagnostic_log", wrap=max(220, right_width - 24))
        except Exception:
            pass

    def toggle_sim(self, sender=None, app_data=None, user_data=None):
        self.is_running = not self.is_running
        self.last_interaction_time = time.perf_counter()
        self.add_log("Simulasyon devam ediyor." if self.is_running else "Simulasyon duraklatildi.")

    def reset_sim(self, sender=None, app_data=None, user_data=None):
        self.is_running = False
        with self.world_lock:
            self.world.reset(seed=42)
            self.update_latest_metrics(self.world.get_metrics())
        self.step_history.clear()
        self.pop_history.clear()
        self.reward_history.clear()
        self.signal_history.clear()
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
        dpg.set_value("territorial_pressure_txt", f"Territorial Pressure: {metrics.get('avg_territorial_pressure', 0.0):.3f}")
        dpg.set_value("culture_drag_txt", f"Culture Drag: {metrics.get('avg_culture_drag', 0.0):.3f}")
        dpg.set_value("signal_anomaly_txt", f"Signal Anomaly: {metrics.get('avg_signal_anomaly', 0.0):.3f}")
        if mimic_attempts == 0 and mimic_success == 0:
            dpg.set_value("mimic_txt", "Mimic: behavior hook present, live activity not yet observed")
        else:
            dpg.set_value("mimic_txt", f"Mimic Success: {mimic_success} / {mimic_attempts}")

        if altruism_events <= 0 and transfer_amount <= 0.0:
            dpg.set_value("altruism_txt", "Altruism: behavior hook present, live activity not yet observed")
        else:
            dpg.set_value("altruism_txt", f"Altruism Rate: {transfer_rate:.2f}")

        stats = self.training_worker.last_stats
        dpg.set_value("inference_txt", "Inference: Background Sim Thread")
        dpg.set_value("training_mode_txt", "Training Loop: Async PPO-lite Worker")
        dpg.set_value("bridge_txt", "Bridge: Taichi -> CPU Replay | Render: Throttled Snapshot")
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
        with self.actor_lock:
            sample = self.actor.sample_actions(obs_tensor)
        self.world.step(sample.sampled_action)
        rollout = {
            "action_mean": sample.action_mean.cpu().numpy().astype(np.float32, copy=False),
            "log_prob": sample.log_prob.cpu().numpy().astype(np.float32, copy=False),
            "value": sample.value.cpu().numpy().astype(np.float32, copy=False),
        }
        self.ring_buffer.add(self.world.get_ring_buffer_data(rollout))

    def simulation_loop(self):
        last_metrics_time = time.perf_counter()
        last_tps_time = time.perf_counter()
        tps_counter = 0

        while not self.shutdown_event.is_set():
            if not self.is_running:
                now = time.perf_counter()
                if now - last_tps_time >= 1.0:
                    self.set_latest_tps(0)
                    last_tps_time = now
                    tps_counter = 0
                time.sleep(0.005)
                continue

            metrics_snapshot = None
            with self.world_lock:
                self.collect_rollout()
                self.event_logger.log_events(self.world.get_behavior_events())
                tps_counter += 1
                now = time.perf_counter()
                if now - last_metrics_time >= self.metrics_interval:
                    metrics_snapshot = self.world.get_metrics()
                    self.metrics_engine.log(metrics_snapshot)
                    last_metrics_time = now

            if metrics_snapshot is not None:
                self.update_latest_metrics(metrics_snapshot)

            render_request = self.get_pending_render_request()
            if render_request is not None:
                with self.world_lock:
                    raw_rgb = self.world.render(*render_request)
                self.publish_frame(raw_rgb)

            now = time.perf_counter()
            if now - last_tps_time >= 1.0:
                elapsed = max(now - last_tps_time, 1e-6)
                self.set_latest_tps(int(round(tps_counter / elapsed)))
                tps_counter = 0
                last_tps_time = now

    def start_simulation_thread(self):
        if self.sim_thread and self.sim_thread.is_alive():
            return
        self.sim_thread = threading.Thread(target=self.simulation_loop, daemon=True, name="ObservatorySimulation")
        self.sim_thread.start()

    def run(self):
        last_render_time = last_metrics_push_time = last_pacing_time = time.perf_counter()
        render_interval = 1.0 / self.target_render_fps
        ui_frame_interval = 1.0 / self.target_ui_fps
        fps_counter = 0
        last_metrics_version = -1
        last_frame_version = -1

        self.training_worker.start()
        self.start_simulation_thread()
        self.add_log("Observatory online. Honest PPO-lite worker active.")

        try:
            while dpg.is_dearpygui_running():
                frame_start = time.perf_counter()
                current_time = frame_start
                self.handle_input()
                self.flush_logs()
                self.refresh_checkpoint_status()
                self.apply_responsive_layout()

                with self.actor_lock:
                    did_sync = self.training_worker.sync_to_main()
                if did_sync and self.training_worker.train_step - self.last_sync_log_step >= 20:
                    self.add_log("AI agirliklari ana politikaya senkronize edildi.")
                    self.last_sync_log_step = self.training_worker.train_step

                metrics_version, metrics, latest_tps = self.get_metrics_snapshot()
                if metrics_version != last_metrics_version or current_time - last_metrics_push_time >= self.metrics_interval:
                    self.push_metrics(metrics)
                    last_metrics_version = metrics_version
                    last_metrics_push_time = current_time
                    dpg.set_value(
                        "policy_status_txt",
                        f"Train {self.training_worker.train_step} | KL {self.training_worker.last_stats.get('approx_kl', 0.0):.4f}",
                    )

                if current_time - last_render_time >= render_interval:
                    self.request_render_frame()
                    last_render_time = current_time

                frame_version, frame = self.consume_latest_frame()
                if frame is not None and frame_version != last_frame_version:
                    self.pixel_buffer[:, :, :3] = frame
                    dpg.set_value("main_viewport", self.pixel_buffer)
                    last_frame_version = frame_version

                dpg.render_dearpygui_frame()
                fps_counter += 1

                if current_time - last_pacing_time >= 1.0:
                    self.latest_fps = fps_counter
                    dpg.set_value("fps_display", f"FPS: {self.latest_fps}")
                    dpg.set_value("tps_display", f"TPS: {latest_tps}")
                    fps_counter = 0
                    last_pacing_time = current_time

                elapsed = time.perf_counter() - frame_start
                if elapsed < ui_frame_interval:
                    time.sleep(ui_frame_interval - elapsed)
        finally:
            self.is_running = False
            self.shutdown_event.set()
            if self.sim_thread:
                self.sim_thread.join(timeout=2.0)
            if self.checkpoint_thread:
                self.checkpoint_thread.join(timeout=2.0)
            self.training_worker.stop()
            self.metrics_engine.log({"step": "SHUTDOWN", "prey_count": 0, "pred_count": 0, "alive_count": 0})
            dpg.destroy_context()


if __name__ == "__main__":
    ObservatoryApp().run()
