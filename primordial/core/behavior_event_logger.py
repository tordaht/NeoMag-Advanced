import csv
import os
import time


class BehaviorEventLogger:
    def __init__(self, filename="behavior_events.csv", session_id=None):
        self.filename = filename
        self.session_id = session_id or time.strftime("%Y%m%d_%H%M%S")
        self.event_index = 0
        self.headers = [
            "session_id",
            "event_index",
            "step",
            "agent_id",
            "tribe",
            "event_type",
            "target_agent_id",
            "target_tribe",
            "local_context",
            "reward_delta",
            "energy_delta",
            "success",
        ]

        should_init = True
        if os.path.exists(self.filename):
            try:
                with open(self.filename, "r", encoding="utf-8") as handle:
                    if handle.readline().strip() == ",".join(self.headers):
                        should_init = False
            except Exception:
                pass

        if should_init:
            with open(self.filename, "w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerow(self.headers)

    def log_events(self, events):
        if not events:
            return

        with open(self.filename, "a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            for event in events:
                self.event_index += 1
                writer.writerow(
                    [
                        self.session_id,
                        self.event_index,
                        event.get("step", 0),
                        event.get("agent_id", ""),
                        event.get("tribe", ""),
                        event.get("event_type", ""),
                        event.get("target_agent_id", ""),
                        event.get("target_tribe", ""),
                        event.get("local_context", ""),
                        event.get("reward_delta", 0.0),
                        event.get("energy_delta", 0.0),
                        int(bool(event.get("success", False))),
                    ]
                )
