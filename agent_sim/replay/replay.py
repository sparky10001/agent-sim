import json
import os
from datetime import datetime

LOG_DIR = os.path.expanduser("~/agent-sim/logs/runs")
os.makedirs(LOG_DIR, exist_ok=True)

class ReplayLogger:
    def __init__(self, run_name=None):
        self.run_name = run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.log_file = os.path.join(LOG_DIR, f"{self.run_name}.jsonl")
        self.episode_id = 0
        self.step_id = 0

        # Write metadata header if new file
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w") as f:
                meta = {
                    "type": "meta",
                    "run_name": self.run_name,
                    "timestamp": datetime.now().isoformat()
                }
                f.write(json.dumps(meta) + "\n")

    def new_episode(self):
        self.episode_id += 1
        self.step_id = 0

    def log_step(self, state, action, reward, done):
        entry = {
            "episode": self.episode_id,
            "step": self.step_id,
            "state": state,
            "action": action,
            "reward": reward,
            "done": done
        }
        self.step_id += 1

        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")