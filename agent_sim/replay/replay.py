import json
import os
from datetime import datetime
from threading import Lock

LOG_DIR = os.environ.get("LOG_DIR", "/app/logs/runs")
os.makedirs(LOG_DIR, exist_ok=True)


class ReplayLogger:
    def __init__(self, run_name=None, worker_id=None, session_id=None, max_in_memory_episodes=1000):
        self.run_name = run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.worker_id = worker_id
        self.session_id = session_id

        self.log_file = os.path.join(LOG_DIR, f"{self.run_name}.jsonl")

        self.episode_id = 0
        self.step_id = 0

        self.lock = Lock()

        # Memory control
        self.max_in_memory_episodes = max_in_memory_episodes
        self.episodes = []
        self.current_episode = []

        # Metadata header
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w") as f:
                meta = {
                    "type": "meta",
                    "run_name": self.run_name,
                    "worker_id": self.worker_id,
                    "session_id": self.session_id,
                    "timestamp": datetime.now().isoformat()
                }
                f.write(json.dumps(meta) + "\n")

    # -------------------- EPISODES --------------------

    def new_episode(self):
        self.episode_id += 1
        self.step_id = 0

        self.current_episode = []
        self.episodes.append(self.current_episode)

        # Prevent unbounded growth
        if len(self.episodes) > self.max_in_memory_episodes:
            self.episodes.pop(0)

    # -------------------- LOGGING --------------------

    def log_step(self, state, action, reward, done):
        entry = {
            "type": "step",
            "timestamp": datetime.now().isoformat(),
            "episode": self.episode_id,
            "step": self.step_id,
            "state": state,
            "action": action,
            "reward": reward,
            "done": done,
            "worker_id": self.worker_id,
            "session_id": self.session_id
        }

        # In-memory tracking
        self.current_episode.append(entry)

        self.step_id += 1

        # Thread-safe write
        with self.lock:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
                f.flush()

        # Auto finalize episode
        if done:
            self._finalize_episode()

    # -------------------- FINALIZATION --------------------

    def _finalize_episode(self):
        total_reward = sum(step["reward"] for step in self.current_episode)
        length = len(self.current_episode)

        summary = {
            "type": "episode_summary",
            "episode": self.episode_id,
            "total_reward": total_reward,
            "length": length,
            "timestamp": datetime.now().isoformat(),
            "worker_id": self.worker_id,
            "session_id": self.session_id
        }

        with self.lock:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(summary) + "\n")
                f.flush()