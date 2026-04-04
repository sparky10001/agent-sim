import os
import json
import socket
from datetime import datetime
from pathlib import Path
from threading import Lock

# -------------------- CONFIG --------------------

DEFAULT_RUNS_DIR = "./runs"
RUNS_DIR = os.environ.get("RUNS_DIR", DEFAULT_RUNS_DIR)

# Ensure base runs dir exists
try:
    Path(RUNS_DIR).mkdir(parents=True, exist_ok=True)
except Exception:
    RUNS_DIR = DEFAULT_RUNS_DIR
    Path(RUNS_DIR).mkdir(parents=True, exist_ok=True)


class RunManifest:
    def __init__(self, config: dict):
        self._lock = Lock()

        self.run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.base_path = Path(RUNS_DIR) / self.run_id
        self.base_path.mkdir(parents=True, exist_ok=True)

        self.manifest_path = self.base_path / "manifest.json"

        self.data = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "config": config,
            "results": {}
        }

        self._write()

    def update_results(self, results: dict, phase: str = "final"):
        if "results" not in self.data:
            self.data["results"] = {}

        if phase in self.data["results"]:
            print(f"⚠️ Overwriting existing results for phase '{phase}'")

        self.data["results"][phase] = results
        self._write()

    def add_field(self, key: str, value):
        self.data[key] = value
        self._write()

    def _write(self):
        with self._lock:
            try:
                temp_path = self.manifest_path.with_suffix(".tmp")

                with open(temp_path, "w") as f:
                    json.dump(self.data, f, indent=2)

                os.replace(temp_path, self.manifest_path)

            except Exception as e:
                print(f"❌ Manifest write failed: {e}")

    def get_run_dir(self):
        return str(self.base_path)