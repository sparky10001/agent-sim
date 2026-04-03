import os
import json
from datetime import datetime
from pathlib import Path


RUNS_DIR = os.environ.get("RUNS_DIR", "/app/runs")


class RunManifest:
    def __init__(self, config: dict):
        self.run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.base_path = Path(RUNS_DIR) / self.run_id
        self.base_path.mkdir(parents=True, exist_ok=True)

        self.manifest_path = self.base_path / "manifest.json"

        self.data = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "config": config,
            "results": {}
        }

        self._write()

    def update_results(self, results: dict):
        self.data["results"] = results
        self._write()

    def add_field(self, key: str, value):
        self.data[key] = value
        self._write()

    def _write(self):
        with open(self.manifest_path, "w") as f:
            json.dump(self.data, f, indent=2)

    def get_run_dir(self):
        return str(self.base_path)