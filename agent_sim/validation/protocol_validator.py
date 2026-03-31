import os
import time
import requests
from typing import Any, Dict

from agent_sim.protocol.env import to_observation, validate_state


# -------------------- CONFIG --------------------

DEFAULT_BASE_URL = "http://localhost:8000/v1"
TIMEOUT = float(os.environ.get("VALIDATOR_TIMEOUT", 2.0))
RETRIES = int(os.environ.get("VALIDATOR_RETRIES", 3))
RETRY_DELAY = float(os.environ.get("VALIDATOR_RETRY_DELAY", 0.5))

VALIDATE_DETERMINISM = os.environ.get("VALIDATE_DETERMINISM", "false").lower() == "true"


# -------------------- HELPER --------------------

class RequestHelper:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def _request(self, method: str, path: str, **kwargs):
        url = f"{self.base_url}{path}"

        for attempt in range(1, RETRIES + 1):
            try:
                response = requests.request(method, url, timeout=TIMEOUT, **kwargs)
                return response
            except Exception as e:
                if attempt < RETRIES:
                    print(f"⏳ Retry {attempt}/{RETRIES} failed for {path}: {e}")
                    time.sleep(RETRY_DELAY)
                else:
                    raise RuntimeError(f"❌ Request failed after retries: {path}") from e

    def get(self, path: str):
        return self._request("GET", path)

    def post(self, path: str, **kwargs):
        return self._request("POST", path, **kwargs)


# -------------------- VALIDATOR --------------------

class ProtocolValidator:
    def __init__(self, base_url: str):
        self.client = RequestHelper(base_url)

    # ---------- HEALTH ----------
    def check_health(self):
        print("🔍 Checking /health...")
        r = self.client.get("/health")
        assert r.status_code == 200, "❌ Health check failed"

    # ---------- RESET ----------
    def check_reset(self):
        print("🔍 Checking /reset...")
        r = self.client.post("/reset")
        assert r.status_code == 200, "❌ Reset failed"

        data = r.json()
        assert "state" in data, "❌ Missing 'state' in reset response"

        state = to_observation(data["state"])
        validate_state(state)

    # ---------- STEP ----------
    def check_step(self):
        print("🔍 Checking /step...")

        r = self.client.post("/reset")
        state = r.json()["state"]

        for action in ["up", "down", "left", "right"]:
            r = self.client.post("/step", json={"action": action})
            assert r.status_code == 200, f"❌ Step failed for action {action}"

            data = r.json()

            # Required fields
            for key in ["state", "reward", "done"]:
                assert key in data, f"❌ Missing '{key}' in step response"

            # Type validation
            assert isinstance(data["reward"], (int, float)), "❌ reward must be numeric"
            assert isinstance(data["done"], bool), "❌ done must be boolean"

            # State validation
            state = to_observation(data["state"])
            validate_state(state)

    # ---------- INVALID ACTION ----------
    def check_invalid_action(self):
        print("🔍 Checking invalid action handling...")

        r = self.client.post("/step", json={"action": "INVALID_ACTION"})

        assert r.status_code in (400, 422), (
            "❌ Invalid action was not rejected properly "
            f"(status={r.status_code})"
        )

    # ---------- DETERMINISM ----------
    def check_determinism(self):
        if not VALIDATE_DETERMINISM:
            return

        print("🔍 Checking determinism...")

        r1 = self.client.post("/reset")
        r2 = self.client.post("/reset")

        s1 = r1.json()["state"]
        s2 = r2.json()["state"]

        assert s1 == s2, "❌ Environment reset is not deterministic"

    # ---------- RUN ALL ----------
    def run_all(self):
        print("\n🧪 Running Protocol Validation Suite...\n")

        self.check_health()
        self.check_reset()
        self.check_step()
        self.check_invalid_action()
        self.check_determinism()

        print("\n✅ Protocol validation PASSED\n")


# -------------------- CLI ENTRY --------------------

if __name__ == "__main__":
    base_url = os.environ.get("BASE_URL", DEFAULT_BASE_URL)

    print(f"🌐 Validator targeting: {base_url}")

    validator = ProtocolValidator(base_url)
    validator.run_all()