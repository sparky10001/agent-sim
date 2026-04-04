import os
import time
from typing import Optional
import requests

from agent_sim.protocol.env import validate_action

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
                resp = requests.request(method, url, timeout=TIMEOUT, **kwargs)
                return resp
            except Exception as e:
                if attempt < RETRIES:
                    print(f"⏳ Retry {attempt}/{RETRIES} failed for {path}: {e}")
                    time.sleep(RETRY_DELAY)
                else:
                    raise RuntimeError(f"❌ Request failed after retries: {path}") from e

    def get(self, path: str):
        return self._request("GET", path)

    def post(self, path: str, json: Optional[dict] = None):
        return self._request("POST", path, json=json)


# -------------------- VALIDATOR --------------------

class ProtocolValidator:
    def __init__(self, base_url: str):
        self.client = RequestHelper(base_url)

    # ---------- HEALTH ----------
    def check_health(self):
        print("🔍 Checking /health...")
        r = self.client.get("/health")
        assert r.status_code == 200, "❌ Health endpoint failed"
        data = r.json()
        assert data.get("status") == "ok", "❌ Health check failed"

    # ---------- RESET ----------
    def check_reset(self):
        print("🔍 Checking /reset...")
        r = self.client.post("/reset")
        assert r.status_code == 200, "❌ Reset failed"

        data = r.json()
        assert "session_id" in data, "❌ Missing session_id"
        assert "state" in data, "❌ Missing state"

        return data["session_id"], data["state"]

    # ---------- STEP ----------
    def check_step(self, session_id: Optional[str] = None):
        print("🔍 Checking /step...")

        if not session_id:
            session_id, _ = self.check_reset()

        prev_state = None

        for action in ["up", "down", "left", "right"]:
            validate_action(action)

            r = self.client.post("/step", json={
                "session_id": session_id,
                "action": action
            })

            data = r.json()

            assert r.status_code == 200, f"❌ Step failed: {data}"

            for key in ["state", "reward", "done"]:
                assert key in data, f"❌ Missing '{key}'"

            # Check progression (basic sanity)
            if prev_state:
                assert data["state"] != prev_state or data["done"], \
                    "❌ State did not change between steps"

            prev_state = data["state"]

    # ---------- INVALID ACTION ----------
    def check_invalid_action(self):
        print("🔍 Checking invalid action handling...")

        session_id, _ = self.check_reset()

        r = self.client.post("/step", json={
            "session_id": session_id,
            "action": "INVALID"
        })

        assert r.status_code == 400, "❌ Invalid action not rejected"

        data = r.json()
        assert "error" in data, "❌ Missing error response"

    # ---------- SESSION FLOW ----------
    def check_session_flow(self):
        print("🔍 Checking session lifecycle...")

        # Step without reset
        r = self.client.post("/step", json={"action": "up"})
        assert r.status_code == 400, "❌ Step without session should fail"

        # Step after done (simulate)
        session_id, _ = self.check_reset()

        for _ in range(200):  # force termination
            r = self.client.post("/step", json={
                "session_id": session_id,
                "action": "up"
            })
            if r.json().get("done"):
                break

        r = self.client.post("/step", json={
            "session_id": session_id,
            "action": "up"
        })

        assert r.status_code == 400, "❌ Step after done should fail"

    # ---------- DETERMINISM ----------
    def check_determinism(self):
        if not VALIDATE_DETERMINISM:
            return

        print("🔍 Checking determinism...")

        _, s1 = self.check_reset()
        _, s2 = self.check_reset()

        assert s1 == s2, "❌ Initial states differ (non-deterministic reset)"

    # ---------- RUN ALL ----------
    def run_all(self):
        print("\n🧪 Running Protocol Validation Suite...\n")

        self.check_health()
        session_id, _ = self.check_reset()
        self.check_step(session_id)
        self.check_invalid_action()
        self.check_session_flow()
        self.check_determinism()

        print("\n✅ Protocol validation PASSED\n")


# -------------------- CLI ENTRY --------------------

if __name__ == "__main__":
    base_url = os.environ.get("BASE_URL", DEFAULT_BASE_URL)
    print(f"🌐 Validator targeting: {base_url}")

    validator = ProtocolValidator(base_url)
    validator.run_all()