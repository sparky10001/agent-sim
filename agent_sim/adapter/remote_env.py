import requests
from requests.exceptions import RequestException

from agent_sim.protocol.env import (
    to_observation,
    to_step,
    validate_action
)
from .env_interface import EnvInterface


class RemoteEnv(EnvInterface):
    def __init__(self, base_url="http://localhost:8000/v1", timeout=2.0, max_retries=2):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries

        self.session_id = None

    # -------------------- CORE API --------------------

    def reset(self):
        data = self._post("/reset")

        if "session_id" not in data or "state" not in data:
            raise RuntimeError("Invalid reset response: missing fields")

        self.session_id = data["session_id"]

        return to_observation(data["state"])

    def step(self, action):
        if not self.session_id:
            raise RuntimeError("Must call reset() before step()")

        # Validate BEFORE sending
        action = validate_action(action)

        payload = {
            "session_id": self.session_id,
            "action": action
        }

        data = self._post("/step", json=payload)

        # Validate structure
        for key in ["state", "reward", "done"]:
            if key not in data:
                raise RuntimeError(f"Invalid response: missing '{key}'")

        return to_step(
            data["state"],
            data["reward"],
            data["done"]
        )

    # -------------------- INTERNALS --------------------

    def _post(self, endpoint, json=None):
        url = f"{self.base_url}{endpoint}"

        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                res = requests.post(url, json=json, timeout=self.timeout)
                self._check_response(res)
                return res.json()

            except RequestException as e:
                last_error = e

            except RuntimeError as e:
                # Handle expired/invalid session gracefully
                if "Invalid session_id" in str(e):
                    self.session_id = None
                    raise RuntimeError("Session expired — call reset() again") from e
                raise

        raise RuntimeError(f"RemoteEnv request failed after retries: {last_error}")

    def _check_response(self, res):
        try:
            res.raise_for_status()
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(
                f"RemoteEnv HTTP error: {res.status_code} - {res.text}"
            ) from e

        try:
            res.json()
        except ValueError as e:
            raise RuntimeError("Invalid JSON response from environment") from e