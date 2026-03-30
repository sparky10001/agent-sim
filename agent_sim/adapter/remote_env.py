import requests
from agent_sim.adapter.env import to_observation
from .env_interface import EnvInterface


class RemoteEnv(EnvInterface):
    def __init__(self, base_url="http://localhost:8000/v1", timeout=2.0):
        """
        Remote environment via HTTP API.

        Args:
            base_url: Base API endpoint (no trailing slash)
            timeout: Request timeout in seconds (prevents hanging)
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # -------------------- CORE API --------------------

    def reset(self):
        res = requests.post(
            f"{self.base_url}/reset",
            timeout=self.timeout
        )
        self._check_response(res)

        state = res.json()["state"]
        return to_observation(state)

    def step(self, action):
        res = requests.post(
            f"{self.base_url}/step",
            json={"action": action},
            timeout=self.timeout
        )
        self._check_response(res)

        data = res.json()

        return (
            to_observation(data["state"]),
            data["reward"],
            data["done"]
        )

    # -------------------- INTERNALS --------------------

    def _check_response(self, res):
        """
        Fail fast on bad responses.
        """
        try:
            res.raise_for_status()
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(
                f"RemoteEnv HTTP error: {res.status_code} - {res.text}"
            ) from e

        # Optional: validate JSON structure early
        try:
            res.json()
        except ValueError as e:
            raise RuntimeError("Invalid JSON response from environment") from e