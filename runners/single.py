# runners/single.py

import os
import random
import socket
import time
import requests

from agents.q_agent import QAgent
from agent_sim.adapter.local_env import LocalEnv
from agent_sim.adapter.remote_env import RemoteEnv
from agent_sim.runner.manifest import RunManifest

# Shared execution logic
from runners.core import run_experiments

# -------------------- CONFIG --------------------

DEFAULT_BASE_URL = "http://localhost:8000/v1"
REMOTE_TIMEOUT = 2

ENV_RETRIES = int(os.environ.get("ENV_RETRIES", 10))
ENV_RETRY_DELAY = float(os.environ.get("ENV_RETRY_DELAY", 1.0))
SEED = int(os.environ.get("SEED", 42))
random.seed(SEED)

ENABLE_PARITY = os.environ.get("ENABLE_PARITY", "true").lower() == "true"
STRICT_MODE = os.environ.get("ENV_STRICT", "false").lower() == "true"
CHAOS_MODE = os.environ.get("ENABLE_CHAOS", "false").lower() == "true"
VERBOSE = os.environ.get("VERBOSE", "true").lower() == "true"

# -------------------- ENV FACTORY --------------------

def create_env():
    mode = os.environ.get("ENV_MODE", "local").lower()
    base_url = os.environ.get("BASE_URL", DEFAULT_BASE_URL)

    print("\n🚀 Agent Runner Starting")
    print(f"⚙️ ENV_MODE = {mode}")
    print(f"🌐 BASE_URL = {base_url}")
    print(f"🖥️ Host = {socket.gethostname()}")

    if mode == "remote":
        print(f"🌐 Attempting RemoteEnv: {base_url}")
        health_url = base_url.rstrip("/") + "/health"

        for attempt in range(1, ENV_RETRIES + 1):
            try:
                response = requests.get(health_url, timeout=REMOTE_TIMEOUT)
                if response.status_code == 200:
                    print("✅ Remote environment reachable")
                    return RemoteEnv(base_url=base_url)
                else:
                    raise RuntimeError(f"Health check failed: {response.status_code}")
            except Exception as e:
                print(f"⏳ Retry {attempt}/{ENV_RETRIES} failed: {e}")
                if attempt < ENV_RETRIES:
                    time.sleep(ENV_RETRY_DELAY)

        print("⚠️ Remote environment unavailable after retries")
        if STRICT_MODE:
            raise RuntimeError("❌ Remote env unreachable in strict mode")

        print("🔁 Falling back to LocalEnv")
        return LocalEnv()

    print("⚡ Using LocalEnv")
    return LocalEnv()


# -------------------- PARITY / CHAOS --------------------

class ParityEnv:
    def __init__(self, env1, env2):
        self.env1 = env1
        self.env2 = env2

    def reset(self):
        s1 = self.env1.reset()
        s2 = self.env2.reset()
        assert type(s1) == type(s2), f"Type mismatch: {type(s1)} != {type(s2)}"
        assert s1 == s2, f"Parity failure at reset: {s1} != {s2}"
        return s1

    def step(self, action):
        r1 = self.env1.step(action)
        r2 = self.env2.step(action)
        assert r1 == r2, f"Parity failure at step: {r1} != {r2}"
        return r1


class ChaosEnv:
    def __init__(self, env, failure_rate=0.05):
        self.env = env
        self.failure_rate = failure_rate

    def reset(self):
        if random.random() < self.failure_rate:
            raise RuntimeError("Chaos: reset failure injected")
        return self.env.reset()

    def step(self, action):
        if random.random() < self.failure_rate:
            raise RuntimeError("Chaos: step failure injected")
        return self.env.step(action)


# -------------------- TRAIN THEN GREEDY --------------------

def train_then_greedy(env, agent, train_episodes=200, eval_episodes=20, max_steps=50):
    """
    Backward-compatible wrapper: train agent, then evaluate greedily.
    """
    print("\n🟢 TRAINING PHASE (Q-learning)")
    run_experiments(env, agent=agent, num_episodes=train_episodes, max_steps=max_steps, mode="train")

    print("\n🔵 EVALUATION PHASE (greedy)")
    results = run_experiments(env, agent=agent, num_episodes=eval_episodes, max_steps=max_steps, mode="eval")

    return results


# -------------------- MAIN ENTRY --------------------

if __name__ == "__main__":
    # Create environment
    base_env = create_env()

    # Apply parity if enabled
    if ENABLE_PARITY:
        print("🔍 Parity mode enabled (Local vs Remote)")
        env = ParityEnv(LocalEnv(), base_env)
    else:
        env = base_env

    # Apply chaos if enabled
    if CHAOS_MODE:
        print("🔥 Chaos mode enabled")
        env = ChaosEnv(env)

    # -------------------- AGENT --------------------

    alpha = float(os.environ.get("ALPHA", 0.5))
    epsilon = float(os.environ.get("EPSILON", 0.3))

    agent = QAgent(alpha=alpha, epsilon=epsilon)
    print(f"🧠 QAgent config: alpha={alpha}, epsilon={epsilon}, seed={SEED}")

    # -------------------- MANIFEST --------------------

    config = {
        "runner_type": "single",
        "env_mode": os.environ.get("ENV_MODE", "local"),
        "base_url": os.environ.get("BASE_URL", DEFAULT_BASE_URL),
        "train_episodes": int(os.environ.get("TRAIN_EPISODES", 200)),
        "eval_episodes": int(os.environ.get("EVAL_EPISODES", 20)),
        "max_steps": int(os.environ.get("MAX_STEPS", 50)),
        "seed": SEED,
        "parity": ENABLE_PARITY,
        "chaos": CHAOS_MODE,
        "strict": STRICT_MODE,
        "agent": "q_agent",
    }

    manifest = RunManifest(config)
    os.environ["LOG_DIR"] = manifest.get_run_dir()
    manifest.add_field("hostname", socket.gethostname())
    manifest.add_field("timestamp_start", time.time())
    manifest.add_field("git_commit", os.environ.get("GIT_COMMIT", "unknown"))

    # -------------------- RUN --------------------

    results = train_then_greedy(
        env,
        agent,
        train_episodes=config["train_episodes"],
        eval_episodes=config["eval_episodes"],
        max_steps=config["max_steps"]
    )

    manifest.update_results(results)
    manifest.add_field("timestamp_end", time.time())

    print(f"\n📁 Run saved to: {manifest.get_run_dir()}")