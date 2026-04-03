import os
import random
import socket
import time
import requests

from agents.q_agent import QAgent
from agent_sim.adapter.local_env import LocalEnv
from agent_sim.adapter.remote_env import RemoteEnv
from agent_sim.runner.manifest import RunManifest

ACTIONS = ["up", "down", "left", "right"]

# -------------------- CONFIG --------------------

DEFAULT_BASE_URL = "http://localhost:8000/v1"
REMOTE_TIMEOUT = 2  # seconds
ENABLE_FALLBACK = True  # fallback to LocalEnv if remote fails

ENV_RETRIES = int(os.environ.get("ENV_RETRIES", 10))
ENV_RETRY_DELAY = float(os.environ.get("ENV_RETRY_DELAY", 1.0))

SEED = int(os.environ.get("SEED", 42))
random.seed(SEED)

ENABLE_PARITY = os.environ.get("ENABLE_PARITY", "true").lower() == "true"
STRICT_MODE = os.environ.get("ENV_STRICT", "false").lower() == "true"
CHAOS_MODE = os.environ.get("ENABLE_CHAOS", "false").lower() == "true"

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
        if STRICT_MODE or not ENABLE_FALLBACK:
            raise RuntimeError("❌ Remote env unreachable in strict mode")
        else:
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

# -------------------- POLICIES --------------------

def random_policy(state):
    return random.choice(ACTIONS)

def greedy_policy(state):
    x, y = state["x"], state["y"]
    if x < 4:
        return "right"
    elif y < 4:
        return "down"
    return random.choice(ACTIONS)

# -------------------- CORE LOGIC --------------------

def run_episode(env, policy=None, agent=None, max_steps=50):
    state = env.reset()
    total_reward = 0

    for step in range(max_steps):
        action = agent.select_action(state) if agent else policy(state)
        next_state, reward, done = env.step(action)

        if reward > 0:
            print(f"🎯 GOAL REACHED at step {step}")

        total_reward += reward

        if agent:
            agent.update(state, action, reward, next_state, done)

        state = next_state
        if done:
            return total_reward, step + 1

    return total_reward, max_steps


def run_experiments(env, policy_name="random", num_episodes=50, max_steps=50, agent=None):
    policy = random_policy

    if policy_name == "greedy":
        policy = greedy_policy
        agent = None
    elif policy_name == "q" and agent is None:
        epsilon = float(os.environ.get("EPSILON", 0.3))
        alpha = float(os.environ.get("ALPHA", 0.5))
        print(f"🧠 QAgent config: alpha={alpha}, epsilon={epsilon}")
        agent = QAgent(alpha=alpha, epsilon=epsilon)
        policy = None

    results = []
    print(f"Running policy: {policy_name if policy_name != 'q' else 'q-learning'}")

    for ep in range(num_episodes):
        try:
            total_reward, steps = run_episode(env, policy, agent, max_steps)
        except Exception as e:
            print(f"⚠️ Episode {ep+1} failed: {e}")
            total_reward, steps = 0, 0

        results.append((total_reward, steps))

        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"Episode {ep + 1}: reward={total_reward}, steps={steps}")

    successes = sum(1 for r, _ in results if r > 0)
    avg_reward = sum(r for r, _ in results) / len(results)
    avg_steps = sum(s for _, s in results) / len(results)

    print("\n===== RUN SUMMARY =====")
    print(f"Episodes: {num_episodes}")
    print(f"Success Rate: {successes / num_episodes:.2f}")
    print(f"Avg Reward: {avg_reward:.3f}")
    print(f"Avg Steps: {avg_steps:.2f}")

    return {
        "episodes": num_episodes,
        "success_rate": successes / num_episodes,
        "avg_reward": avg_reward,
        "avg_steps": avg_steps
    }


def train_then_greedy(env, train_episodes=200, eval_episodes=20, max_steps=50):
    print("\n🟢 TRAINING PHASE (Q-learning)")
    run_experiments(env, policy_name="q", num_episodes=train_episodes, max_steps=max_steps)

    print("\n🔵 EVALUATION PHASE (Greedy)")
    results = run_experiments(env, policy_name="greedy", num_episodes=eval_episodes, max_steps=max_steps)

    return results

# -------------------- ENTRY POINT --------------------

if __name__ == "__main__":
    remote_env = create_env()

    if ENABLE_PARITY:
        env = ParityEnv(LocalEnv(), remote_env)
    else:
        env = remote_env

    if CHAOS_MODE:
        env = ChaosEnv(env)

    # -------------------- MANIFEST --------------------

    config = {
        "env_mode": os.environ.get("ENV_MODE", "local"),
        "base_url": os.environ.get("BASE_URL", DEFAULT_BASE_URL),
        "train_episodes": int(os.environ.get("TRAIN_EPISODES", 200)),
        "eval_episodes": int(os.environ.get("EVAL_EPISODES", 20)),
        "max_steps": int(os.environ.get("MAX_STEPS", 50)),
        "seed": SEED,
        "parity": ENABLE_PARITY,
        "chaos": CHAOS_MODE,
        "strict": STRICT_MODE,
        "agent": "q_agent"
    }

    manifest = RunManifest(config)

    # Co-locate replay logs
    os.environ["LOG_DIR"] = manifest.get_run_dir()

    # Extra metadata
    manifest.add_field("hostname", socket.gethostname())
    manifest.add_field("timestamp_start", time.time())

    # -------------------- RUN --------------------

    results = train_then_greedy(
        env,
        train_episodes=config["train_episodes"],
        eval_episodes=config["eval_episodes"],
        max_steps=config["max_steps"]
    )

    # -------------------- FINALIZE --------------------

    manifest.update_results(results)
    manifest.add_field("timestamp_end", time.time())

    print(f"\n📁 Run saved to: {manifest.get_run_dir()}")