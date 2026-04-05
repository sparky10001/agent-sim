# runners/distributed.py

import os
from threading import Thread

from agent_sim.adapter.remote_env import RemoteEnv
from runners.core import run_experiments, QAgent, ChaosEnv, ParityEnv
from agent_sim.validation.protocol_validator import ProtocolValidator

# -------------------- CONFIG --------------------

BASE_URLS = os.environ.get("BASE_URLS", "http://localhost:8000/v1").split(",")
TRAIN_EPISODES = int(os.environ.get("TRAIN_EPISODES", 200))
EVAL_EPISODES = int(os.environ.get("EVAL_EPISODES", 20))
MAX_STEPS = int(os.environ.get("MAX_STEPS", 50))
ENABLE_CHAOS = os.environ.get("ENABLE_CHAOS", "false").lower() == "true"
ENABLE_PARITY = os.environ.get("ENABLE_PARITY", "false").lower() == "true"

# -------------------- WORKER --------------------

def _worker(worker_id, url):
    print(f"\n🌐 Worker {worker_id} targeting {url}")

    # Validate environment
    validator = ProtocolValidator(url)
    validator.run_all()

    env = RemoteEnv(url)
    if ENABLE_PARITY:
        env = ParityEnv(env, RemoteEnv(url))
    if ENABLE_CHAOS:
        env = ChaosEnv(env)

    # Create a QAgent
    alpha = float(os.environ.get("ALPHA", 0.5))
    epsilon = float(os.environ.get("EPSILON", 0.3))
    agent = QAgent(alpha=alpha, epsilon=epsilon)

    # Train then evaluate
    print(f"\n🟢 Worker {worker_id} training...")
    run_experiments(env, policy_name="q", num_episodes=TRAIN_EPISODES, max_steps=MAX_STEPS, agent=agent)

    print(f"\n🔵 Worker {worker_id} evaluating greedy policy...")
    results = run_experiments(env, policy_name="greedy", num_episodes=EVAL_EPISODES, max_steps=MAX_STEPS, agent=None)

    print(f"\n✅ Worker {worker_id} finished: {results}")


# -------------------- DISTRIBUTED ENTRYPOINT --------------------

def run_distributed():
    threads = []
    for i, url in enumerate(BASE_URLS):
        t = Thread(target=_worker, args=(i + 1, url))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    print("\n🎯 All workers completed.")


# -------------------- CLI / MODULE TEST --------------------

if __name__ == "__main__":
    run_distributed()