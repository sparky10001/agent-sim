# runners/core.py
import random
import time
import os

from agents.q_agent import QAgent
from agent_sim.adapter.local_env import LocalEnv
from agent_sim.adapter.remote_env import RemoteEnv

# -------------------- ENV DECORATORS --------------------

class ParityEnv:
    """Wraps two environments to ensure deterministic parity."""
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
    """Wraps an environment and randomly injects failures."""
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

# -------------------- EXECUTION LOGIC --------------------

def run_episode(env, agent=None, max_steps=50, greedy=False, seed=None):
    if seed is not None:
        random.seed(seed)

    state = env.reset()
    total_reward = 0

    for step in range(max_steps):
        if agent:
            action = agent.act(state, greedy=greedy)
        else:
            action = random.choice(["up", "down", "left", "right"])

        next_state, reward, done = env.step(action)
        total_reward += reward
        state = next_state

        if done:
            break

    return total_reward, step + 1

def run_experiments(env, agent=None, num_episodes=50, max_steps=50, mode="train", policy_name="random"):
    results = []
    for ep in range(num_episodes):
        total_reward, steps_taken = run_episode(
            env,
            agent=agent,
            max_steps=max_steps,
            greedy=(policy_name=="greedy")
        )
        results.append({
            "episode": ep + 1,
            "reward": total_reward,
            "steps": steps_taken,
            "mode": mode,
            "policy": policy_name
        })
    return results

def train_then_greedy(env, agent, train_episodes=200, eval_episodes=20, max_steps=50):
    print("\n🟢 TRAINING PHASE")
    train_results = run_experiments(env, agent=agent, num_episodes=train_episodes, max_steps=max_steps, mode="train")

    print("\n🔵 EVALUATION PHASE (greedy)")
    eval_results = run_experiments(env, agent=agent, num_episodes=eval_episodes, max_steps=max_steps, mode="eval", policy_name="greedy")

    return {"train": train_results, "eval": eval_results}