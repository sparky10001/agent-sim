import requests
import random
import os

from agent_sim.agents.q_agent import QAgent

BASE_URL = "http://localhost:8000/v1"
ACTIONS = ["up", "down", "left", "right"]


# -------------------- POLICIES --------------------

def random_policy(state):
    return random.choice(ACTIONS)


def greedy_policy(state):
    # Simple hardcoded goal direction (for testing)
    x, y = state["x"], state["y"]

    if x < 4:
        return "right"
    elif y < 4:
        return "down"
    else:
        return random.choice(ACTIONS)


# -------------------- RUN EPISODE --------------------

def run_episode(policy, agent=None, max_steps=50):
    res = requests.post(f"{BASE_URL}/reset")
    state = res.json()["state"]

    total_reward = 0

    for step in range(max_steps):

        # ✅ CRITICAL: agent controls actions if present
        if agent:
            action = agent.select_action(state)
        else:
            action = policy(state)

        res = requests.post(f"{BASE_URL}/step", json={"action": action})
        data = res.json()

        next_state = data["state"]
        reward = data["reward"]
        done = data["done"]

        # 🔍 Debug: confirm learning signal
        if reward > 0:
            print(f"🎯 GOAL REACHED at step {step}")

        total_reward += reward

        # Q-learning update
        if agent:
            agent.update(state, action, reward, next_state, done)

        state = next_state

        if done:
            return total_reward, step + 1

    return total_reward, max_steps


# -------------------- RUN EXPERIMENTS --------------------

def run_experiments(policy_name="random", num_episodes=50, max_steps=50):
    policy = random_policy
    agent = None

    if policy_name == "greedy":
        policy = greedy_policy

    elif policy_name == "q":
        epsilon = float(os.environ.get("EPSILON", 0.3))
        alpha = float(os.environ.get("ALPHA", 0.5))

        agent = QAgent(alpha=alpha, epsilon=epsilon)
        policy = None  # agent decides

    results = []

    print(f"Running policy: {policy_name if policy_name != 'q' else 'q-learning'}")

    for ep in range(num_episodes):
        total_reward, steps = run_episode(policy, agent, max_steps)
        results.append((total_reward, steps))

        print(f"Episode {ep+1}: reward={total_reward}, steps={steps}")

    # -------------------- SUMMARY --------------------

    successes = sum(1 for r, _ in results if r > 0)
    avg_reward = sum(r for r, _ in results) / len(results)
    avg_steps = sum(s for _, s in results) / len(results)

    print("\n===== RUN SUMMARY =====")
    print(f"Episodes: {num_episodes}")
    print(f"Success Rate: {successes / num_episodes:.2f}")
    print(f"Avg Reward: {avg_reward:.2f}")
    print(f"Avg Steps: {avg_steps:.2f}")


# -------------------- ENTRY POINT --------------------

if __name__ == "__main__":
    policy_name = os.environ.get("POLICY", "random")
    run_experiments(policy_name=policy_name, num_episodes=200, max_steps=50)