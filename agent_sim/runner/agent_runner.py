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
    x, y = state["x"], state["y"]

    goal = state["goal"]
    if isinstance(goal, dict):
        goal_x, goal_y = goal["x"], goal["y"]
    else:
        goal_x, goal_y = goal

    dx = goal_x - x
    dy = goal_y - y

    # Move along the axis with the greatest distance
    if abs(dx) > abs(dy):
        return "right" if dx > 0 else "left"
    else:
        return "down" if dy > 0 else "up"


def epsilon_greedy_policy(state, epsilon=0.1):
    if random.random() < epsilon:
        return random_policy(state)
    return greedy_policy(state)


def get_policy():
    policy_name = os.environ.get("POLICY", "random").lower()

    if policy_name == "greedy":
        return greedy_policy, "greedy", None

    elif policy_name == "epsilon":
        epsilon = float(os.environ.get("EPSILON", 0.1))
        return (
            lambda s: epsilon_greedy_policy(s, epsilon),
            f"epsilon-greedy (ε={epsilon})",
            None,
        )

    elif policy_name == "q":
        agent = QAgent(
            alpha=float(os.environ.get("ALPHA", 0.1)),
            gamma=float(os.environ.get("GAMMA", 0.9)),
            epsilon=float(os.environ.get("EPSILON", 0.1)),
        )
        return agent.select_action, "q-learning", agent

    else:
        return random_policy, "random", None


# -------------------- RUNNER --------------------

def run_episode(policy, agent=None, max_steps=50):
    res = requests.post(f"{BASE_URL}/reset")
    state = res.json()["state"]

    total_reward = 0

    for step in range(max_steps):
        action = policy(state)

        res = requests.post(f"{BASE_URL}/step", json={"action": action})
        data = res.json()

        next_state = data["state"]
        reward = data["reward"]
        done = data["done"]

        total_reward += reward

        # Q-learning update (only if agent exists)
        if agent:
            agent.update(state, action, reward, next_state, done)

        state = next_state

        if done:
            return total_reward, step + 1

    return total_reward, max_steps


def run_experiments(num_episodes=50, max_steps=50):
    policy, policy_name, agent = get_policy()
    print(f"Running policy: {policy_name}")

    results = []

    for ep in range(num_episodes):
        reward, steps = run_episode(policy, agent, max_steps)
        results.append((reward, steps))

        print(f"Episode {ep+1}: reward={reward}, steps={steps}")

    # -------------------- SUMMARY --------------------
    total_rewards = sum(r for r, _ in results)
    successes = sum(1 for r, _ in results if r > 0)
    avg_reward = total_rewards / num_episodes
    avg_steps = sum(s for _, s in results) / num_episodes

    print("\n===== RUN SUMMARY =====")
    print(f"Episodes: {num_episodes}")
    print(f"Success Rate: {successes / num_episodes:.2f}")
    print(f"Avg Reward: {avg_reward:.2f}")
    print(f"Avg Steps: {avg_steps:.2f}")


if __name__ == "__main__":
    run_experiments()