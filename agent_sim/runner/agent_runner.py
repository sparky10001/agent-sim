import os
import requests
import random
import time

BASE_URL = "http://localhost:8000/v1"
ACTIONS = ["up", "down", "left", "right"]

# Policy selection via environment variable
POLICY = os.environ.get("POLICY", "random")


def random_policy(state):
    """Randomly pick an action."""
    return random.choice(ACTIONS)


def greedy_policy(state):
    """Move toward the goal deterministically."""
    x, y = state["x"], state["y"]
    goal_x, goal_y = state["goal"]

    if x < goal_x:
        return "right"
    elif x > goal_x:
        return "left"
    elif y < goal_y:
        return "up"
    elif y > goal_y:
        return "down"

    # fallback (already at goal)
    return random.choice(ACTIONS)


def select_action(state):
    """Select action based on chosen policy."""
    if POLICY == "greedy":
        return greedy_policy(state)
    return random_policy(state)


def run_episode(max_steps=50, delay=0):
    """Run a single episode."""
    res = requests.post(f"{BASE_URL}/reset")
    state = res.json()["state"]

    total_reward = 0

    for step in range(max_steps):
        action = select_action(state)

        res = requests.post(
            f"{BASE_URL}/step",
            json={"action": action}
        )
        data = res.json()

        state = data["state"]
        reward = data["reward"]
        done = data["done"]

        total_reward += reward

        if delay:
            time.sleep(delay)

        if done:
            break

    return total_reward, step + 1


def run_experiments(num_episodes=10, max_steps=50, delay=0):
    """Run multiple episodes and report metrics."""
    print(f"Running policy: {POLICY}")
    results = []

    for ep in range(num_episodes):
        total_reward, steps = run_episode(max_steps=max_steps, delay=delay)
        print(f"Episode {ep+1}: reward={total_reward}, steps={steps}")
        results.append((total_reward, steps))

    # Metrics
    successes = sum(1 for r, _ in results if r > 0)
    success_rate = successes / len(results)

    avg_reward = sum(r for r, _ in results) / len(results)
    avg_steps = sum(s for _, s in results) / len(results)

    print("\n===== RUN SUMMARY =====")
    print(f"Episodes: {len(results)}")
    print(f"Success Rate: {success_rate:.2f}")
    print(f"Avg Reward: {avg_reward:.2f}")
    print(f"Avg Steps: {avg_steps:.2f}")

    return results


if __name__ == "__main__":
    run_experiments(num_episodes=10, max_steps=50, delay=0)