import requests
import random
import time

BASE_URL = "http://localhost:8000/v1"

ACTIONS = ["up", "down", "left", "right"]


def random_policy(state):
    return random.choice(ACTIONS)


def run_episode(max_steps=50, delay=0):
    # Reset environment
    res = requests.post(f"{BASE_URL}/reset")
    state = res.json()["state"]

    total_reward = 0

    for step in range(max_steps):
        action = random_policy(state)

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


def run_experiments(num_episodes=10):
    results = []

    for ep in range(num_episodes):
        total_reward, steps = run_episode()
        print(f"Episode {ep+1}: reward={total_reward}, steps={steps}")
        results.append((total_reward, steps))

    return results


if __name__ == "__main__":
    run_experiments(10)