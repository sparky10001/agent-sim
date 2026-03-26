import sys
from agent_sim.replay.loader import load_run

def summarize(run_file):
    data = load_run(run_file)

    episodes = data["episodes"]

    total_episodes = len(episodes)
    total_steps = 0
    total_reward = 0

    for ep, steps in episodes.items():
        ep_reward = sum(s["reward"] for s in steps)
        total_reward += ep_reward
        total_steps += len(steps)

    avg_reward = total_reward / total_episodes if total_episodes else 0
    avg_steps = total_steps / total_episodes if total_episodes else 0

    print("===== RUN SUMMARY =====")
    print(f"Episodes: {total_episodes}")
    print(f"Total Reward: {total_reward}")
    print(f"Avg Reward per Episode: {avg_reward:.2f}")
    print(f"Avg Steps per Episode: {avg_steps:.2f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python summarize.py <run_file>")
        sys.exit(1)

    summarize(sys.argv[1])