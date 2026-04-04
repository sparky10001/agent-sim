import sys
from statistics import mean

from agent_sim.replay.loader import load_run


def summarize(run_file):
    data = load_run(run_file)

    episodes = data["episodes"]
    summaries = data.get("summaries", {})

    total_episodes = len(episodes)

    if total_episodes == 0:
        print("No episodes found.")
        return

    rewards = []
    lengths = []

    # Prefer summaries if available
    for ep in episodes:
        if ep in summaries:
            rewards.append(summaries[ep]["total_reward"])
            lengths.append(summaries[ep]["length"])
        else:
            steps = episodes[ep]
            rewards.append(sum(s["reward"] for s in steps))
            lengths.append(len(steps))

    total_reward = sum(rewards)
    total_steps = sum(lengths)

    avg_reward = mean(rewards)
    avg_steps = mean(lengths)

    min_reward = min(rewards)
    max_reward = max(rewards)

    # Simple trend check
    split = max(1, total_episodes // 5)
    early_avg = mean(rewards[:split])
    late_avg = mean(rewards[-split:])

    trend = "↗ improving" if late_avg > early_avg else "↘ declining" if late_avg < early_avg else "→ flat"

    print("\n===== RUN SUMMARY =====")
    print(f"Episodes: {total_episodes}")
    print(f"Total Reward: {total_reward}")
    print(f"Avg Reward per Episode: {avg_reward:.2f}")
    print(f"Avg Steps per Episode: {avg_steps:.2f}")

    print("\n----- Distribution -----")
    print(f"Best Episode Reward: {max_reward}")
    print(f"Worst Episode Reward: {min_reward}")

    print("\n----- Trend -----")
    print(f"Early Avg Reward: {early_avg:.2f}")
    print(f"Late Avg Reward: {late_avg:.2f}")
    print(f"Trend: {trend}")

    print()


# -------------------- CLI --------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python summarize.py <run_file>")
        sys.exit(1)

    summarize(sys.argv[1])