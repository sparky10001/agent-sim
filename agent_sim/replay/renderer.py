import sys
import time
import os

from agent_sim.replay.loader import load_run


# -------------------- CONFIG --------------------

DEFAULT_DELAY = float(os.environ.get("RENDER_DELAY", 0.3))
GRID_WIDTH = int(os.environ.get("GRID_WIDTH", 5))
GRID_HEIGHT = int(os.environ.get("GRID_HEIGHT", 5))


# -------------------- RENDER CORE --------------------

def clear():
    os.system("cls" if os.name == "nt" else "clear")


def render_grid(state):
    grid = [["." for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]

    x, y = state["x"], state["y"]
    gx, gy = state["goal"]

    # Place goal
    grid[gy][gx] = "G"

    # Place agent (overwrites goal if same position)
    grid[y][x] = "A"

    for row in grid:
        print(" ".join(row))


def render_step(step, delay=DEFAULT_DELAY):
    clear()

    print(f"Episode: {step['episode']} | Step: {step['step']}")
    print(f"Action: {step['action']} | Reward: {step['reward']} | Done: {step['done']}")
    print()

    render_grid(step["state"])

    time.sleep(delay)


# -------------------- EPISODE PLAYBACK --------------------

def play_episode(steps, delay=DEFAULT_DELAY):
    for step in steps:
        render_step(step, delay)

    print("\n🏁 Episode complete")
    time.sleep(1)


def play_run(run_file, episode=None, delay=DEFAULT_DELAY):
    data = load_run(run_file)

    episodes = data["episodes"]

    if not episodes:
        print("No episodes found.")
        return

    # Sort episode keys (important!)
    episode_ids = sorted(episodes.keys())

    if episode is not None:
        if episode not in episodes:
            print(f"Episode {episode} not found.")
            return

        play_episode(episodes[episode], delay)
        return

    # Play all episodes
    for ep_id in episode_ids:
        print(f"\n▶ Playing Episode {ep_id}")
        time.sleep(1)
        play_episode(episodes[ep_id], delay)


# -------------------- CLI --------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python renderer.py <run_file> [episode] [delay]")
        sys.exit(1)

    run_file = sys.argv[1]
    episode = int(sys.argv[2]) if len(sys.argv) > 2 else None
    delay = float(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_DELAY

    play_run(run_file, episode=episode, delay=delay)