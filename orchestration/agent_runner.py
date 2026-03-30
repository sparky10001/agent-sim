import os
import random

from agents.q_agent import QAgent
from agent_sim.adapter.local_env import LocalEnv
from agent_sim.adapter.remote_env import RemoteEnv

ACTIONS = ["up", "down", "left", "right"]

# -------------------- ENV FACTORY --------------------

def create_env():
    """
    Factory for selecting environment implementation.
    Controlled via ENV_MODE:
        - "local"  (default)
        - "remote" (API / Docker / VM)
    """
    mode = os.environ.get("ENV_MODE", "local")

    if mode == "remote":
        base_url = os.environ.get("BASE_URL", "http://localhost:8000/v1")
        print(f"🌐 Using RemoteEnv: {base_url}")
        return RemoteEnv(base_url=base_url)

    print("⚡ Using LocalEnv")
    return LocalEnv()


# -------------------- POLICIES --------------------

def random_policy(state):
    return random.choice(ACTIONS)


def greedy_policy(state):
    """Deterministic test policy for reaching bottom-right goal."""
    x, y = state["x"], state["y"]

    if x < 4:
        return "right"
    elif y < 4:
        return "down"
    else:
        return random.choice(ACTIONS)


# -------------------- RUN EPISODE --------------------

def run_episode(env, policy, agent=None, max_steps=50):
    state = env.reset()
    total_reward = 0

    for step in range(max_steps):

        if agent:
            action = agent.select_action(state)
        else:
            action = policy(state)

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


# -------------------- RUN EXPERIMENTS --------------------

def run_experiments(env, policy_name="random", num_episodes=50, max_steps=50, agent=None):
    """
    Run a series of episodes.
    If agent is provided, it controls actions.
    """
    # Determine policy if no agent
    policy = random_policy
    if policy_name == "greedy":
        policy = greedy_policy
        agent = None
    elif policy_name == "q" and agent is None:
        # Initialize QAgent if not provided
        epsilon = float(os.environ.get("EPSILON", 0.3))
        alpha = float(os.environ.get("ALPHA", 0.5))
        agent = QAgent(alpha=alpha, epsilon=epsilon)
        policy = None  # agent decides

    results = []

    print(f"Running policy: {policy_name if policy_name != 'q' else 'q-learning'}")

    for ep in range(num_episodes):
        total_reward, steps = run_episode(env, policy, agent, max_steps)
        results.append((total_reward, steps))

        # Minimal logging
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"Episode {ep + 1}: reward={total_reward}, steps={steps}")

    # -------------------- SUMMARY --------------------
    successes = sum(1 for r, _ in results if r > 0)
    avg_reward = sum(r for r, _ in results) / len(results)
    avg_steps = sum(s for _, s in results) / len(results)

    print("\n===== RUN SUMMARY =====")
    print(f"Episodes: {num_episodes}")
    print(f"Success Rate: {successes / num_episodes:.2f}")
    print(f"Avg Reward: {avg_reward:.2f}")
    print(f"Avg Steps: {avg_steps:.2f}")

    return agent  # return agent for later evaluation if needed


# -------------------- TRAIN THEN GREEDY EVALUATION --------------------

def train_then_greedy(env, train_episodes=200, eval_episodes=20, max_steps=50):
    """
    Train a Q-learning agent, then evaluate with greedy policy
    """
    print("\n🟢 TRAINING PHASE (Q-learning)")
    agent = run_experiments(env, policy_name="q", num_episodes=train_episodes, max_steps=max_steps)

    print("\n🔵 EVALUATION PHASE (Greedy using learned Q-table)")
    # Create a temporary policy function that uses QAgent deterministically (greedy)
    def learned_greedy_policy(state):
        return agent.select_action(state)

    run_experiments(env, policy_name="greedy", num_episodes=eval_episodes, max_steps=max_steps, agent=agent)


# -------------------- ENTRY POINT --------------------

if __name__ == "__main__":
    env = create_env()
    train_then_greedy(env,
                      train_episodes=int(os.environ.get("TRAIN_EPISODES", 200)),
                      eval_episodes=int(os.environ.get("EVAL_EPISODES", 20)),
                      max_steps=int(os.environ.get("MAX_STEPS", 50)))