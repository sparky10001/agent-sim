# runners/core.py

def run_episode(env, agent=None, max_steps=50, greedy=False, seed=None):
    ...

def run_experiments(env, agent=None, num_episodes=50, max_steps=50, mode="train"):
    ...

def train_then_greedy(env, agent, train_episodes=200, eval_episodes=20, max_steps=50):
    ...