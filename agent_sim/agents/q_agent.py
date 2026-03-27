import random
from collections import defaultdict

ACTIONS = ["up", "down", "left", "right"]

class QAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
        self.alpha = alpha      # learning rate
        self.gamma = gamma      # discount factor
        self.epsilon = epsilon  # exploration rate

    def _state_key(self, state):
        """Convert state dict → hashable key"""
        goal = state["goal"]
        if isinstance(goal, dict):
            goal = (goal["x"], goal["y"])
        else:
            goal = tuple(goal)

        return (state["x"], state["y"], goal)

    def select_action(self, state):
        """ε-greedy action selection"""
        key = self._state_key(state)

        if random.random() < self.epsilon:
            return random.choice(ACTIONS)

        return max(self.q[key], key=self.q[key].get)

    def update(self, state, action, reward, next_state, done):
        """Q-learning update"""
        key = self._state_key(state)
        next_key = self._state_key(next_state)

        current_q = self.q[key][action]
        max_next_q = max(self.q[next_key].values()) if not done else 0

        new_q = current_q + self.alpha * (
            reward + self.gamma * max_next_q - current_q
        )

        self.q[key][action] = new_q