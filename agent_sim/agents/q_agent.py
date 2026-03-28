import random
from collections import defaultdict

ACTIONS = ["up", "down", "left", "right"]

class QAgent:
    def __init__(self, alpha=0.5, gamma=0.9, epsilon=0.3):
        self.alpha = alpha      # learning rate
        self.gamma = gamma      # discount factor
        self.epsilon = epsilon  # exploration rate

        # Q-table: state -> action -> value
        self.q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})

    def _state_key(self, state):
        """
        Convert state dict into a hashable key.
        Assumes state like: {"x": int, "y": int}
        """
        return (state["x"], state["y"])

    def select_action(self, state):
        key = self._state_key(state)

        # Ensure state exists in Q-table
        if key not in self.q:
            self.q[key] = {a: 0.0 for a in ACTIONS}

        # ε-greedy exploration
        if random.random() < self.epsilon:
            return random.choice(ACTIONS)

        # Exploitation with RANDOM tie-breaking (CRITICAL FIX)
        q_values = self.q[key]
        max_q = max(q_values.values())

        best_actions = [a for a, q in q_values.items() if q == max_q]

        return random.choice(best_actions)

    def update(self, state, action, reward, next_state, done):
        key = self._state_key(state)
        next_key = self._state_key(next_state)

        # Ensure next state exists
        if next_key not in self.q:
            self.q[next_key] = {a: 0.0 for a in ACTIONS}

        # Q-learning update
        max_next_q = max(self.q[next_key].values())

        target = reward if done else reward + self.gamma * max_next_q

        self.q[key][action] += self.alpha * (target - self.q[key][action])