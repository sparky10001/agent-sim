# environments/gridworld.py

class GridWorld:
    def __init__(self, width=5, height=5):
        self.width = width
        self.height = height
        self.goal = (width - 1, height - 1)
        self.state = None

    def reset(self):
        """Reset environment to deterministic initial state"""
        self.state = {
            "x": 0,
            "y": 0,
            "goal": self.goal
        }
        return self.state

    def step(self, action):
        """
        Move the agent.
        action: "up", "down", "left", "right"
        Returns: new state, reward, done
        """
        x, y = self.state["x"], self.state["y"]

        if action == "up" and y > 0:
            y -= 1
        elif action == "down" and y < self.height - 1:
            y += 1
        elif action == "left" and x > 0:
            x -= 1
        elif action == "right" and x < self.width - 1:
            x += 1

        self.state["x"], self.state["y"] = x, y

        # Simple reward: +1 for reaching goal, 0 otherwise
        done = (x, y) == self.goal
        reward = 1 if done else 0

        return self.state, reward, done
