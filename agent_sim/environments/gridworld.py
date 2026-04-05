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
            "goal": list(self.goal)  # JSON-safe
        }
        return self.state

    def step(self, action):
        """
        Move the agent.
        action: "up", "down", "left", "right"
        Returns: new state, reward, done
        """
        x, y = self.state["x"], self.state["y"]

        # Calculate candidate move
        new_x, new_y = x, y
        if action == "up":
            new_y -= 1
        elif action == "down":
            new_y += 1
        elif action == "left":
            new_x -= 1
        elif action == "right":
            new_x += 1

        # Clamp to grid bounds
        new_x = max(0, min(new_x, self.width - 1))
        new_y = max(0, min(new_y, self.height - 1))

        # Check if moved
        moved = (new_x != x) or (new_y != y)

        # Update state
        self.state = {
            "x": new_x,
            "y": new_y,
            "goal": list(self.goal)
        }

        # Reward / done
        done = (new_x, new_y) == self.goal
        if done:
            reward = 1.0
        elif not moved:
            reward = -0.1  # penalty for no-op
        else:
            reward = 0.0

        return self.state, reward, done