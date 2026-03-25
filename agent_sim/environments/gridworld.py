class GridWorld:
    def __init__(self, size=5):
        self.size = size
        self.reset()

    def reset(self):
        self.x, self.y = 0, 0
        self.goal = (self.size - 1, self.size - 1)
        return self.state()

    def step(self, action):
        if action == "right":
            self.x = min(self.size - 1, self.x + 1)
        elif action == "down":
            self.y = min(self.size - 1, self.y + 1)
        elif action == "left":
            self.x = max(0, self.x - 1)
        elif action == "up":
            self.y = max(0, self.y - 1)

        done = (self.x, self.y) == self.goal
        reward = 1 if done else -0.01

        return self.state(), reward, done

    def state(self):
        return {
            "x": self.x,
            "y": self.y,
            "goal": self.goal
        }