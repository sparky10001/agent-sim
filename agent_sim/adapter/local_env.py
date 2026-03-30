from agent_sim.environments.gridworld import GridWorld
from agent_sim.adapter.env import to_observation
from .env_interface import EnvInterface

class LocalEnv(EnvInterface):
    def __init__(self):
        self.env = GridWorld()

    def reset(self):
        state = self.env.reset()
        return to_observation(state)

    def step(self, action):
        state, reward, done = self.env.step(action)
        return to_observation(state), reward, done