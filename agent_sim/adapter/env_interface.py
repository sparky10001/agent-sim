from abc import ABC, abstractmethod


class EnvInterface(ABC):
    """
    Abstract environment interface.

    ALL environment implementations must follow this contract.
    """

    @abstractmethod
    def reset(self):
        """
        Reset environment to initial state.

        Returns:
            state (dict): normalized observation
        """
        pass

    @abstractmethod
    def step(self, action):
        """
        Apply action to environment.

        Args:
            action (str)

        Returns:
            state (dict): normalized observation
            reward (float or int)
            done (bool)
        """
        pass