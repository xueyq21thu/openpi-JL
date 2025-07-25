from abc import ABC, abstractmethod

class NoiseModel(ABC):
    """
    Abstract base class for noise models.

    The `NoiseModel` class serves as an abstract base class for implementing noise models in a reinforcement learning or control environment.
    It defines the structure and required methods for any derived noise model, including `sample`, `reset`, `compute_noise`, and `compute_reward`.
    These methods allow for injecting noise into actions, resetting the noise model, computing noise based on the current state and action, and calculating rewards based on observations and goals.

    Additionally, the file provides utility functions such as `sample_noise`, which generates Gaussian noise for specific action components (e.g., position, wrist, or gripper) based on the current step in a sequence.
    The `noise_injection` function adds noise to an action vector, ensuring the resulting action remains within valid bounds. Lastly, the `compute_reward` function is a placeholder for implementing task-specific reward calculations, which are essential for evaluating agent performance in the environment.

    This modular design allows for flexible integration of noise models and noise injection mechanisms into various robotic or simulation tasks, enabling robust testing and training of control algorithms.
    """
    def __init__(self, config=None):
        self.config = config if config is not None else {}
    
    @abstractmethod
    def sample(self, state, action=None, image=None, text_instruction=None):
        """
        Sample noise based on the current state and action.
        Args:
            state: Current state of the environment.
            action: Current action to be modified with noise.
            image: Optional image input (if applicable).
            text_instruction: Optional text instruction input (if applicable).
        Return:
            Delta: noise with randomized amplitude and direction.
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Reset the noise model to its initial state.
        Call this method when starting a new episode or task.
        """
        pass

    @abstractmethod
    def compute_noise(self, state, action=None, image=None):
        """
        Compute noise based on the current state and action.
        The key method to generate noise for the action.
        Diffrent from sample_noise, this method should be called to compute noise,
        while this method would net consider the time step.
        Args:
            state: Current state of the environment.
            action: Current action to be modified with noise.
            image: Optional image input (if applicable).
        Return:
            Delta: noise with randomized amplitude and direction.
        """
        pass

    @abstractmethod
    def compute_reward(self, success: bool, delta):
        """
        Compute the reward based on the current observation and goal.
        Args:
            obs: Current observation of the environment.
            goal: Target goal for the agent.
        Return:
            reward: Computed reward based on the task.
        """
        pass

    @abstractmethod
    def compute_loss(self, predicted_delta, target_delta=None, reward=None):
        """
        Compute the loss based on the current delta and reward.
        """
        pass
