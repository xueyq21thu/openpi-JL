import numpy as np
from abc import ABC, abstractmethod

'''
Action: ndarray in shape (7,)
    0-2: xyz
    3-5: wrist in axis angle
    6: gripper: -1 is close, 1 is open
'''

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
    def sample(self, state, action=None, image=None):
        """
        Sample noise based on the current state and action.
        Args:
            state: Current state of the environment.
            action: Current action to be modified with noise.
            image: Optional image input (if applicable).
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
    def compute_reward(self, obs, goal=None):
        """
        Compute the reward based on the current observation and goal.
        Args:
            obs: Current observation of the environment.
            goal: Target goal for the agent.
        Return:
            reward: Computed reward based on the task.
        """
        pass


def sample_noise(
        curr_step: int,
        insert_step: int,
        last_step: int,
        action: np.ndarray,
        type: str = "xyz",
        noise_scale: float = 0.3,
):
    '''
    Sample noise from a Gaussian distribution
    :param
        curr_step: current step
        insert_step: step to insert noise
        action: action to add noise to
        type: type of noise to add
        noise_scale: scale of noise
    '''
    noise = np.zeros(7)
    action_copy = action.copy()

    if curr_step == insert_step:
        print("inserting noise at step ", insert_step)
        print("Noise type: ", type)
        print("action: ", action)
    
    if curr_step >= insert_step and curr_step <= insert_step + last_step:
        if type == "xyz":
            noise[0:3] = np.random.normal(0, noise_scale, size=3)
            action_copy += noise
        elif type == "wrist":
            noise[3:6] = np.random.normal(0, 0.1, size=3)
            action_copy += noise
        elif type == "gripper":
            noise[0:3] = np.random.normal(0, noise_scale, size=3)
            if action[6] > 0:
                action_copy[6] = -1
            else:
                action_copy[6] = 1
            action_copy += noise
        elif type == "all":
            noise[0:3] = np.random.normal(0, noise_scale, size=3)
            noise[3:6] = np.random.normal(0, 0.1, size=3)
            if action[6] > 0:
                action_copy[6] = -1
            else:
                action_copy[6] = 1
            action_copy += noise
    
    return action_copy


def noise_injection(obs, action, noise_level=0.1):
    """
    Inject noise into the action vector.
    :param obs: Observation dictionary containing the current state of the environment.
    :param action: Action vector to inject noise into.
    :param noise_level: Scale of noise to inject.
    :return: Action vector with injected noise.
    """
    if obs is None or action is None:
        return None
    
    # TODO: Check che proper condition for noise injection
    if obs["robot0_gripper_qpos"] < -0.2:
        pass
    # Inject noise into the action vector
    noisy_action = action.copy()
    noisy_action[0:3] += np.random.normal(0, noise_level, 3)  # Add noise to x, y, z positions
    noisy_action[3:6] += np.random.normal(0, noise_level, 3)  # Add noise to wrist angles
    if action[6] < -0.2:
        noisy_action[6] = 1.0  # Open gripper
    return np.clip(noisy_action, -1.0, 1.0)  # Clip gripper action to [-1, 1]


def compute_reward(obs, goal):
    #TODO: Implement reward function based on the task
    # This reward is used to evaluate the performance of the agent in the environment
    # For now, we just return a dummy reward of 0.0
    return 0.0