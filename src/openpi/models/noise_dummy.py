import numpy as np
from noise_model import NoiseModel

class DummyNoiseModel(NoiseModel):
    def __init__(self, config=None):
        super().__init__(config)
        # Config parameters
        cfg = self.config
        self.min_amp = cfg.get('min_amplitude', 0.01) # minimum amplitude of noise
        self.max_amp = cfg.get('max_amplitude', 0.2) # maximum amplitude of noise
        self.duration = cfg.get('duration', 0) # duration of noise injection in steps
        self.episode_length = cfg.get('episode_length', 220) # length of the total episode
       
        self.insert_step = None  
        self.noise_std = None
        self.current_step = 0 

    def reset(self):
        # Randomly choose a step to insert noise within the episode length
        self.insert_step = np.random.randint(0, self.episode_length)
        # randomly set the noise amplitude within the specified range
        self.noise_std = np.random.uniform(self.min_amp, self.max_amp)
        # reset the current step to 0
        self.current_step = 0

    def compute_noise(self, state, action=None, image=None):
        # calculate noise based on the action provided
        if action is None:
            raise ValueError("DummyNoiseModel requires an action to compute noise.")
        # turn action into numpy array if it is not already
        action_val = action
        if isinstance(action, np.ndarray) is False:
            action_val = np.array(action, dtype=float)
        noise = np.zeros_like(action_val)
        dim = noise.shape[-1]
        # all: add noise to all dimensions of the action
        if dim > 1:
            # add noise to all dimensions except the last one
            noise[0:dim-1] = np.random.normal(0, self.noise_std, size=(dim-1,))
        if dim > 0:
            # add noise to the last dimension (e.g., gripper open/close)
            if action_val[-1] > 0:
                noise[-1] = -1 - action_val[-1]
            else:
                noise[-1] = 1 - action_val[-1]
        return noise

    def sample(self, state, action=None, image=None):
        # judge if the current step is within the noise injection period
        if (self.current_step >= self.insert_step) and \
           (self.current_step <= self.insert_step + self.duration):
            # return noise based on the current state and action
            noise = self.compute_noise(state, action, image)
        else:
            # return zero noise if outside the noise injection period
            if action is not None:
                noise = np.zeros_like(action)
            else:
                noise = np.zeros_like(state)
        # increment the current step
        self.current_step += 1
        return noise

    def compute_reward(self, obs, goal=None):
        # For the dummy noise model, the reward is always 0.
        return 0.0
