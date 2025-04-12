import numpy as np

'''
Action: ndarray in shape (7,)
    0-2: xyz
    3-5: wrist in axis angle
    6: gripper: -1 is close, 1 is open
'''

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