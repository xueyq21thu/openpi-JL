import collections
import dataclasses
import logging
import math
import pathlib

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

# import openpi.models.noise_model as noise_model
from openpi.models.noise_model import sample_noise

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data

import matplotlib.pyplot as plt

@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
        # TODO: Add more task suites
        # libero_spatial task suite with noise injection and replanning
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos

    img_out_path: str = "data/libero/images"  # Path to save images

    seed: int = 7  # Random Seed (for reproducibility)

    noise_type: str = "all"  # Noise type to insert. Options: xyz, wrist, gripper, all

    noise_insert_step: int = 75  # Step to insert noise

    noise_last_step: int = 10  # Steps to keep noise

    noise_scale: float = 0.3  # Scale of noise

def quat2axisangle(quat):

    """
    Convert a quaternion to axis-angle representation.
    :param quat: Quaternion to convert.
    :return: Axis-angle representation of the quaternion.
    """

    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


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


def get_libero_env(task, resolution, seed):
    """
    Get the LIBERO environment and task description.
    :param task: Task to create the environment for.
    :param resolution: Resolution of the environment.
    :param seed: Random seed for reproducibility.
    :return: LIBERO environment and task description.
    """

    task_description = task.language

    # TODO: Add support for other task suites
    task_bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file

    env_args = {"bddl_file_name": task_bddl, "camera_heights": resolution, "camera_widths": resolution}

    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description



# Global variables
success_rate_list_per_task: list = []  # List to store success rates of each task



def eval_libero(args: Args):
    """
    Evaluate the LIBERO environment with noise injection and replanning.
    :param args: Arguments for evaluation.
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Set random seed for reproducibility
    np.random.seed(args.seed)

    # Create output directories if they don't exist
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.img_out_path).mkdir(parents=True, exist_ok=True)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks

    max_num_steps = 400  # Maximum number of steps per trial

    # configure for steps to wait
    num_steps_wait = args.num_steps_wait  # Number of steps to wait for objects to stabilize in sim

    # Initialize OpenPI client
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start Episode
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")
            logging.info(f"Episode: {episode_idx + 1}")
            
            # Reset environment and get initial observation
            env.reset()

            # Set initial state of the environment
            obs = env.set_init_state(initial_states[episode_idx])

            # Action plan for the episode
            action_plan = collections.deque()

            # Init step
            step = 0

            replay_images_agent = []  # List to store images for replay video
            replay_images_wrist = []  # List to store images for replay video

            while step < max_num_steps + num_steps_wait:
                try:
                    
                    if step < num_steps_wait:
                        # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                        # and we need to wait for them to fall

                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION) # Dummy action to wait for objects to stabilize
                        step += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    # Save preprocessed image for replay video
                    replay_images_agent.append(img)
                    replay_images_wrist.append(wrist_img)

                    if not action_plan:
                        # Get action plan from OpenPI server
                        # Previous acton chunk is finished, so we can get a new one
                        # TODO: Add support for other action types
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }

                        # Query model to get action
                        #TODO: check the output of the model
                        action_chunk = client.infer(element)["actions"]
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()


                except Exception as e:
                    logging.error(f"Error during evaluation: {e}")
                    break