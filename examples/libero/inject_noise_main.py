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

# data collection
import rlds
from rlds import writer
import tensorflow as tf
import numpy as np
import pathlib

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

    data_out_path: str = "data/libero/rlds"  # Path to save data

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
    pathlib.Path(args.data_out_path).mkdir(parents=True, exist_ok=True)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    max_num_steps = 400  # Maximum number of steps per trial

    # configure for steps to wait
    num_steps_wait = args.num_steps_wait  # Number of steps to wait for objects to stabilize in sim

    # Initialize OpenPI client
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Initialize data writer
    data_writer = writer.Writer(
        pathlib.Path(args.data_out_path) / f"libero_data_{args.task_suite_name}.tfrecord"
    )

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

        # # Init Episode
        # ep = data_writer.write_episode()

        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")
            
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

            step_data = []  # List to store data for RLDS episode

            logging.info(f"Starting episode {task_episodes+1}...")
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

                    # TODO: collect obs, action, reward, done, info for each step
                    action = action_plan.popleft()

                    disturbed_action = action.copy()

                    # Execute action in environment
                    disturbed_action = sample_noise(
                        step,
                        args.noise_insert_step,
                        args.noise_last_step,
                        action,
                        args.noise_type,
                        args.noise_scale,
                    )
                    
                    # Execute action in environment
                    obs, reward, done, info = env.step(disturbed_action.tolist())

                    # TODO: add noise injection flag to the record
                    # Write RLDS step
                    step_data.append(
                        {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "action": disturbed_action,
                            "reward": reward,
                            "done": done,
                            "info": info,
                            "language_instruction": str(task_description),
                      }
                    )

                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    step += 1

                except Exception as e:
                    logging.error(f"Error during evaluation: {e}")
                    break

            # Finalize and close RLDS episode
            ep.end_of_episode()

            task_episodes += 1
            total_episodes += 1

            # Save the succeded episode
            if done:
                # Save video of the episode
                task_segment = task_description.replace(" ", "_")
                video_path = pathlib.Path(args.video_out_path)/ f"0{task_id}_{task_segment}" / f"Episode_{task_episodes}.mp4"
                imageio.mimwrite(
                    video_path,
                    [np.asarray(x) for x in replay_images_agent],
                    fps=10,
                )
                logging.info(f"Saved video to {video_path}")

                # save the data to TFRecord
                ep = data_writer.write_episode()
                for step_record in step_data:
                    ep.write_step(step_record)
                ep.end_of_episode()

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

        # Stack success rates for each task
        success_rate_list_per_task.append(task_successes / task_episodes)
        logging.info(f"Success rate for task {task_id}: {task_successes / task_episodes}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")

    # Plot success rate per task
    plt.plot(success_rate_list_per_task)
    plt.xlabel("Task ID")
    plt.ylabel("Success Rate")
    plt.title("Success Rate per Task")
    plt.savefig(pathlib.Path(args.img_out_path) / "success_rate_per_task.png")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)