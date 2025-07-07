import collections
import dataclasses
import logging
import math
import torch
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

import matplotlib.pyplot as plt

import json
with open("./configs/noise_model_config.json", "r") as f:
    config = json.load(f)

# Choose model type: 'dummy' or 'history' or 'vision'
model_type = config.get("model", "dummy")
if model_type == "dummy":
    from noise.model.noise_dummy import DummyNoiseModel
    noise_model = DummyNoiseModel(config["dummy"])
elif model_type == "history":
    from noise.model.noise_history import HistoryNoiseModel
    noise_model = HistoryNoiseModel(config["history"])
elif model_type == "vision":
    from noise.model.noise_vision import VisionNoiseModel
    noise_model = VisionNoiseModel(config["vision"])
    # load checkpoint if available
    if config["vision"].get("checkpoints", None) is not None:
        chkp = torch.load(config["vision"]["checkpoints"], map_location="cpu")
        print(f"Loading noise model checkpoint from {config['vision']['checkpoints']}...")
        noise_model.load_state_dict(chkp, strict=False)
        print("Checkpoint loaded successfully.")
else:
    raise ValueError(f"Unsupported noise model type: {model_type}")

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data

# Global variables
success_rate_list_per_task: list = []  # List to store success rates of each task
curr_offset = 40 # Offset for episode numbering


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
    num_trials_per_task: int = 10  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos_250617"  # Path to save videos

    img_out_path: str = "data/libero/images"  # Path to save images

    data_out_path: str = "data/libero/npy"  # Path to save data
    
    noise_out_path: str = "data/libero/noise"  # Path to save noise data

    seed: int = 7  # Random Seed (for reproducibility)

    save_data: bool = True  # Save data


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


def eval_libero(args: Args):
    """
    Evaluate the LIBERO environment with noise injection and replanning.
    :param args: Arguments for evaluation.
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Set random seed for reproducibility
    np.random.seed(args.seed)

    # Create output directories if they don't exist
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.img_out_path).mkdir(parents=True, exist_ok=True)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    max_num_steps = noise_model.episode_length  # Maximum number of steps per trial

    # configure for steps to wait
    num_steps_wait = args.num_steps_wait  # Number of steps to wait for objects to stabilize in sim

    # Initialize OpenPI client
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Initialize data writer
    data_path = pathlib.Path(args.data_out_path) / f"{args.task_suite_name}_no_noops"
    data_path.mkdir(parents=True, exist_ok=True)

    noise_out_path = pathlib.Path(args.noise_out_path) / f"{args.task_suite_name}_no_noops"
    noise_out_path.mkdir(parents=True, exist_ok=True)

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

            episode = []  # List to store data for RLDS episode
            noise_episode = []  # List to store data for noise model episode

            noise_model.reset()  # Reset noise model for new episode
            
            collect_threshold = noise_model.collect_threshold
            is_collecting = False

            logging.info(f"Starting episode {task_episodes+1}...")

            done = False

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
                        action_chunk = client.infer(element)["actions"]
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    # TODO: collect obs, action, reward, done, info for each step
                    action = action_plan.popleft()
                    state_vec = np.concatenate((
                        obs["robot0_eef_pos"],
                        quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    ))

                    # add noise to the action
                    img_flat = img.astype(np.float32) / 255.0
                    delta = noise_model.sample(state=state_vec, action=action, image=img_flat)
                    if model_type == "history" or model_type == "vision":
                        delta_np = delta.detach().cpu().numpy()
                    else: # dummy noise model
                        delta_np = delta.numpy() if hasattr(delta, "numpy") else delta

                    # check and squeeze delta
                    if delta_np.ndim > 1:
                        delta_np = delta_np.squeeze()
                    disturbed_action = np.clip(np.array(action) + delta_np, -1.0, 1.0)

                    # Log the action and delta
                    # env.step(disturbed_action.tolist())  # Apply the action to the environment

                    # Execute action in environment
                    obs, reward, done, info = env.step(disturbed_action.tolist())
                    # obs, reward, done, info = env.step(action.tolist())
                    
                    adv_reward = noise_model.compute_reward(
                        success=bool(info.get("success", False)),
                        delta=delta if isinstance(delta, torch.Tensor) else torch.tensor(delta)
                    )

                    # Write data step
                    if is_collecting:
                        episode_step = {
                            "observation": {
                                "image": img,
                                "wrist_image": wrist_img,
                                "state": np.concatenate(
                                    (
                                        obs["robot0_eef_pos"],
                                        quat2axisangle(obs["robot0_eef_quat"]),
                                        obs["robot0_gripper_qpos"],
                                    )
                                ),
                            },
                            "action": disturbed_action,
                            "reward": reward,
                            "discount": 1.0,
                            "language_instruction": str(task_description),
                        }
                        episode.append(episode_step)

                    if not is_collecting and np.linalg.norm(delta_np) > collect_threshold:
                        logging.info(f"Collecting data at step {step + 1} with delta {delta_np} and threshold {collect_threshold}")
                        is_collecting = True


                    # Write noise model step
                    noise_step = {
                        "state": state_vec,
                        "action": action,
                        "image": img_flat,
                        "delta": delta_np,
                        "reward": adv_reward,
                        "success": done,
                    }
                    noise_episode.append(noise_step)

                    if done:
                        task_successes += 1
                        total_successes += 1
                        break

                    # increment step
                    step += 1

                except Exception as e:
                    logging.error(f"Error during evaluation: {e}")
                    break

            # Finalize and close RLDS episode

            task_episodes += 1
            total_episodes += 1
            task_segment = task_description.replace(" ", "_")

            # Save the succeded episode
            if done:
                # Save video of the episode
                video_path = pathlib.Path(args.video_out_path)/ f"0{task_id}_{task_segment}" / f"Episode_{task_episodes}.mp4"
                video_path.parent.mkdir(parents=True, exist_ok=True)
                imageio.mimwrite(
                    video_path,
                    [np.asarray(x) for x in replay_images_agent],
                    fps=10,
                )
                logging.info(f"Saved video to {video_path}")

                # Save npy data
                if len(episode) > 0 and args.save_data:
                    # Save npy data
                    npy_path = pathlib.Path(args.data_out_path) / f"Episode_{task_id}_{curr_offset+task_episodes}.npy"
                    np.save(npy_path, episode)
                    logging.info(f"Saved episode to {npy_path}")
                elif len(episode) == 0:
                    logging.warning(f"Episode {task_episodes} is empty, not saving.")
                    

            else:
                # save video with failure tag
                video_path = pathlib.Path(args.video_out_path)/ f"0{task_id}_{task_segment}" / f"Episode_{task_episodes}_failure.mp4"
                video_path.parent.mkdir(parents=True, exist_ok=True)
                imageio.mimwrite(
                    video_path,
                    [np.asarray(x) for x in replay_images_agent],
                    fps=10,
                )
            
            # save noise model episode
            if len(noise_episode) > 0 and args.save_data:
                noise_npy_path = pathlib.Path(args.noise_out_path) / f"Episode_{task_id}_{curr_offset+task_episodes}.npy"
                noise_npy_path.parent.mkdir(parents=True, exist_ok=True)
                # Save noise model episode
                np.save(noise_npy_path, noise_episode)
                logging.info(f"Saving noise model episode to {noise_npy_path}")
            elif len(noise_episode) == 0:
                logging.warning(f"Noise model episode {task_episodes} is empty, not saving.")
                

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

    # log final results
    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")

    # Plot success rate per task
    plt.plot(success_rate_list_per_task)
    plt.xlabel("Task ID")
    plt.ylabel("Success Rate")
    plt.title("Success Rate per Task")
    plt.savefig(pathlib.Path(args.img_out_path) / f"success_rate_per_task_sr{float(total_successes) / float(total_episodes)}.png")

    # Save all episodes to TFRecord
    logging.info(f"Saving all episodes to {data_path}")



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)