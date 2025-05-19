"""
Script to convert both Hugging Face 'physical-intelligence/libero' dataset
and local .npy files into a single LeRobot dataset.
"""

import shutil
import os
import pathlib
import h5py
import math
import numpy as np

from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tyro
import tensorflow_datasets as tfds

# For Hugging Face dataset loading
from datasets import load_dataset
import PIL.Image # To check image types if necessary

# --- Configuration ---

'''
Specify which configuration/subset of libero you want, e.g., 'libero_spatial', 'libero_10', etc.
You might need to inspect the dataset on Hugging Face to see available configs/subsets.
If the dataset is simple and has no configs, you might not need `name`.
Example: HUGGINGFACE_DATASET_CONFIG = "libero_spatial" # Or one of its sub-components like "libero_spatial-STEPHEN_PICK_UP_THE_BLACK_PENCIL_AND_PUT_IT_INTO_THE_TRASH_BIN"
For libero, it seems the structure is by task suites and then individual tasks.
Let's assume you want to load a specific split and perhaps a subset if available.
The dataset viewer (https://huggingface.co/datasets/physical-intelligence/libero/viewer/default/train)
shows columns like 'episode_metadata', 'steps'. 'steps' is a list of dictionaries.
Each step has 'action', 'discount', 'is_first', 'is_last', 'is_terminal', 'language_embedding',
'language_instruction', 'observation', 'reward'.
'observation' itself is a dict with 'agentview_image', 'robot0_eef_pos', etc.
'''

LEROBOT_REPO_NAME = "lerobot_noise"
HF_DATASET_NAME = "physical-intelligence/libero"

# RLDS Data Source Configuration (from openvla/modified_libero_rlds or similar)
RLDS_DATASET_NAMES = [
    # "libero_10",
    # "libero_goal",
    # "libero_object",
    "libero_spatial",
]

# --- Feature definition - MUST BE CONSISTENT for both datasets ---
# LeRobot Feature Definition (should match both RLDS and your .npy structure)
# Based on your provided script:
LEROBOT_FEATURES = {
    "image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
    "wrist_image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
    "state": {"dtype": "float32", "shape": (8,), "names": ["state"]}, # e.g., 3 pos, 3 axis-angle, 2 gripper
    "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
}

def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
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



def main(
    rlds_data_dir: str = "/workspace/datasets",
    npy_data_dir: str = "data/libero/npy", 
    output_path: str = "/workspace/data", 
    push_to_hub: bool = False
    ):
    # Clean up any existing dataset in the output directory
    output_path = pathlib.Path(output_path) / LEROBOT_REPO_NAME
    print(f"Output path: {output_path}")

    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=output_path,
        robot_type="panda", # Adjust if needed
        fps=10, # Adjust if needed
        features=LEROBOT_FEATURES,
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # --- 1. Process RLDS Libero datasets ---
    print(f"\nProcessing RLDS Libero datasets from: {rlds_data_dir}")
    total_episode_processed = 0
    for dataset_name in RLDS_DATASET_NAMES:
        print(f"\nProcessing dataset: {dataset_name}")

        # Load the h5py file
        dataset_path = os.path.join(rlds_data_dir, dataset_name)
        print(f"Dataset path: {dataset_path}")
        for file_name in os.listdir(dataset_path):
            if not file_name.endswith(".hdf5"):
                continue
            dataset_file_path = os.path.join(dataset_path, file_name)

            with h5py.File(dataset_file_path, "r") as f:
                # Get language instruction from file name
                # pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo.hdf5
                language_instruction = file_name.split(".")[0]

                # get data from the file
                for episode_idx in f["data"]:
                    episode = f["data"][episode_idx]
                    obs = episode["obs"]

                    actions = np.array(episode["actions"]) # Should be (episode_length, action_dim)

                    # Process each step in the episode
                    frame_num = 0

                    for step_idx in range(actions.shape[0]):
                        
                        # image size: (128, 128, 3), type numpy.uint8
                        image = obs["agentview_rgb"][step_idx]
                        wrist_image = obs["eye_in_hand_rgb"][step_idx]
                        # Resize images to (256, 256, 3) if necessary
                        if image.shape != (256, 256, 3):
                            image = np.array(PIL.Image.fromarray(image).resize((256, 256)))
                        if wrist_image.shape != (256, 256, 3):
                            wrist_image = np.array(PIL.Image.fromarray(wrist_image).resize((256, 256)))
                        ee_states = np.array(obs["ee_states"][step_idx])
                        gripper_state = np.array(obs["gripper_states"][step_idx])

                        state = np.concatenate([ee_states, gripper_state])  # (3+3+2=8)

                        # Prepare data for dataset.add_frame
                        frame_data = {
                            "image": image,       # Should be HxWxC numpy array
                            "wrist_image": wrist_image, # Should be HxWxC numpy array
                            "state": state,       # Should be (state_dim,) numpy array
                            "actions": actions[step_idx],                   # Should be (action_dim,) numpy array
                        }

                        dataset.add_frame(frame_data)

                        frame_num += 1
                    
                    if frame_num > 0:
                        dataset.save_episode(task=language_instruction)
                        total_episode_processed += 1

    print(f"\nTotal RLDS episodes processed: {total_episode_processed}")
    print(f"RLDS dataset processing completed.")

    # --- 2. Process local .npy files ---
    print(f"\nProcessing local .npy files from: {npy_data_dir}")
    source_path = pathlib.Path(npy_data_dir)
    npy_episode_files = sorted(list(source_path.glob("Episode_*.npy")))

    totol_episode_processed = 0
    if not npy_episode_files:
        print(f"No .npy files found in {source_path} matching the pattern.")

    else:
        print(f"Found {len(npy_episode_files)} episode files.")
        for npy_file_path in npy_episode_files:
            print(f"Processing episode: {npy_file_path.name}")
            try:
                # Load the .npy file
                episode_steps = np.load(npy_file_path, allow_pickle=True)
                if not isinstance(episode_steps, (list, np.ndarray)) or len(episode_steps) == 0:
                    print(f"  Skipping empty or invalid episode: {npy_file_path.name}")
                    continue

                # If it's an ndarray of objects (dicts), convert to list
                if isinstance(episode_steps, np.ndarray) and episode_steps.dtype == 'object':
                    episode_steps = list(episode_steps)

                last_npy_instruction_str = "libero_spatial"
                frame_num = 0
                # Process each step in the episode
                for step_idx, step_data in enumerate(episode_steps):
                    # Ensure step_data is a dictionary
                    if not isinstance(step_data, dict):
                        print(f"  Skipping invalid step (not a dict) in {npy_file_path.name} at step {step_idx}")
                        continue

                    # Prepare data for dataset.add_frame
                    frame_data = {
                        "image": step_data["observation"]["image"],       # Should be HxWxC numpy array
                        "wrist_image": step_data["observation"]["wrist_image"], # Should be HxWxC numpy array
                        "state": step_data["observation"]["state"],       # Should be (state_dim,) numpy array
                        "actions": step_data["action"],                   # Should be (action_dim,) numpy array
                    }

                    
                    dataset.add_frame(frame_data)

                    # Get the language instruction
                    if "language_instruction" in step_data:
                        last_npy_instruction_str = step_data["language_instruction"]
                        if isinstance(last_npy_instruction_str, bytes):
                            last_npy_instruction_str = last_npy_instruction_str.decode('utf-8', 'ignore')
                        elif isinstance(last_npy_instruction_str, str):
                            last_npy_instruction_str = last_npy_instruction_str
                        else:
                            last_npy_instruction_str = str(last_npy_instruction_str)
                            
                    frame_num += 1

                if frame_num == 0:
                    print(f"  Skipping empty episode: {npy_file_path.name}")
                    continue

                dataset.save_episode(task=last_npy_instruction_str)
                totol_episode_processed += 1

            except Exception as e:
                print(f"Error processing {npy_file_path.name}: {e}")
                continue
            
        print(f"Processed {totol_episode_processed} episodes from local .npy files.")

    print(f"Local .npy file processing completed.")

    # --- 3. Consolidate the dataset ---
    print(f"Consolidating the dataset...")
    dataset.consolidate(run_compute_stats=True)
    print(f"Dataset consolidation completed.")

    # --- 4. Optionally push to the Hugging Face Hub ---
    if push_to_hub:
        print(f"Pushing dataset to Hugging Face Hub...")
        dataset.push_to_hub(
            tags=["libero", "panda", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )
        print(f"Dataset pushed to Hugging Face Hub.")
    else:
        print(f"Dataset not pushed to Hugging Face Hub.")

if __name__ == "__main__":
    # Example usage:
    # python scripts/libero_dataset.py --rlds_data_dir /path/to/rlds --npy_data_dir /path/to/npy --output_path /path/to/output --push_to_hub True
    tyro.cli(main)