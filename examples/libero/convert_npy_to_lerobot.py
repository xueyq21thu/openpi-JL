"""
Minimal example script for converting a dataset to LeRobot format.

We use the Libero dataset (stored in RLDS) for this example, but it can be easily
modified for any other data you have saved in a custom format.

Usage:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

Note: to run the script, you need to install tensorflow_datasets:
`uv pip install tensorflow tensorflow_datasets`

You can download the raw Libero datasets from https://huggingface.co/datasets/openvla/modified_libero_rlds
The resulting dataset will get saved to the $LEROBOT_HOME directory.
Running this conversion script will take approximately 30 minutes.
"""

import shutil
import os
import pathlib # For path manipulation
import numpy as np # For loading .npy files

from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
# import tensorflow_datasets as tfds
import tyro

REPO_NAME = "lerobot_npy"  # Name of the output dataset, also used for the Hugging Face Hub
RAW_DATASET_NAMES = [
    "libero_spatial_no_noops",
]  # This task suite is used for the first stage of the training pipeline


def main(npy_data_dir: str = "data/libero/npy", *, push_to_hub: bool = False):
    # Clean up any existing dataset in the output directory
    output_path = LEROBOT_HOME / REPO_NAME
    print(f"Output path: {output_path}")
    
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="panda",
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # load npy files
    source_path = pathlib.Path(npy_data_dir)
    print(f"Scanning for .npy episode files in: {source_path}")

    episode_files = sorted(list(source_path.glob("Episode_*.npy"))) # Adjust glob pattern if needed
    if not episode_files:
        print(f"No .npy files found in {source_path} matching the pattern.")
        return

    print(f"Found {len(episode_files)} episode files.")

    for npy_file_path in episode_files:
        print(f"Processing episode: {npy_file_path.name}")
        try:
            # episode_data is a list of step dictionaries
            episode_steps = np.load(npy_file_path, allow_pickle=True)
            if not isinstance(episode_steps, (list, np.ndarray)) or len(episode_steps) == 0:
                print(f"  Skipping empty or invalid episode: {npy_file_path.name}")
                continue

            # If it's an ndarray of objects (dicts), convert to list
            if isinstance(episode_steps, np.ndarray) and episode_steps.dtype == 'object':
                episode_steps = list(episode_steps)

            last_instruction = "libero_spatial" # Default task name

            for step_idx, step_data in enumerate(episode_steps):
                # Ensure step_data is a dictionary
                if not isinstance(step_data, dict):
                    print(f"  Skipping invalid step (not a dict) in {npy_file_path.name} at step {step_idx}")
                    continue

                # Prepare data for dataset.add_frame
                # Ensure keys match what you defined in LeRobotDataset.create features
                # And that the shapes/dtypes are compatible.
                # LeRobot's "image" dtype will handle PIL images or numpy arrays.
                # Your .npy files store images as numpy arrays (uint8, 0-255).
                
                # Check if image shape matches expected (after resize_with_pad)
                # Your main.py saves images that are (resize_size, resize_size, 3)
                # The LeRobot features should match this. I've set it to 224x224x3, adjust if needed.

                frame_to_add = {
                    "image": step_data["observation"]["image"],       # Should be HxWxC numpy array
                    "wrist_image": step_data["observation"]["wrist_image"], # Should be HxWxC numpy array
                    "state": step_data["observation"]["state"],       # Should be (state_dim,) numpy array
                    "actions": step_data["action"],                   # Should be (action_dim,) numpy array
                }
                
                # if "language_instruction" in features: # If you added it
                #    frame_to_add["language_instruction"] = step_data.get("language_instruction", "N/A")

                dataset.add_frame(frame_to_add)
                
                # Get language instruction for saving the episode
                if "language_instruction" in step_data:
                    instruction_bytes = step_data["language_instruction"]
                    if isinstance(instruction_bytes, bytes):
                        last_instruction = instruction_bytes.decode('utf-8', 'ignore')
                    elif isinstance(instruction_bytes, str):
                        last_instruction = instruction_bytes
                    else:
                        last_instruction = str(instruction_bytes)


            if len(episode_steps) > 0: # Only save if episode had steps
                dataset.save_episode(task=last_instruction)
            else:
                print(f"  Not saving episode {npy_file_path.name} as it had no valid steps processed.")

        except Exception as e:
            print(f"  Error processing episode {npy_file_path.name}: {e}")
            # Optionally, clean up potentially partially written episode from dataset if error handling is complex
            # For now, LeRobotDataset might handle this internally or just move to next.
            continue
    # --- MODIFICATION END ---

    # Consolidate the dataset, skip computing stats since we will do that later
    # (or compute them now if you prefer: run_compute_stats=True)
    dataset.consolidate(run_compute_stats=False) 

    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda", "custom_npy"], # Adjusted tags
            private=False,
            push_videos=True, # This might require ffmpeg and videos to be present or generated
            license="apache-2.0",
        )

if __name__ == "__main__":
    tyro.cli(main)