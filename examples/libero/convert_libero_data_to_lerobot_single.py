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
import pathlib
import imageio
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tensorflow_datasets as tfds
import tyro
import tqdm
import os
import numpy as np


# task_name = "put both the alphabet soup and the tomato sauce in the basket"
# task_name = "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate".replace("_", " ")
# task_name = "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate".replace("_", " ")
task_name = "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate".replace("_", " ")

REPO_NAME = "your_hf_username/full_libero_" + task_name.replace(" ", "_")[:25]
RAW_DATASET_NAMES = [
    # "libero_10_no_noops",
    # "libero_goal_no_noops",
    # "libero_object_no_noops",
    "libero_spatial_no_noops",
]  # For simplicity we will combine multiple Libero datasets into one training dataset

def save_video(frames, video_path, fps=10):
    pathlib.Path(video_path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(video_path, [np.asarray(frame) for frame in frames], fps=fps)

def main(data_dir: str, *, push_to_hub: bool = False):
    # Clean up any existing dataset in the output directory
    output_path = LEROBOT_HOME / REPO_NAME
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

    # Loop over raw Libero datasets and write episodes to the LeRobot dataset
    # You can modify this for your own data format
    for raw_dataset_name in RAW_DATASET_NAMES:
        raw_dataset = tfds.load(raw_dataset_name, data_dir=data_dir, split="train")
        total_episodes = 0

        for i, episode in enumerate(tqdm.tqdm(raw_dataset, desc=f"Processing {raw_dataset_name}")):
            frames = []
            episode_name = f"{raw_dataset_name}_episode_{i}.mp4"
            output_dir = "videos"
            video_path = os.path.join(output_dir, episode_name)
            for step in episode["steps"].as_numpy_iterator():
                if step["language_instruction"].decode() != task_name:
                    break
                dataset.add_frame(
                    {
                        "image": step["observation"]["image"],
                        "wrist_image": step["observation"]["wrist_image"],
                        "state": step["observation"]["state"],
                        "actions": step["action"],
                    }
                )
                frames.append(step["observation"]["image"])  # Assuming the image is stored here
            
            if step["language_instruction"].decode() == task_name:
                print(step["language_instruction"].decode())
                dataset.save_episode(task=step["language_instruction"].decode())
                save_video(frames, video_path, 10)
                total_episodes += 1
            # if total_episodes >= 4:
            #     break

    # Consolidate the dataset, skip computing stats since we will do that later
    dataset.consolidate(run_compute_stats=False)

    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
