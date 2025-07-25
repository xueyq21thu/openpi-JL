import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from glob import glob

from typing import Dict, Any, List
from pathlib import Path


"""
This script implements a dataset class for training noise models.

The dataset loads and processes trajectory data containing states, actions, images and success flags.
Each trajectory consists of:
- states: Robot state vectors over time [T, state_dim] 
- actions: Robot action vectors over time [T, action_dim]
- images: Camera images over time [T, C, H, W]
- success: Binary success flag for the trajectory

The data is loaded from .npy files containing lists of trajectory dictionaries.
Images are normalized to [0,1] range and can be optionally transformed.
Sequence length can be optionally capped using max_seq_len.

Classes:
    NoiseDataset: PyTorch Dataset class for noise model training data

Example usage:
    dataset = NoiseDataset(data_folder='path/to/data',
                          transform=transforms.Compose([...]),
                          max_seq_len=100)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
"""

class NoiseDataset(Dataset):
    '''
    Noise dataset for noise model training and testing.
    Args:
        data_folder (str): Path to the folder containing noise data files.
        transform (callable, optional): Optional transform to be applied on a sample.
        max_seq_len (int, optional): Maximum sequence length for the data.
    '''
    def __init__(self, config: dict, transform=None):
        self.config = config
        self.data_folder = config.get('data_dir', None)
        self.transform = transform
        self.max_seq_len = config.get('max_seq_len', None)

        self.file_list = []
        print(f"Loading data from {self.data_folder}")
        if self.data_folder:
            # Load all .npy files from the data folder
            self.file_list = glob(os.path.join(self.data_folder, '*.npy'))
            if not self.file_list:
                raise ValueError(f"No .npy files found in {self.data_folder}")
        else:
            raise ValueError("data_folder is not specified in the config")
        print(f"Loaded {len(self.file_list)} files from {self.data_folder}")


    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        path = self.file_list[idx]
        episode_data = np.load(path, allow_pickle=True)

        # lists of noise model data
        state_list = []
        action_list = []
        image_list = []
        delta_list = []
        reward_list = []
        success_list = []
                
        # episode_data is a list of dicts        
        for step in episode_data:
            state_list.append(step["state"])
            action_list.append(step["action"])
            delta_list.append(step["delta"])
            reward_list.append(step["reward"])

            # [H, W, C] -> [C, H, W]
            image = step["image"]
            if image.ndim == 3 and image.shape[-1] == 3:
                image = image.transpose(2, 0, 1)
            if image.ndim == 1:
                image = image.reshape(256, 256, 3)
                image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)  # reshape to [C, H, W]
                
            elif image.ndim == 4 and image.shape[0] == 1:
                image = image.squeeze(0)

            # Check the image size and resize if necessary
            if image.shape[0] != 3 or image.shape[1] != 224 or image.shape[2] != 224:
                # Resize the image to 224x224 if it is not already
                image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)

            image_list.append(image)

            # convert success to float
            success = step["success"]
            if isinstance(success, bool):
                success = float(success)
            success_list.append(success)
        
        # Convert lists to tensors
        states = torch.tensor(np.array(state_list), dtype=torch.float32)         # [T, state_dim]
        actions = torch.tensor(np.array(action_list), dtype=torch.float32)       # [T, action_dim]
        deltas = torch.tensor(np.array(delta_list), dtype=torch.float32)         # [T, action_dim]
        rewards = torch.tensor(np.array(reward_list), dtype=torch.float32)       # [T, action_dim]
        images = torch.tensor(np.array(image_list), dtype=torch.float32) / 255.0  # [T, C, H, W]
        successes = torch.tensor(np.array(success_list), dtype=torch.float32)         # [T, 1]

        # Normalize images
        if self.transform:
            images = torch.stack([self.transform(img) for img in images])
        
        traj_tensor = {
            "state": states,
            "action": actions,
            "delta": deltas,
            "reward": rewards,
            "image": images,
            "success": successes
        }
        # If max_seq_len is specified, truncate the sequences
        if self.max_seq_len:
            traj_tensor["state"] = states[:self.max_seq_len]
            traj_tensor["action"] = actions[:self.max_seq_len]
            traj_tensor["delta"] = deltas[:self.max_seq_len]
            traj_tensor["reward"] = rewards[:self.max_seq_len]
            traj_tensor["image"] = images[:self.max_seq_len]
            traj_tensor["success"] = successes[:self.max_seq_len]

        return traj_tensor  # single episode data


# pre-processing
image_transform = transforms.Compose([
    # transforms.Resize((224, 224)),
    # transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # Imagenet标准均值
        std=[0.229, 0.224, 0.225]
    )
])

class StepNoiseDataset(Dataset):
    '''
    Step-level dataset for noise model training.
    Args:
        config (dict): Configuration dictionary containing data_dir and other parameters.
        transform (callable, optional): Optional transform to be applied on a sample.
    '''
    def __init__(self, config: dict, transform=None):
        self.config = config
        self.data_folder = config.get('data_dir', None)
        self.transform = transform

        self.file_list = glob(os.path.join(self.data_folder, '*.npy'))
        self.step_index = []  # (episode_idx, step_idx)
        self.episode_data = []

        # preprocess the data and create step_index
        for epi_idx, path in enumerate(tqdm(self.file_list, desc="Loading episodes")):
            episode = np.load(path, allow_pickle=True)
            self.episode_data.append(episode)
            for t in range(len(episode)):
                self.step_index.append((epi_idx, t))

    def __len__(self):
        return len(self.step_index)

    def __getitem__(self, idx):
        epi_idx, step_idx = self.step_index[idx]
        step = self.episode_data[epi_idx][step_idx]

        state = torch.tensor(step["state"], dtype=torch.float32)
        action = torch.tensor(step["action"], dtype=torch.float32)
        delta = torch.tensor(step["delta"], dtype=torch.float32)
        reward = torch.tensor(step["reward"], dtype=torch.float32)


        image = step["image"]
        if image.ndim == 3 and image.shape[-1] == 3:
            image = image.transpose(2, 0, 1)

        if image.ndim == 1:
            image = image.reshape(256, 256, 3)
            image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)

        elif image.ndim == 4 and image.shape[0] == 1:
            image = image.squeeze(0)

        if image.shape[1] != 224 or image.shape[2] != 224:
            image = image.transpose(1, 2, 0)
            image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
            image = image.transpose(2, 0, 1)

        image = torch.tensor(image, dtype=torch.float32) / 255.0
        if self.transform:
            image = self.transform(image)
        success = float(step["success"]) if isinstance(step["success"], bool) else step["success"]

        return {
            "state": state,
            "action": action,
            "delta": delta,
            "reward": reward,
            "image": image,
            "success": torch.tensor(success, dtype=torch.float32)
        }


class TrajectoryDataset(Dataset):
    """
    A PyTorch Dataset for loading trajectory data.
    
    This version is finalized to provide all necessary data fields for both
    offline Critic training (discounted returns) and offline Actor training (rewards,
    dones, masks, old log_probs for GAE and policy gradient calculations).
    """

    def __init__(self, trajectory_files: List[Path], config: Dict[str, Any], history_len: int = 10):
        self.config = config
        self.history_len = history_len
        self.data_points = self._process_trajectories(trajectory_files)

    def _process_trajectories(self, trajectory_files: List[Path]) -> List[Dict[str, Any]]:
        all_data_points = []
        print(f"Loading and processing {len(trajectory_files)} trajectories...")

        for file_path in tqdm(trajectory_files, desc="Processing Trajectories"):
            # ... (file loading logic is unchanged) ...
            try:
                episode_data = np.load(file_path, allow_pickle=True)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue

            # --- DATA PREPARATION ---
            rewards = [step['reward'] for step in episode_data]
            # Pre-calculate discounted returns, which are the targets for the Critic
            returns_for_critic = self._compute_discounted_returns(rewards, self.config.get('gamma', 0.99))

            # --- UNROLL TRAJECTORY ---
            for t in range(len(episode_data)):
                current_step = episode_data[t]
                
                # --- State-Action History ---
                start_idx = max(0, t - self.history_len + 1)
                history_steps = episode_data[start_idx : t + 1]
                states = np.array([step['state'] for step in history_steps], dtype=np.float32)
                actions = np.array([step['action'] for step in history_steps], dtype=np.float32)
                state_action_hist = np.concatenate([states, actions], axis=-1)
                
                if len(state_action_hist) < self.history_len:
                    padding_needed = self.history_len - len(state_action_hist)
                    padding = np.zeros((padding_needed, state_action_hist.shape[1]), dtype=np.float32)
                    state_action_hist = np.vstack([padding, state_action_hist])

                # --- Image Processing ---
                current_image_np = current_step['image'] # Assuming (H, W, C) uint8
                if current_image_np.shape != (256, 256, 3):
                    current_image_np = cv2.resize(current_image_np, (256, 256))
                
                # Convert to (C, H, W) tensor and normalize
                image_tensor = torch.from_numpy(current_image_np).float().permute(2, 0, 1) / 255.0

                # --- FINAL DATA POINT DICTIONARY ---
                # This dictionary now contains EVERYTHING needed for training.
                all_data_points.append({
                    "context": {
                        "state_action_history": torch.from_numpy(state_action_hist).float(),
                        "image": image_tensor,
                        "text": current_step.get('language_instruction', "Default Task Instruction")
                    },
                    # Target for the Critic
                    "return_for_critic": returns_for_critic[t],
                    
                    # Data needed for Actor's GAE calculation and update
                    "reward": torch.tensor(current_step.get('reward', 0.0), dtype=torch.float32),
                    "done": torch.tensor(current_step.get('done', False), dtype=torch.float32),
                    "mask_action": torch.tensor(current_step.get('mask', 0.0), dtype=torch.float32),
                    "old_log_prob": torch.tensor(current_step.get('log_prob', 0.0), dtype=torch.float32)
                })
        
        print(f"Successfully processed {len(all_data_points)} data points.")
        return all_data_points
    
    # ... (__compute_discounted_returns, __len__, __getitem__ are unchanged) ...
    @staticmethod
    def _compute_discounted_returns(rewards: List[float], gamma: float) -> torch.Tensor:
        returns = []
        future_return = 0.0
        for r in reversed(rewards):
            future_return = r + gamma * future_return
            returns.append(future_return)
        return torch.tensor(list(reversed(returns)), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.data_points)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data_points[idx]
    

# Example usage
if __name__ == "__main__":
    config = {
        "data_folder": "path/to/your/data",
        "transform": image_transform,
        "max_seq_len": 220
    }
    dataset = NoiseDataset(config = config)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
