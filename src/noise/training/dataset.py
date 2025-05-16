import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from glob import glob


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
    def __init__(self, config: dict):
        self.config = config
        self.data_folder = config.get('data_folder', None)
        self.transform = config.get('transform', None)
        self.max_seq_len = config.get('max_seq_len', None)

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        path = self.file_list[idx]
        episode_data = np.load(path, allow_pickle=True)
        
        # episode_data is a list of dicts
        trajectories = []
        for traj in episode_data:
            states = torch.tensor(traj["state"], dtype=torch.float32)         # [T, state_dim]
            actions = torch.tensor(traj["action"], dtype=torch.float32)       # [T, action_dim]
            
            images = traj["image"]                                            # [T, C, H, W] or [T, H, W, C]
            if images.shape[-1] == 3:  # Convert [T, H, W, C] to [T, C, H, W]
                images = images.transpose(0, 3, 1, 2)
            images = torch.tensor(images, dtype=torch.float32) / 255.0

            if self.transform:
                images = torch.stack([self.transform(img) for img in images])  # apply transform per frame

            success = traj.get("success", None)
            if success is not None:
                success = torch.tensor(success, dtype=torch.float32)

            if self.max_seq_len:
                states = states[:self.max_seq_len]
                actions = actions[:self.max_seq_len]
                images = images[:self.max_seq_len]

            traj_tensor = {
                "state": states,
                "action": actions,
                "image": images,
                "success": success,
            }
            trajectories.append(traj_tensor)

        return trajectories  # batch of episodes
    

# pre-processing
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # Imagenet标准均值
        std=[0.229, 0.224, 0.225]
    )
])

# Example usage
if __name__ == "__main__":
    config = {
        "data_folder": "path/to/your/data",
        "transform": image_transform,
        "max_seq_len": 220
    }
    dataset = NoiseDataset(config = config)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
