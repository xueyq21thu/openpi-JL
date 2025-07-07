# append the src directory to the path
import sys
sys.path.append("./src")

import argparse
import json
import torch
from torch.utils.data import DataLoader

# import from src
from noise.training.dataset import NoiseDataset, image_transform, StepNoiseDataset
from noise.model.noise_vision import VisionNoiseModel
from noise.training.pretraining import pre_training
from noise.training.postraining import post_training
from noise.utils import utils



def main():
    # Parse arguments
    args = utils.parse_args()

    # Load the training configuration
    training_config = utils.load_config(args.config)
    task = training_config['task']
    training_config = training_config[task]
    model_config = utils.load_config(training_config['model_config'])["vision"]

    # Initialize wandb
    # utils.init_wandb(args, training_config)

    # Load the dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # dataset = NoiseDataset(config = training_config, transform=image_transform)
    dataset = NoiseDataset(config=training_config, transform=image_transform)
    dataloader = DataLoader(dataset, batch_size=training_config['batch_size'], shuffle=True)

    # Create the model
    model = VisionNoiseModel(config=model_config).to(device)

    # task
    if task == "pretraining":
        # Pre-training
        pre_training(model, dataloader, training_config, device)
    elif task == "postraining":
        # Post-training
        post_training(model, dataloader, training_config, device)
    else:
        raise ValueError(f"Unknown task: {task}")



if __name__ == '__main__':
    main()