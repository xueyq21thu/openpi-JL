import argparse
import json
import wandb
import torch
from torch.utils.data import DataLoader

# import from src
from noise.training.dataset import NoiseDataset, image_transform
from noise.model.noise_vision import VisionNoiseModel
from noise.training.pretraining import pre_training

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def parse_args():
    parser = argparse.ArgumentParser(description="Noise Model Pretraining Script")
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--project', type=str, default='noise-pretrain', help='wandb project name')
    parser.add_argument('--name', type=str, default='baseline', help='wandb run name')
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_config(args.config)

    wandb.init(project=args.project, name=args.name, config=config, save_code=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = NoiseDataset(config = config)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    model = VisionNoiseModel(config).to(device)
    pre_training(model, dataloader, config, device)

if __name__ == '__main__':
    main()