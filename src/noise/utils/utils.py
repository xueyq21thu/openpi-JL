import os
import json
import wandb
import argparse

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def parse_args():
    parser = argparse.ArgumentParser(description="Noise Model Pretraining Script")
    parser.add_argument('--config', type=str, default='configs/noise_train.json', help='Path to json config file')
    parser.add_argument('--project', type=str, default='CAL', help='wandb project name')
    parser.add_argument('--name', type=str, default='baseline', help='wandb run name')
    return parser.parse_args()

def init_wandb(args, config):
    wandb.init(project=args.project, name=args.name, config=config, save_code=True)
    wandb.config.update(config)