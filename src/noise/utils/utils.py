import os
import json
import torch
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

# in data_utils.py or at the top of train_critic_offline.py
def collate_multimodal_batch(batch_list):
    # batch_list is a list of dicts from your dataset's __getitem__
    elem = batch_list[0]
    # This is a simplified collate function
    return {
        'context': {
            'state_action_history': torch.stack([d['context']['state_action_history'] for d in batch_list]),
            'image': torch.stack([d['context']['image'] for d in batch_list]),
            'text': [d['context']['text'] for d in batch_list] # Keep text as a list of strings
        },
        'return': torch.stack([d['return'] for d in batch_list])
    }