# noise/utils/utils.py

import os
import json
import torch
import wandb
import argparse
from typing import Dict, Any, List

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


def collate_multimodal_batch(batch_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function to combine a list of data point dictionaries
    into a single batch dictionary for Actor-Critic training.
    """
    # This function takes a list of dicts and turns it into a dict of lists/tensors
    
    # Batch tensors by stacking them
    batched_data = {
        # Context items
        'state_action_history': torch.stack([d['context']['state_action_history'] for d in batch_list]),
        'image': torch.stack([d['context']['image'] for d in batch_list]),
        
        # Critic target
        'return_for_critic': torch.stack([d["return_for_critic"] for d in batch_list]),
        
        # Actor data
        'reward': torch.stack([d["reward"] for d in batch_list]),
        'done': torch.stack([d["done"] for d in batch_list]),
        'mask_action': torch.stack([d["mask_action"] for d in batch_list]),
        'old_log_prob': torch.stack([d["old_log_prob"] for d in batch_list]),
    }
    
    # Keep text as a list of strings
    texts = [d['context']['text'] for d in batch_list]
    
    # We will format this into the final structure needed by the training script
    final_batch = {
        "context": {
            "state_action_history": batched_data["state_action_history"],
            "image": batched_data["image"],
            "text_instruction": texts
        },
        "returns_for_critic": batched_data["return_for_critic"],
        "rewards": batched_data["reward"],
        "dones": batched_data["done"],
        "mask_actions": batched_data["mask_action"],
        "old_log_probs": batched_data["old_log_prob"]
    }

    return final_batch