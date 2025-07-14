# scripts/train_actor_offline.py

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json

from src.noise.model.noise_fusion import FusionNoiseModel
from src.noise.model.valuenet import ValueNetwork, compute_gae_and_returns
from src.noise.trpo_updater import trpo_step
# Assume you have your TrajectoryDataset and collate function
from scripts.critic_training import TrajectoryDataset, collate_multimodal_batch
from torch.utils.data import DataLoader

# --- Configuration ---
def get_config():
    # Load your main config file
    # For now, let's define it here for simplicity
    return {
        "actor_checkpoint_path": "path/to/actor/checkpoint.pth", # Optional
        "critic_checkpoint_path": "checkpoints/critic_offline/critic_best.pth", # Required
        "data_source_path": "data/libero/noise/libero_spatial_no_noops",
        "save_dir": "checkpoints/actor_trpo",
        # Model config
        'state_dim': 8, 'action_dim': 7,
        'gru_hidden_dim': 256, 'n_heads': 8,
        'clip_model_path': "/path/to/your/clip-model-local", # Use local path
        # TRPO Hyperparameters
        "max_kl": 0.01,
        "damping": 0.1,
        # Training config
        "epochs": 10,
        "batch_size": 256, # TRPO often works better with larger batches
        "history_len": 10,
        # GAE config
        "gamma": 0.99,
        "gae_lambda": 0.95,
    }

def train_actor_with_trpo(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(config["save_dir"]).mkdir(parents=True, exist_ok=True)

    # --- 1. Load Models ---
    actor = FusionNoiseModel(config).to(device)
    critic = ValueNetwork(config).to(device)
    
    print(f"Loading Critic from: {config['critic_checkpoint_path']}")
    critic.load_state_dict(torch.load(config['critic_checkpoint_path'], map_location=device))
    critic.eval() # Critic is fixed during actor update

    if config.get("actor_checkpoint_path"):
        print(f"Loading Actor from: {config['actor_checkpoint_path']}")
        actor.load_state_dict(torch.load(config['actor_checkpoint_path'], map_location=device))
    
    # --- 2. Load Data ---
    # We will process all data at once for this TRPO implementation
    trajectory_files = list(Path(config["data_source_path"]).rglob("*.npy"))
    dataset = TrajectoryDataset(trajectory_files, config, history_len=config["history_len"])
    # We use a dataloader to prepare batches, but TRPO will process a large chunk.
    loader = DataLoader(dataset, batch_size=len(dataset), collate_fn=collate_multimodal_batch)
    
    # --- 3. Prepare Data for TRPO ---
    print("Preparing data for TRPO update...")
    data_batch = next(iter(loader))
    
    # Move all context data to device
    context = data_batch['context']
    states_for_trpo = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in context.items()}
    
    rewards = data_batch['reward'].to(device)
    # NOTE: Your dataset needs to also provide 'done' flags for GAE
    dones = data_batch['done'].to(device) 
    
    # --- 4. Compute Advantages using the Critic ---
    print("Computing values and advantages with the Critic...")
    with torch.no_grad():
        # This is a simplification. For recurrent models, you need to evaluate values sequentially.
        # For a batch implementation, you'd process each trajectory separately.
        # Let's assume a simplified batch processing for now.
        values, _ = critic.forward(**states_for_trpo)
        
        # We need values for T+1 steps. This requires careful handling of trajectory boundaries.
        # For now, let's use a placeholder for the next state value.
        values_for_gae = torch.cat([values, values[-1:]], dim=0) # Simple approximation
        
        advantages, _ = compute_gae_and_returns(rewards, values_for_gae, dones, config["gamma"], config["gae_lambda"])
        # Normalize advantages for stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # --- 5. Get old log probabilities from the initial actor ---
    print("Computing old log probabilities...")
    with torch.no_grad():
        dist_old, _ = actor.get_distribution(**states_for_trpo)
        # Your data should contain the 'mask' that was actually taken
        actions_taken = data_batch['mask'].to(device)
        old_log_probs = dist_old.log_prob(actions_taken)

    # --- 6. Perform TRPO Update ---
    print("Performing TRPO step...")
    trpo_step(
        actor=actor,
        states=states_for_trpo,
        advantages=advantages,
        old_log_probs=old_log_probs,
        max_kl=config["max_kl"],
        damping=config["damping"]
    )
    
    # --- 7. Save updated actor ---
    save_path = Path(config["save_dir"]) / "actor_updated.pth"
    torch.save(actor.state_dict(), save_path)
    print(f"✅ Actor successfully updated and saved to {save_path}")

if __name__ == "__main__":
    CONFIG = get_config()
    train_actor_with_trpo(CONFIG)