# critic_training.py

# Append the src directory to the path to find the 'noise' module
import sys
sys.path.append("./src")

from pathlib import Path
from typing import Dict, Any, List

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
import numpy as np
import wandb

from noise.model.valuenet import ValueNetwork, compute_reward
from noise.utils.utils import collate_multimodal_batch

# ==============================================================================
# SECTION 2: THE TRAINING SCRIPT
# ==============================================================================

def train_critic_from_rollouts(config: Dict[str, Any]):
    """
    Main function to orchestrate the offline training of the ValueNetwork.
    """
    # --- 1. Setup Environment and Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. Load Data ---
    data_path = Path(config["data_source_path"])
    if not data_path.is_dir():
        raise FileNotFoundError(f"Data source path not found: {data_path}")
    
    # Recursively find all .npy files in the specified directory
    trajectory_files = list(data_path.rglob("*.npy"))
    if not trajectory_files:
        raise FileNotFoundError(f"No .npy files found in {data_path}")

    # Instantiate the dataset
    full_dataset = TrajectoryDataset(trajectory_files, config, history_len=config.get('history_len', 10))
    
    # Split into training and validation sets
    val_split = config.get("val_split_ratio", 0.1)
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"\nDataset loaded: {len(full_dataset)} total samples.")
    print(f"  - Training set size: {len(train_dataset)}")
    print(f"  - Validation set size: {len(val_dataset)}")
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True,
        collate_fn=collate_multimodal_batch, num_workers=config.get('num_workers', 4), pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False,
        collate_fn=collate_multimodal_batch, num_workers=config.get('num_workers', 4), pin_memory=True
    )
    
    # --- 3. Initialize Model and Optimizer ---
    critic_model = ValueNetwork(config).to(device)

    checkpoint_path = config.get("save_dir", None)
    # check if a pre-trained model exists
    if checkpoint_path:
        checkpoint_path = Path(checkpoint_path) / "critic_best.pth"
        if checkpoint_path.exists():
            print(f"Loading pre-trained model from {checkpoint_path}")
            critic_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        else:
            print(f"No pre-trained model found at {checkpoint_path}, starting from scratch.")

    optimizer = optim.AdamW(critic_model.parameters(), lr=config.get("lr", 1e-4), weight_decay=1e-3)
    loss_fn = nn.MSELoss()

    # --- 4. Initialize Logging (W&B) ---
    if config.get("use_wandb", False):
        wandb.init(project=config.get("wandb_project", "cal-critic-offline"), config=config)
        wandb.watch(critic_model, log_freq=100)

    # --- 5. Training Loop ---
    best_val_loss = float('inf')
    save_dir = Path(config.get("save_dir", "checkpoints/critic_offline"))
    save_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- Starting Offline Critic Training ---")
    for epoch in range(config.get("epochs", 20)):
        critic_model.train()
        train_loss_total = 0.0
        
        # Training epoch
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} [Train]", leave=False):
            context = batch['context']
            target_returns = batch['return'].to(device).unsqueeze(-1)
            
            # Prepare inputs
            state_hist = context['state_action_history'].to(device)
            image = context['image'].to(device)
            text = context['text']
            
            # Forward pass
            predicted_value, _ = critic_model.forward(state_hist, text, image)
            
            # Compute loss
            loss = loss_fn(predicted_value, target_returns)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(critic_model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss_total += loss.item()
        
        avg_train_loss = train_loss_total / len(train_loader)

        # Validation epoch
        critic_model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1} [Val]", leave=False):
                # Similar logic as training, but without backpropagation
                context = batch['context']
                target_returns = batch['return'].to(device).unsqueeze(-1)
                state_hist, image, text = context['state_action_history'].to(device), context['image'].to(device), context['text']
                predicted_value, _ = critic_model.forward(state_hist, text, image)
                loss = loss_fn(predicted_value, target_returns)
                val_loss_total += loss.item()
        
        avg_val_loss = val_loss_total / len(val_loader)
        
        # Logging and Checkpointing
        print(f"Epoch [{epoch + 1:02d}] | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")
        if config.get("use_wandb", False):
            wandb.log({"epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": avg_val_loss})

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = save_dir / "critic_best.pth"
            torch.save(critic_model.state_dict(), best_model_path)
            print(f"  -> New best model saved to {best_model_path}")
    
    print("\nTraining finished.")
    if config.get("use_wandb", False):
        wandb.finish()


# ==============================================================================
# SECTION 3: SCRIPT EXECUTION
# ==============================================================================

if __name__ == '__main__':
    
    # --- Central Configuration ---
    CONFIG = {
        # Data source path from your rollout script
        "data_source_path": "data/libero/noise",

        # Model Configuration (must match the model used during rollout)
        'state_dim': 8, # Example from LIBERO: 7 for eef_pos+quat_axis_angle, 7 for gripper
        'action_dim': 7, # 6 for eef, 1 for gripper
        'gru_hidden_dim': 256,
        'n_heads': 8,
        'clip_model_name': "openai/clip-vit-base-patch32",
        
        # Reward parameters (should match those used during data collection)
        'reward_alpha': 0.01,
        'reward_beta': 0.1,
        
        # Training Hyperparameters
        "lr": 3e-4,
        "epochs": 500,
        "batch_size": 128,
        "gamma": 0.99, # Discount factor for calculating returns
        "val_split_ratio": 0.15,
        "history_len": 10, # Sequence length for the GRU
        
        # System & Logging
        "num_workers": 4,
        "use_wandb": False, # Set to True to enable logging
        "save_dir": "checkpoints/critic_offline",
    }
    
    # Run the training process
    train_critic_from_rollouts(CONFIG)


