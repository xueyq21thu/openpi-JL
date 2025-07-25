# scripts/train_offline.py

import sys
sys.path.append("./src") 

from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm, trange
import wandb

from noise.model.noise_fusion import FusionNoiseModel
from noise.model.valuenet import ValueNetwork, compute_gae_and_returns
from noise.training.updater import trpo_step, ppo_step
from noise.training.dataset import TrajectoryDataset
from noise.utils.utils import collate_multimodal_batch

# ==============================================================================
# SECTION 1: MAIN TRAINING FUNCTION
# ==============================================================================

def train_actor_critic_offline(config: Dict[str, Any]):
    """
    Main function to orchestrate the offline training of both the Actor (FusionNoiseModel)
    and the Critic (ValueNetwork) using collected trajectory data.
    """
    # --- 1. Setup Environment and Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Setup ---")
    print(f"Using device: {device}")
    
    save_dir = Path(config.get("save_dir", "checkpoints/actor_critic_offline"))
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints will be saved to: {save_dir.resolve()}")

    # --- 2. Load and Prepare Data ---
    print("\n--- Data Loading ---")
    data_path = Path(config["data_source_path"])
    trajectory_files = list(data_path.rglob("*.npy"))
    if not trajectory_files:
        raise FileNotFoundError(f"No .npy files found in {data_path}")

    full_dataset = TrajectoryDataset(
        trajectory_files, config, history_len=config.get('history_len', 10)
    )
    
    val_split = config.get("val_split_ratio", 0.15)
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Dataset loaded: {len(full_dataset)} total samples from {len(trajectory_files)} trajectories.")
    print(f"  - Training set size: {len(train_dataset)}")
    print(f"  - Validation set size: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True,
        collate_fn=collate_multimodal_batch, num_workers=config.get('num_workers', 0)
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"],
        collate_fn=collate_multimodal_batch, num_workers=config.get('num_workers', 0)
    )

    # --- 3. Initialize Models and Optimizers ---
    print("\n--- Model Initialization ---")
    actor_model = FusionNoiseModel(config).to(device)
    critic_model = ValueNetwork(config).to(device)
    
    critic_checkpoint_path = config.get("critic_checkpoint_path")
    if critic_checkpoint_path and Path(critic_checkpoint_path).exists():
        print(f"Loading pre-trained Critic from: {critic_checkpoint_path}")
        critic_model.load_state_dict(torch.load(critic_checkpoint_path, map_location=device))

    actor_checkpoint_path = config.get("actor_checkpoint_path")
    if actor_checkpoint_path and Path(actor_checkpoint_path).exists():
        print(f"Loading pre-trained Actor from: {actor_checkpoint_path}")
        actor_model.load_state_dict(torch.load(actor_checkpoint_path, map_location=device))

    optimizer_critic = optim.AdamW(critic_model.parameters(), lr=config.get("lr_critic", 3e-4))
    loss_fn_critic = nn.MSELoss()

    optimizer_actor = optim.AdamW(actor_model.parameters(), lr=config.get("lr_actor", 3e-5))


    # --- 4. Initialize Logging (W&B) ---
    if config.get("use_wandb", False):
        wandb.init(project=config.get("wandb_project", "cal-actor-critic-offline"), config=config)
        wandb.watch(critic_model, log_freq=100)
        wandb.watch(actor_model, log_freq=100)
        print("\n--- Weights & Biases Logging Enabled ---")

    # --- 5. Main Training Loop ---
    best_critic_val_loss = float('inf')
    # --- MODIFICATION: Add tracker for best actor performance ---
    best_actor_proxy_loss = float('inf') 
    
    print("\n--- Starting Offline Actor-Critic Training ---")
    for epoch in trange(config.get("epochs", 20), desc="Overall Training Progress"):
        
        # --- PHASE 1: CRITIC TRAINING & VALIDATION ---
        critic_model.train()
        total_critic_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} [Critic Train]", leave=False):
            context = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch['context'].items()}
            target_returns = batch['returns_for_critic'].to(device).unsqueeze(-1)
            predicted_value, _ = critic_model.forward(**context)
            loss = loss_fn_critic(predicted_value, target_returns)
            optimizer_critic.zero_grad(); loss.backward(); optimizer_critic.step()
            total_critic_loss += loss.item()
        avg_critic_loss = total_critic_loss / len(train_loader)
        
        critic_model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1} [Critic Val]", leave=False):
                context = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch['context'].items()}
                target_returns = batch['returns_for_critic'].to(device).unsqueeze(-1)
                predicted_value, _ = critic_model.forward(**context)
                val_loss = loss_fn_critic(predicted_value, target_returns)
                total_val_loss += val_loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  -> Critic Training Loss: {avg_critic_loss:.5f} | Critic Validation Loss: {avg_val_loss:.5f}")
        
        if avg_val_loss < best_critic_val_loss:
            best_critic_val_loss = avg_val_loss
            critic_save_path = save_dir / "critic_best.pth"
            torch.save(critic_model.state_dict(), critic_save_path)
            print(f"  -> ✅ New best Critic saved to {critic_save_path}")

        # --- PHASE 2: ACTOR TRAINING ---
        # actor_model.train(); critic_model.eval()
        # full_train_batch = collate_multimodal_batch(list(train_dataset))
        # context = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in full_train_batch['context'].items()}
        # rewards = full_train_batch['rewards'].to(device)
        # dones = full_train_batch['dones'].to(device)
        # mask_actions = full_train_batch['mask_actions'].to(device)
        # old_log_probs = full_train_batch['old_log_probs'].to(device)

        print("\n-> Phase 2: Updating Actor with PPO...") # <--- MODIFICATION: Changed name
        actor_model.train()
        critic_model.eval()

        full_train_batch = collate_multimodal_batch(list(train_dataset))
        
        context = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in full_train_batch['context'].items()}
        rewards = full_train_batch['rewards'].to(device)
        dones = full_train_batch['dones'].to(device)
        mask_actions = full_train_batch['mask_actions'].to(device)
        old_log_probs = full_train_batch['old_log_probs'].to(device)

        with torch.no_grad():
            values, _ = critic_model.forward(**context)
            values = values.squeeze(-1)
            values_for_gae = torch.cat([values, torch.zeros(1, device=device)], dim=0)
            advantages, _ = compute_gae_and_returns(
                rewards.unsqueeze(1), values_for_gae.unsqueeze(1), dones.unsqueeze(1),
                config['gamma'], config.get('gae_lambda', 0.95)
            )
            advantages = advantages.squeeze(1)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # trpo_step(
        #     actor=actor_model, states=context, advantages=advantages,
        #     old_log_probs=old_log_probs, mask_actions=mask_actions,
        #     max_kl=config.get('trpo_max_kl', 0.01), damping=config.get('trpo_damping', 0.1)
        # )

        ppo_step(
            actor=actor_model,
            optimizer=optimizer_actor,
            states=context,
            advantages=advantages,
            old_log_probs=old_log_probs,
            mask_actions=mask_actions,
            ppo_epochs=config.get('ppo_epochs', 10),
            clip_param=config.get('ppo_clip_param', 0.2)
        )
        print(f"  -> Actor updated with PPO (Epoch {epoch + 1})")

        # --- MODIFICATION: PHASE 3 - ACTOR VALIDATION AND SAVING ---
        print("-> Phase 3: Evaluating and Saving Actor...")
        actor_model.eval()
        with torch.no_grad():
            # Use the validation set to evaluate the new actor
            full_val_batch = collate_multimodal_batch(list(val_dataset))
            val_context = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in full_val_batch['context'].items()}
            val_rewards = full_val_batch['rewards'].to(device)
            val_dones = full_val_batch['dones'].to(device)
            val_mask_actions = full_val_batch['mask_actions'].to(device)
            val_old_log_probs = full_val_batch['old_log_probs'].to(device)

            # Calculate advantages on the validation set
            val_values, _ = critic_model.forward(**val_context)
            val_values = val_values.squeeze(-1)
            val_values_for_gae = torch.cat([val_values, torch.zeros(1, device=device)], dim=0)
            val_advantages, _ = compute_gae_and_returns(
                val_rewards.unsqueeze(1), val_values_for_gae.unsqueeze(1), val_dones.unsqueeze(1),
                config['gamma'], config.get('gae_lambda', 0.95)
            )
            val_advantages = val_advantages.squeeze(1)
            val_advantages = (val_advantages - val_advantages.mean()) / (val_advantages.std() + 1e-8)

            # Calculate the policy loss on the validation set as our proxy metric
            val_dist, _ = actor_model.get_distribution(**val_context)
            val_new_log_probs = val_dist.log_prob(val_mask_actions)
            val_ratio = torch.exp(val_new_log_probs - val_old_log_probs)
            actor_proxy_loss = -(val_ratio * val_advantages).mean().item()
            
            print(f"  -> Actor Proxy Loss (on val set): {actor_proxy_loss:.5f}")

            if actor_proxy_loss < best_actor_proxy_loss:
                best_actor_proxy_loss = actor_proxy_loss
                actor_save_path = save_dir / "actor_best.pth"
                torch.save(actor_model.state_dict(), actor_save_path)
                print(f"  -> ✅ New best Actor saved to {actor_save_path} (Loss: {best_actor_proxy_loss:.5f})")
        
        if config.get("use_wandb", False):
            wandb.log({"epoch": epoch + 1, "actor_proxy_loss": actor_proxy_loss})


    print("\n--- Training finished ---")
    if config.get("use_wandb", False):
        wandb.finish()


# ==============================================================================
# SECTION 3: SCRIPT EXECUTION CONFIGURATION
# ==============================================================================

if __name__ == '__main__':
    
    # --- Central Configuration ---
    CONFIG = {
        # --- Paths ---
        "data_source_path": "data/libero/noise",
        "save_dir": "checkpoints/actor_critic_offline",
        'clip_model_path': "checkpoints/clip_model_local",
        "critic_checkpoint_path": None,
        "actor_checkpoint_path": None,
        
        # --- Model Architecture ---
        'state_dim': 8,
        'action_dim': 7,
        'gru_hidden_dim': 256,
        'n_heads': 8,
        
        # --- Training Hyperparameters ---
        "epochs": 50,
        "batch_size": 256,
        "lr_critic": 3e-4,
        "lr_actor": 3e-5,
        "history_len": 10,
        "val_split_ratio": 0.15,
        
        # --- GAE & TRPO Hyperparameters ---
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ppo_epochs": 10,
        "ppo_clip_param": 0.2,
        
        # --- System & Logging ---
        "num_workers": 0,
        "use_wandb": False,
        "wandb_project": "cal-actor-critic-offline",
    }

    train_actor_critic_offline(CONFIG)
