# pretrain_noise_model.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from typing import Dict, Any


"""
This script implements pretraining of the noise model using Imitation Learning.

The pretraining process initializes the noise model by training it to mimic an
"expert" or target noise distribution from a static, pre-collected dataset.
The data is expected to contain trajectories with:
- state_action_history: Robot state and action history [B, T_seq, D_state+D_action]
- image: Camera images at the current step [B, C, H, W]
- text: The language instruction for the task
- target_mask: The ground-truth noise mask {0, 1} to be imitated [B, 1]
"""

def pre_training(model, dataloader: DataLoader, config: Dict[str, Any], device: torch.device):
    """
    Performs imitation learning pre-training on the given noise model.

    Args:
        model: The FusionNoiseModel instance to train.
        dataloader: A DataLoader providing batches of pre-collected expert data.
        config: A dictionary with training hyperparameters.
        device: The device to train on ('cuda' or 'cpu').
    """
    # Set the model to training mode
    model.train()

    # --- Setup Optimizer, Scheduler, and Loss Function ---
    optimizer = optim.Adam(model.parameters(), lr=config.get("lr", 1e-4))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.get("lr_step", 10), gamma=0.5)
    
    # Binary Cross-Entropy with Logits is the standard and numerically stable
    # choice for binary classification tasks like this one.
    criterion = nn.BCEWithLogitsLoss()

    # --- Training Configuration ---
    epochs = config.get("epochs", 50)
    log_freq = config.get("log_freq", 10) # Log more frequently for demonstration

    print("Pre-training via Imitation Learning started...")

    # --- Create Checkpoint Directory ---
    save_dir = "checkpoints/noise/pretraining"
    os.makedirs(save_dir, exist_ok=True)
    print(f"Models will be saved in '{save_dir}'")

    # --- Main Training Loop ---
    for epoch in range(epochs):

        total_loss = 0.0

        # Iterate over the dataset with a progress bar
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            # --- 1. Get the data and move to device ---
            state_action_hist = batch['state_action_history'].to(device)
            image = batch['image'].to(device)
            text = batch['text']  # List of strings, remains on CPU
            
            # This is the supervision signal for imitation learning
            target_mask = batch.get('target_mask').to(device, dtype=torch.float32)

            # --- 2. Forward pass ---
            optimizer.zero_grad()
            
            # The model's forward pass returns the raw logit for the mask
            # We assume each batch is independent, so hidden state is re-initialized (or None)
            predicted_logit, _ = model.forward(state_action_hist, text, image)

            # --- 3. Compute Loss ---
            # The loss measures how well the model's prediction (logit) matches the target
            loss = criterion(predicted_logit, target_mask)

            # --- 4. Backward pass and optimization ---
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # --- End of Epoch ---
        avg_loss = total_loss / len(dataloader)
        scheduler.step() # Update the learning rate

        # --- Logging ---
        if (epoch + 1) % log_freq == 0 or (epoch + 1) == epochs:
            # Optional: Log to wandb
            # wandb.log({"epoch": epoch + 1, "loss": avg_loss, "lr": scheduler.get_last_lr()[0]})
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}, Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # --- Optional: Save intermediate models ---
        # model_save_freq = config.get("model_save_freq", 50)
        # if (epoch + 1) % model_save_freq == 0:
        #     torch.save(model.state_dict(), f"{save_dir}/noise_model_{epoch+1}.pth")

    # --- Final Model Saving ---
    final_model_path = os.path.join(save_dir, "fusion_noise_model_pretrained_final.pth")
    torch.save(model.state_dict(), final_model_path)

    print(f"Pre-training completed. Final model saved as '{final_model_path}'")

# --- Example Usage ---
if __name__ == '__main__':

    # Define model and training configurations
    config = {
        "lr": 1e-4,
        "lr_step": 10,
        "epochs": 20,
        "batch_size": 16,
        # Model-specific configs
        'state_dim': 16,
        'action_dim': 8,
        'gru_hidden_dim': 256,
        'n_heads': 8,
        'clip_model_name': "openai/clip-vit-base-patch32"
    }

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate model
    model = FusionNoiseModel(config).to(device)

    # Create a dummy dataloader for demonstration
    class ImitationDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples=500):
            self.num_samples = num_samples
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            # Each item must be a dictionary with the required keys
            return {
                'state_action_history': torch.randn(10, config['state_dim'] + config['action_dim']),
                'image': torch.randn(3, 224, 224),
                'text': "a sample instruction",
                'target_mask': torch.tensor([float(idx % 2)]) # Alternate 0s and 1s for variety
            }

    dataset = ImitationDataset()
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    # Run the pre-training function
    pre_training(model, dataloader, config, device)