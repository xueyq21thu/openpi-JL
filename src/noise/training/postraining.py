import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import wandb
from tqdm import tqdm

def post_training(model, dataloader, config, device):
    """
    Post-training function for the noise model.
    """
    # Set the model to training mode
    model.train()

    # Configurations
    epochs = config.get("epochs", 50)
    log_freq = config.get("log_freq", 50)
    model_save_freq = config.get("model_save_freq", 500)

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['lr_step'], gamma=0.5)

    print("Post-training started...")

    # Create the models folder if it doesn't exist
    if not os.path.exists("checkpoints/noise/postraining"):
        os.makedirs("checkpoints/noise/postraining", exist_ok=True)
    print(f"Models will be saved in 'checkpoints/noise/postraining'")


    # Training loop
    for epoch in range(epochs):
        total_loss = 0.0
        total_steps = 0

        for episode in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            # Unpack episode tensors
            states = episode["state"].to(device)         # [T, state_dim]
            actions = episode["action"].to(device)       # [T, action_dim]
            images = episode["image"].to(device)         # [T, C, H, W]
            successes = episode["success"].to(device)    # [T, 1] or [T]

            # Forward pass
            optimizer.zero_grad()
            deltas = model(states, actions, images)

            # Compute Reward for each step
            rewards = torch.tensor([
                model.compute_reward(success, delta)
                for success, delta in zip(successes, deltas)
            ], device=device)

            # Compute loss
            loss = model.compute_loss(deltas, reward=rewards, state=states, action=actions, image=images)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Update the loss and steps
            total_loss += loss.item()
            total_steps += 1

        avg_loss = total_loss / total_steps
        # Update the learning rate
        scheduler.step()

        # Log the loss
        if (epoch + 1) % log_freq == 0:
            # wandb.log({"epoch": epoch + 1, "loss": avg_loss})
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

        # Save the model
        if (epoch + 1) % model_save_freq == 0:
            model_path = f"checkpoints/noise/postraining/model_epoch_{epoch + 1}.pth"
            torch.save(model.state_dict(), model_path)
            print(f"Model saved at {model_path}")

    # Final save of the model
    final_model_path = "checkpoints/noise/postraining/model_final.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"Post-training completed. Final model saved at {final_model_path}")

