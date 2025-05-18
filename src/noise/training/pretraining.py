import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import os
import wandb

"""
This script implements pretraining of noise models using data collected from pi0 policy and dummy noise models.

The pretraining process helps initialize the noise model parameters before online training.
The data consists of trajectories collected by running pi0 policy with dummy noise models.
Each trajectory contains:
- states: Robot state vectors over time [T, state_dim]
- actions: Robot action vectors over time [T, action_dim] 
- images: Camera images over time [T, C, H, W]
- success: Binary success flag for the trajectory
"""

def pre_training(model, dataloader, config, device):
    # Set the model to training mode
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=config.get("lr", 1e-4))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.get("lr_step", 10), gamma=0.5)

    epochs = config.get("epochs", 50)
    use_supervised = config.get("supervised", True)

    log_freq = config.get("log_freq", 50)
    model_save_freq = config.get("model_save_freq", 500)

    print("Pretraining started...")

    # create the models folder if it doesn't exist
    if not os.path.exists("checkpoints/noise/pretraining"):
        os.makedirs("checkpoints/noise/pretraining", exist_ok=True)
    print(f"Models will be saved in 'checkpoints/noise/pretraining'")

    for epoch in range(epochs):

        total_loss = 0.0

        # iterate over the dataset
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            # get the data
            state = batch['state'].to(device)
            action = batch['action'].to(device)
            image = batch['image'].to(device)
            target_delta = batch.get('delta', None).to(device)

            # forward pass
            optimizer.zero_grad()
            predicted_delta = model(state, action, image)

            if use_supervised and target_delta is not None:
                loss = model.compute_loss(predicted_delta, target_delta)
            else:
                success = batch['success'].to(device)
                loss = model.compute_loss(predicted_delta, reward=model.compute_reward(success, predicted_delta))

            # backward pass
            loss.backward()
            optimizer.step()

            # update the loss
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        # update the learning rate
        scheduler.step()

        # log the loss
        if (epoch + 1) % log_freq == 0:
            # wandb.log({"epoch": epoch + 1, "loss": avg_loss, "lr": scheduler.get_last_lr()[0]})
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}, Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # save the model
        # if (epoch + 1) % model_save_freq == 0:
        #     torch.save(model.state_dict(), f"models/noise_model_{epoch+1}.pth")

    # save the final model
    torch.save(model.state_dict(), f"checkpoints/noise/pretraining/noise_model_pretraining.pth")

    print(f"Pretraining completed. Final model saved as 'models/noise_model_final.pth'")