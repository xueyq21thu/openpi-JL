# fake_sample.py

# Append the src directory to the path to find the 'noise' module
import sys
sys.path.append("./src")

import torch
from noise.model.noise_fusion import FusionNoiseModel

# ==============================================================================
# --- Configuration ---
# This config must match the one used to initialize the model.
# ==============================================================================
config = {
    # Dimensions
    "state_dim": 8,     # e.g., 3 (pos) + 3 (axis-angle) + 8 (gripper)
    "action_dim": 7,     # e.g., 7-DoF robot arm actions
    
    # Model Hyperparameters
    "clip_model_name": "openai/clip-vit-base-patch32",
    "gru_hidden_dim": 256,
    "n_heads": 4,
    
    # Reward Function Hyperparameters (not used in sampling, but needed for init)
    "reward_k": 1.0,
    "reward_alpha": 0.1,
    "reward_beta": 0.05,
    
    # Noise Generation
    "noise_std": 0.1,
}

print("--- 1. Initializing FusionNoiseModel ---")
# Set device (CUDA if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

try:
    # Instantiate the model and move it to the correct device
    model = FusionNoiseModel(config).to(device)
    # Set the model to evaluation mode, which is important for layers like Dropout
    model.eval() 
    print("✅ Model instantiated successfully.")
except Exception as e:
    print(f"\n❌ An error occurred during model initialization: {e}")
    print("   Please ensure you have an internet connection and all dependencies (torch, transformers) are installed.")
    exit()


# ==============================================================================
# --- Testing a Single Sample Step ---
# This simulates what happens in one iteration of the environment loop.
# ==============================================================================
print("\n--- 2. Testing a Single Sample Step ---")

# Reset the model's internal state before starting a new sequence
model.reset()

# Generate fake input data for a single time step (batch size = 1)
B = 1
# State and action are simple vectors
state_t = torch.randn(B, config["state_dim"]).to(device)
action_t = torch.randn(B, config["action_dim"]).to(device)
# Image must be in [0, 1] range and have shape (B, C, H, W)
image_t = torch.rand(B, 3, 224, 224).to(device) 
# Text instruction is a list of strings
text_instruction = ["pick up the green cup and place it on the plate"]

print("Input shapes:")
print(f"  - State:  {state_t.shape}")
print(f"  - Action: {action_t.shape}")
print(f"  - Image:  {image_t.shape}")
print(f"  - Text:   '{text_instruction[0]}'")

# Test the noise sampling method
with torch.no_grad():
    delta, mask, log_prob = model.sample(state_t, action_t, image_t, text_instruction)

print("\nOutput from model.sample():")
print(f"  - Sampled Mask:      {mask.item()}")
print(f"  - Log Probability:   {log_prob.item():.4f}")
print(f"  - Delta Shape:       {delta.shape}")
print(f"  - Delta magnitude:   {torch.linalg.norm(delta).item():.4f}")
if mask.item() == 1:
    print("  - Decision: ✅ Noise was injected.")
else:
    print("  - Decision: ❌ No noise was injected (delta is all zeros).")


# ==============================================================================
# --- Testing a Multi-Step Sequence ---
# This tests the model's recurrent nature (how it handles history).
# ==============================================================================
print("\n--- 3. Testing a 30-Step Recurrent Sequence ---")

# Reset the model's state to start a fresh episode
model.reset()
print("Model state has been reset.")

for step in range(30):
    print(f"\n>>> Step {step + 1} <<<")
    # Generate new data for each step
    state_step = torch.randn(B, config["state_dim"]).to(device)
    action_step = torch.randn(B, config["action_dim"]).to(device)
    image_step = torch.rand(B, 3, 224, 224).to(device)
    
    # The 'sample' method automatically handles the GRU's hidden state
    # and the state-action history internally.
    with torch.no_grad():
        delta_step, mask_step, _ = model.sample(state_step, action_step, image_step, text_instruction)
    
    print(f"  - Mask for this step: {mask_step.item()}")
    
    # Verify that the internal history is growing
    # The history is a list of tensors, its length should be step + 1
    history_len = len(model.state_action_history)
    print(f"  - Internal history length: {history_len}")
    assert history_len == step + 1, "Internal history is not being updated correctly!"

print("\n✅ Recurrent sequence test passed. The model correctly maintains its internal history.")
print("✅ Test script finished successfully.")
# ==============================================================================
# End of script
# ==============================================================================