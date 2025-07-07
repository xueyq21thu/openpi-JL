# append the src directory to the path
import sys
sys.path.append("./src")

import torch
import numpy as np
from noise.model.noise_vision import VisionNoiseModel

# Example config for VisionNoiseModel
config = {
    'state_dim': 16,
    'action_dim': 7,
    'image_dim': [3, 224, 224],
    'd_model': 128,
    'n_heads': 8,
    'alpha': 0.1,
    'beta': 1.0,
}

# Instantiate the model
model = VisionNoiseModel(config)

# Test weight loading (should print checkpoint info if exists)
# This is done automatically in __init__

# Generate fake input data
state = torch.randn(1, config['state_dim'])
action = torch.randn(1, config['action_dim'])
image = torch.randn(1, 3, 224, 224)  # Single image

# Test noise sampling
with torch.no_grad():
    delta = model.sample(state, action, image)
    print("Sampled noise delta:", delta)
    print("Delta shape:", delta.shape)