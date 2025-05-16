import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, vit_b_16, vit_b_32
# from torchvision.models import ResNet18_Weights, ViT_B_16_Weights, ViT_B_32_Weights
from noise_model import NoiseModel
import os

"""
VisionNoiseModel is a neural network model designed to generate noise for actions in a vision-based environment. 
It extends the NoiseModel and PyTorch's nn.Module, leveraging both state and action information along with image data 
to compute noise that can be applied to actions. The model uses a Vision Transformer (ViT) for image processing 
and multi-head attention to align state and action representations with visual features.

Attributes:
    state_dim (int): Dimensionality of the state input.
    action_dim (int): Dimensionality of the action input.
    image_dim (list): Dimensions of the input image, defaulting to [3, 224, 224].
    d_model (int): Dimensionality of the model's internal representations.
    n_heads (int): Number of attention heads in the multi-head attention mechanism.
    alpha (float): Weight for regularization or other loss components.
    beta (float): Weight for semantic alignment in the loss function.

Methods:
    cumpute_noise(state, action, image):
        Computes noise based on the current state, action, and image inputs. 
        Utilizes cross-module attention to align state and action with image features, 
        and outputs noise through a linear transformation.
"""


class VisionNoiseModel(NoiseModel, nn.Module):
    def __init__(self, config):
        NoiseModel.__init__(self, config)
        nn.Module.__init__(self)
        cfg = self.config

        # configurations
        self.state_dim = cfg.get('state_dim')
        self.action_dim = cfg.get('action_dim')
        self.image_dim = cfg.get('image_dim', [3, 224, 224])
        self.d_model = cfg.get('d_model', 128)
        self.n_heads = cfg.get('n_heads', 8)

        self.alpha = config.get('alpha', 0.1)
        self.beta = config.get('beta', 1.0)  # semantic alignment weight

        checkpoint_path = "checkpoints/noise/pretraining/vit_b_32-d86f8d99.pth"
        if os.path.exists(checkpoint_path):
            vit = vit_b_32(weights=None)
            vit.load_state_dict(torch.load(checkpoint_path))
            print(f"Loaded ViT checkpoint from {checkpoint_path}")
        else:
            print(f"ViT Checkpoint not found at {checkpoint_path}, downloading checkpoints...")
            vit = vit_b_32(pretrained=True)
        
        # Set the vit classifier to identity
        vit.heads = nn.Identity()
        vit_output_dim = vit.hidden_dim

        # image encoder: PreTrained Vision Transformer(ViT)
        self.image_encoder = nn.Sequential(
            vit,
            nn.LayerNorm(vit_output_dim),
            nn.Linear(vit_output_dim, self.d_model),
        )

        # action / state encoder
        self.state_encoder = nn.Linear(self.state_dim, self.d_model)
        self.action_encoder = nn.Linear(self.action_dim, self.d_model)

        # cross module attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.n_heads,
            batch_first=True,
        )

        # output decoder
        self.delta_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.action_dim)
        )

        self.delta_history = []
        self.delta_amp_list = []

    def compute_noise(self, state, action=None, image=None):
        '''
        Compute the noise based on the state, action, and image.
        The cross module attention is used to align the state and action with the image.
        The noise is then computed by the delta head.
        Args:
            state: Current state of the environment.
            action: Current action to be modified with noise.
            image: Optional image input (if applicable).
        Return:
            Delta: noise with randomized amplitude and direction.
        '''
        # img shape: torch.tensor[1, 78, 3, 224, 224], unstack to [78, 3, 224, 224]
        if image is not None and image.ndim == 5 and image.shape[0] == 1:
            image = image.squeeze(0) # [1, S, C, H, W] -> [S, C, H, W]
        x_state = self.state_encoder(state)     # (B, D)
        x_action = self.action_encoder(action)  # (1, B, D)
        x_image = self.image_encoder(image)     # (1, B, D)

        # squeeze x_action and x_state to (B, D)
        x_state = x_state.squeeze(0)  # (1, B, D) -> (B, D)
        x_action = x_action.squeeze(0)  # (1, B, D) -> (B, D)

        # reshape for attention: (B, D) -> (B, 1, D)
        query = x_image.unsqueeze(1)  # image: query
        key_value = torch.stack([x_state, x_action], dim=1)  # state + action: keys/values

        attn_out, _ = self.cross_attention(query, key_value, key_value)
        delta = self.delta_head(attn_out.squeeze(1))

        # calculate the amplitude of delta
        delta_amp = torch.norm(delta, p=2, dim=-1)
        self.delta_amp_list.append(delta_amp.item())
        return delta
    
    def forward(self, state, action, image):
        """
        Forward pass through the model.
        Args:
            state: Current state of the environment.
            action: Current action to be modified with noise.
            image: Optional image input (if applicable).
        Return:
            Delta: noise with randomized amplitude and direction.
        """
        delta = self.compute_noise(state, action, image)
        return delta
    
    def sample(self, state, action=None, image=None):
        '''
        Sample noise based on the current state and action.
        Args:
            state: Current state of the environment.
            action: Current action to be modified with noise.
            image: Optional image input (if applicable).
        Return:
            Delta: noise with randomized amplitude and direction.
        '''
        with torch.no_grad():
            delta = self.forward(state, action, image)
            self.delta_history.append(delta)
            return delta
        
    def reset(self):
        """
        Reset the noise model to its initial state.
        Call this method when starting a new episode or task.
        """
        self.delta_history = []

    def compute_loss(self, predicted_delta, target_delta=None, reward=None, state=None, action=None, image=None):
        """
        Compute the loss based on the current delta and reward.
        Two modes of loss computation:
        1. Supervised learning: compute loss based on predicted and target deltas.
        2. Reinforcement learning: compute loss based on reward signal.
        The loss computation can be a combination of both modes.
        Args:
            predicted_delta: Predicted noise delta.
            target_delta: Target noise delta (if applicable).
            reward: Reward signal (if applicable).
            state: Current state of the environment.
            action: Current action to be modified with noise.
            image: Optional image input (if applicable).
        Return:
            loss: Computed loss based on the task.
        """
        
        loss = 0.0 # initialize loss
        
        # Supervised learning mode
        if target_delta is not None:
            loss += F.mse_loss(predicted_delta, target_delta)
        
        # Reinforcement learning mode
        if reward is not None:
            reward_val = reward if isinstance(reward, torch.Tensor) else torch.tensor(reward, dtype=torch.float32)
            loss += -reward_val.mean()

        # Alignment between vision and state-action
        if state is not None and image is not None:
            x_state = self.state_encoder(state)
            x_image = self.image_encoder(image)
            sim = F.cosine_similarity(x_state, x_image, dim=-1)
            align_loss = (1.0 - sim).mean()
            loss += self.beta * align_loss
        
        return loss

    def compute_reward(self, success: bool, delta: torch.Tensor = None):
        """
        Compute the reward based on the current observation and goal.
        Args:
            success: Boolean indicating if the task was successful.
            delta: Optional delta value (if applicable).
        """
        r_succ = -1.0 if success else 0.0
        if delta is not None:
            r_amp = -self.alpha * delta.pow(2).sum().item()
        else:
            r_amp = -self.alpha * torch.stack(self.delta_history).pow(2).sum().item()
        return r_succ + r_amp
