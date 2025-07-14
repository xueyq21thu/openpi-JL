# valuenet.py

from typing import Dict, Any, Tuple, List, Optional

import torch
import torch.nn as nn

# 导入 Actor 的定义，我们的 Critic 将复用其结构
from noise_fusion import FusionNoiseModel

# ==============================================================================
# SECTION 1: CRITIC NETWORK DEFINITION
# ==============================================================================

class ValueNetwork(nn.Module):
    """
    The Critic Network in the Actor-Critic framework for CAL.

    This version is specifically designed to work with the *existing* FusionNoiseModel
    class without requiring any modifications to it. It achieves this by using a
    PyTorch forward hook to capture the intermediate feature representations.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the ValueNetwork.
        
        Args:
            config: A configuration dictionary that must be compatible with
                    FusionNoiseModel.
        """
        super().__init__()
        self.config = config

        # --- 1. Shared Backbone ---
        # We instantiate the Actor's model to serve as our feature extractor.
        self.backbone = FusionNoiseModel(config)
        
        # --- 2. Critic-Specific Value Head ---
        # This MLP takes the shared feature representation from the backbone
        # and regresses it to a single scalar value, V(s).
        d_model = self.backbone.d_model
        self.value_head = nn.Sequential(
            nn.LayerNorm(d_model), # Adding a LayerNorm for stability
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1) # Output a single scalar value
        )
        
        # --- 3. Architecture Cleanup & Hook Setup ---
        # This is the core logic to make the Critic work without modifying the Actor.
        
        # This will hold the intermediate features captured by our hook.
        self.captured_features: Optional[torch.Tensor] = None

        # Define the hook function that will be attached to a layer.
        def hook_fn(module, input_args, output_from_module):
            """
            This hook captures the output of a specific layer.
            The 'input_args' to the mask_head is a tuple containing one element:
            the fused_features tensor we want to capture.
            """
            # `input_args` is a tuple, we need the first element.
            self.captured_features = input_args[0]

        # We attach the hook to the Actor's `mask_head`. When we call the
        # backbone's forward pass, just before `mask_head` is executed,
        # our hook will capture its input, which is exactly the fused_features.
        if hasattr(self.backbone, 'mask_head'):
            self.backbone.mask_head.register_forward_hook(hook_fn)
        else:
            raise AttributeError("The provided backbone (FusionNoiseModel) does not have a 'mask_head' to attach a hook to.")

    def forward(
        self, 
        state_action_history: torch.Tensor, 
        text_instruction: List[str], 
        image: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a forward pass to compute the state-value V(s).
        
        Args:
            state_action_history: Tensor of recent state-action pairs.
            text_instruction: List of language goal strings.
            image: Current visual observation tensor.
            hidden_state: Optional previous hidden state for the GRU.
            
        Returns:
            A tuple containing:
            - value (torch.Tensor): The predicted state-value. Shape: (Batch, 1).
            - next_hidden_state (torch.Tensor): The updated hidden state from the GRU.
        """
        # Step 1: Execute the backbone's full forward pass.
        # We don't need its direct return values (`mask_logit`), but this call
        # is necessary to trigger the forward hook we registered.
        with torch.no_grad(): # We don't need gradients from the backbone's final layers here
            _, next_hidden_state = self.backbone.forward(
                state_action_history, text_instruction, image, hidden_state
            )
        
        # Step 2: Check if the hook successfully captured the features.
        # If the forward pass ran correctly, self.captured_features will now hold
        # the tensor we need.
        if self.captured_features is None:
            raise RuntimeError(
                "Forward hook did not capture any features. "
                "Ensure the backbone's forward pass was called."
            )
        
        # Step 3: Pass the captured features through our value-specific head
        # to get the final scalar value prediction.
        value = self.value_head(self.captured_features)
        
        # Step 4: Reset the captured features to None for the next forward pass.
        # This is important to prevent using stale data in subsequent calls.
        self.captured_features = None
        
        return value, next_hidden_state

# ==============================================================================
# SECTION 2: REWARD AND ADVANTAGE COMPUTATION UTILITIES
# (This section remains unchanged as its logic is self-contained and correct.)
# ==============================================================================

def compute_reward(
    success: Optional[bool],
    delta: torch.Tensor,
    mask: torch.Tensor,
    config: Dict[str, Any]
) -> float:
    """Computes the reward for the noise policy at a single timestep."""
    r_succ = 0.0
    if success is not None:
        r_succ = 1.0 if not success else -1.0

    alpha = config.get('reward_alpha', 0.01)
    r_amp = -alpha * torch.sum(delta.pow(2)).item() if mask.item() == 1 else 0.0

    beta = config.get('reward_beta', 0.1)
    r_freq = -beta * mask.item()

    total_reward = r_succ + r_amp + r_freq
    return total_reward

def compute_gae_and_returns(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    gae_lambda: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes Generalized Advantage Estimation (GAE) and the value targets."""
    advantages = torch.zeros_like(rewards)
    last_gae_lam = 0.0
    num_steps = rewards.size(0)

    for t in reversed(range(num_steps)):
        next_nonterminal = 1.0 - dones[t]
        next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        
        advantages[t] = last_gae_lam = delta + gamma * gae_lambda * next_nonterminal * last_gae_lam
        
    returns = advantages + values[:-1]
    
    return advantages, returns