# valuenet.py

from typing import Dict, Any, Tuple, List, Optional

import torch
import torch.nn as nn

# To enable architecture sharing, we import the Actor's definition.
# This assumes 'noise_fusion.py' is in the same directory or Python path.
from noise_fusion import FusionNoiseModel

# ==============================================================================
# SECTION 1: CRITIC NETWORK DEFINITION
# ==============================================================================

class ValueNetwork(nn.Module):
    """
    The Critic Network in the Actor-Critic framework for CAL (Curriculum Adversarial Learning).

    This network's primary purpose is to learn the state-value function, V(s).
    It estimates the expected cumulative future reward from any given state, which,
    in the context of an adversarial setup, corresponds to the "failure potential".

    To ensure that both the Actor (FusionNoiseModel) and the Critic (ValueNetwork)
    interpret the world similarly, this Critic shares its core feature extraction
    architecture with the Actor.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the ValueNetwork.

        The configuration for the Critic should be identical to the Actor's to
        facilitate weight sharing and consistent representation learning.

        Args:
            config: A configuration dictionary containing all necessary model
                    hyperparameters (e.g., state_dim, action_dim, clip_model_name).
        """
        super().__init__()
        self.config = config

        # --- 1. Shared Backbone ---
        # We instantiate the Actor's model to serve as our feature extractor.
        # This is a powerful and common technique in modern Actor-Critic algorithms,
        # as it allows the Critic to leverage the rich, fused representations
        # that the Actor is already learning.
        self.backbone = FusionNoiseModel(config)
        
        # --- 2. Critic-Specific Value Head ---
        # While the backbone is shared, the final output layer is unique to the Critic.
        # This small MLP takes the shared feature representation and regresses it
        # to a single scalar value, representing V(s).
        d_model = self.backbone.d_model
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1) # Output a single scalar: the state value V(s)
        )
        
        # --- 3. Architecture Cleanup ---
        # The backbone (FusionNoiseModel) has a `mask_head` for the Actor's policy.
        # We must delete it from this instance to ensure that its parameters
        # are not included in the Critic's optimizer.
        if hasattr(self.backbone, 'mask_head'):
            del self.backbone.mask_head

    def forward(
        self, 
        state_action_history: torch.Tensor, 
        text_instruction: List[str], 
        image: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a forward pass to compute the state-value V(s).

        This method takes the exact same multimodal input as the Actor.

        Args:
            state_action_history: A tensor of recent state-action pairs.
            text_instruction: A list of language goal strings.
            image: The current visual observation tensor.
            hidden_state: The optional previous hidden state for the GRU.

        Returns:
            A tuple containing:
            - value (torch.Tensor): The predicted state-value. Shape: (Batch, 1).
            - next_hidden_state (torch.Tensor): The updated hidden state from the GRU.
        """
        # Step 1: Use the shared backbone to get a rich, fused feature vector
        # that encapsulates the entire multimodal context.
        fused_features, next_hidden_state = self.backbone.get_fused_features(
            state_action_history, text_instruction, image, hidden_state
        )
        
        # Step 2: Pass the fused features through the value-specific head
        # to get the final scalar value prediction.
        value = self.value_head(fused_features)
        
        return value, next_hidden_state

# ==============================================================================
# SECTION 2: REWARD AND ADVANTAGE COMPUTATION UTILITIES
# ==============================================================================

def compute_reward(
    success: Optional[bool],
    delta: torch.Tensor,
    mask: torch.Tensor,
    config: Dict[str, Any]
) -> float:
    """
    Computes the reward for the noise policy at a single timestep.

    This function implements the reward structure from your presentation (Slide 25),
    balancing three key objectives for the adversary:
    1.  Induce failure in the main policy (Curriculum).
    2.  Use minimal force (Subtle).
    3.  Act sparsely (Crucial).

    Args:
        success: Whether the main policy's task was successful *at the end of the
                 episode*. This is a terminal signal. For non-terminal steps,
                 this should be None.
        delta: The noise vector applied at the current step. Shape: (action_dim,).
        mask: The binary mask {0, 1} indicating if noise was injected.
        config: A dictionary containing reward hyperparameters (alpha, beta).

    Returns:
        The scalar reward for the current timestep.
    """
    # --- 1. Success Reward (r_succ): The Primary Objective ---
    # This is a sparse, terminal reward. The adversary is rewarded only if the
    # main policy ultimately fails. We give a positive reward for failure and
    # a negative one for success to create a clear objective.
    r_succ = 0.0
    if success is not None:
        r_succ = 1.0 if not success else -1.0

    # --- 2. Amplitude Penalty (r_amp): The "Subtlety" Component ---
    # We penalize the magnitude of the noise to encourage the adversary to find
    # the *minimal effective perturbation* rather than applying brute force.
    alpha = config.get('reward_alpha', 0.01)
    # The penalty is proportional to the squared L2 norm of the noise vector,
    # and is only applied if noise was actually injected (mask=1).
    r_amp = -alpha * torch.sum(delta.pow(2)).item() if mask.item() == 1 else 0.0

    # --- 3. Frequency Penalty (r_freq): The "Cruciality" Component ---
    # We penalize the act of injecting noise itself. This encourages the adversary
    # to be sparse and only act at the most critical moments.
    beta = config.get('reward_beta', 0.1)
    r_freq = -beta * mask.item()

    # The total reward is the sum of these components.
    total_reward = r_succ + r_amp + r_freq
    return total_reward

def compute_gae_and_returns(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    gae_lambda: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes Generalized Advantage Estimation (GAE) and the value targets (Returns).

    This function is a standard utility in modern policy gradient methods (like PPO).
    It takes a sequence of rewards and value estimates from a rollout and computes
    a more stable estimate of the advantage for each step.

    Args:
        rewards: A tensor of rewards collected at each step. Shape: (T, B).
        values: A tensor of state-values from the Critic. Shape: (T+1, B)
                (Includes the value of the state *after* the last action).
        dones: A tensor of done flags (1 if episode ended). Shape: (T, B).
        gamma: The discount factor for future rewards.
        gae_lambda: The GAE lambda parameter for balancing bias and variance.

    Returns:
        A tuple of (advantages, returns):
        - advantages: The GAE, used to update the Actor.
        - returns: The value targets (V_target), used to update the Critic.
    """
    advantages = torch.zeros_like(rewards)
    last_gae_lam = 0.0
    num_steps = rewards.size(0)

    # We iterate backwards from the last step to propagate future rewards.
    for t in reversed(range(num_steps)):
        # Determine the value of the *next* state.
        # If the episode terminated, the value of the next state is 0.
        next_nonterminal = 1.0 - dones[t]
        next_value = values[t + 1]
        
        # Calculate the TD Error: how much better was the actual outcome (r_t + V(s_{t+1}))
        # than what the critic initially predicted (V(s_t))?
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        
        # The GAE is a discounted sum of these TD errors.
        advantages[t] = last_gae_lam = delta + gamma * gae_lambda * next_nonterminal * last_gae_lam
        
    # The "returns" are the targets for the value function update.
    # Return(s_t) = Advantage(s_t, a_t) + V(s_t)
    returns = advantages + values[:-1] # We only need values for the T steps, not T+1
    
    return advantages, returns