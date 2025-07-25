# noise_fusion.py

################################################################################
#
# This script defines the FusionNoiseModel, the "Actor" in the Actor-Critic
# framework for the CAL (Curriculum Adversarial Learning) project.
#
# Its primary role is to act as an intelligent adversary that learns when to
# inject noise into the main policy's actions to make tasks more challenging
# in a targeted way.
#
# The model architecture is as follows:
# 1. State/Action History -> GRU Encoder (Captures temporal dynamics)
# 2. Image/Text Goal     -> Pre-trained CLIP (Extracts semantic context)
# 3. Fusion              -> Cross-Attention (Intelligently merges dynamics
#                           with semantic context)
# 4. Decision            -> MLP Head (Outputs probability for injecting noise)
#
################################################################################


from typing import List, Dict, Any, Tuple, Optional

import torch
import itertools
import torch.nn as nn
from torch.distributions import Bernoulli
from transformers import CLIPProcessor, CLIPModel

# The script assumes a file 'noise_model.py' exists in the same directory
# and contains the 'NoiseModel' abstract base class.
from noise_model import NoiseModel

from torch.distributions import Bernoulli, Distribution


class FusionNoiseModel(NoiseModel, nn.Module):
    """
    The Actor Network in the Actor-Critic framework for CAL.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the FusionNoiseModel.
        """
        NoiseModel.__init__(self, config)
        nn.Module.__init__(self)

        self._build_model_architecture()
        self.reset()

    # ==========================================================================
    # --- Model Architecture Construction ---
    # ==========================================================================

    def _build_model_architecture(self):
        """
        Helper method to construct all the neural network layers.
        """
        clip_model_name = self.config.get('clip_model_path', "openai/clip-vit-base-patch32")
        
        # --- 1. Load Pre-trained CLIP Encoders for Semantic Understanding ---
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        
        # ======================================================================
        # THE FIX IS HERE: Add 'use_safetensors=True' to the model loading call.
        # This forces the use of the secure .safetensors format, avoiding the
        # torch.load vulnerability and the associated error.
        self.clip_model = CLIPModel.from_pretrained(clip_model_name, use_safetensors=True)
        # ======================================================================

        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.d_model = self.clip_model.config.projection_dim

        # --- 2. State/Action GRU Encoder for Temporal Dynamics ---
        state_action_dim = self.config['state_dim'] + self.config['action_dim']
        gru_hidden_dim = self.config.get('gru_hidden_dim', 256)
        self.state_action_encoder = nn.GRU(
            input_size=state_action_dim,
            hidden_size=gru_hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        self.state_action_projector = nn.Linear(gru_hidden_dim, self.d_model)

        # --- 3. Cross-Attention for Intelligent Multimodal Fusion ---
        n_heads = self.config.get('n_heads', 8)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=n_heads,
            batch_first=True,
            dropout=0.1
        )
        
        # --- 4. Actor's Final Decision Head ---
        self.mask_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Linear(self.d_model // 2, 1)
        )

        # configure
        self.episode_length = self.config.get('episode_length', 360)

    def trainable_parameters(self):
        """
        Returns an iterator over the model's trainable parameters,
        explicitly excluding the frozen CLIP model.
        """
        return itertools.chain(
            self.state_action_encoder.parameters(),
            self.state_action_projector.parameters(),
            self.cross_attention.parameters(),
            self.mask_head.parameters()
        )

    # ==========================================================================
    # --- Core Forward Pass ---
    # ==========================================================================

    def forward(
        self,
        state_action_history: torch.Tensor,
        text_instruction: List[str],
        image: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = self.device
        
        text_inputs = self.clip_processor(text=text_instruction, return_tensors="pt", padding=True)
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        text_features = self.clip_model.get_text_features(**text_inputs)
        
        image_inputs = self.clip_processor(images=image, return_tensors="pt")
        image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
        image_features = self.clip_model.get_image_features(**image_inputs)
        
        gru_out, next_hidden_state = self.state_action_encoder(state_action_history, hidden_state)
        last_gru_out = gru_out[:, -1, :]
        state_action_features = self.state_action_projector(last_gru_out)
        
        query = state_action_features.unsqueeze(1)
        context_kv = torch.stack([image_features, text_features], dim=1)
        attn_out, _ = self.cross_attention(query=query, key=context_kv, value=context_kv)
        
        fused_features = attn_out.squeeze(1)
        mask_logit = self.mask_head(fused_features)
        
        return mask_logit, next_hidden_state

    def get_distribution(
        self,
        state_action_history: torch.Tensor,
        text_instruction: List[str],
        image: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None
    ) -> Tuple[Distribution, torch.Tensor]:
        """
        Computes the policy distribution for the given state.
        This is essential for TRPO to compute KL divergence.
        """
        mask_logit, next_hidden_state = self.forward(
            state_action_history, text_instruction, image, hidden_state
        )
        # Our action space is binary (inject or not), so a Bernoulli distribution is perfect.
        dist = Bernoulli(logits=mask_logit)
        return dist, next_hidden_state

    # ==========================================================================
    # --- Implementation of Abstract Methods from NoiseModel ---
    # ==========================================================================

    def reset(self):
        self.hidden_state: Optional[torch.Tensor] = None
        self.state_action_history: List[torch.Tensor] = []

    def sample(self, state: torch.Tensor, action: torch.Tensor, image: torch.Tensor, text_instruction: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            state_action = torch.cat([state, action], dim=-1).to(self.device)
            self.state_action_history.append(state_action)
            history_tensor = torch.stack(self.state_action_history, dim=1)
            
            mask_logit, new_hidden_state = self.forward(
                history_tensor, text_instruction, image, self.hidden_state
            )
            self.hidden_state = new_hidden_state.detach()

            dist = Bernoulli(logits=mask_logit)
            mask = dist.sample()
            log_prob = dist.log_prob(mask)

            base_noise = self._generate_base_noise(action)
            delta = mask * base_noise
            
        return delta, mask.int(), log_prob

    def compute_noise(self, state_action_history: torch.Tensor, image: torch.Tensor, text_instruction: List[str]) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            mask_logit, _ = self.forward(state_action_history, text_instruction, image)
            dist = Bernoulli(logits=mask_logit)
            mask = dist.sample()
            
            action_dim = self.config['action_dim']
            last_action = state_action_history[:, -1, -action_dim:]
            
            base_noise = self._generate_base_noise(last_action)
            delta = mask * base_noise
        return delta

    def compute_reward(self, success: bool, delta: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        k = self.config.get('reward_k', 1.0)
        alpha = self.config.get('reward_alpha', 0.1)
        beta = self.config.get('reward_beta', 0.05)
        
        r_succ = k if not success else 0.0
        r_amp = -alpha * torch.sum(delta ** 2).item()
        r_freq = -beta * mask.float().sum().item()
        
        total_reward = torch.tensor(r_succ + r_amp + r_freq, device=self.device)
        return total_reward

    def compute_loss(self, log_probs: torch.Tensor, advantages: torch.Tensor) -> torch.Tensor:
        policy_loss = -(log_probs * advantages.detach()).mean()
        return policy_loss
    
    # ==========================================================================
    # --- Helper Methods ---
    # ==========================================================================
    
    def _generate_base_noise(self, action: torch.Tensor) -> torch.Tensor:
        noise_std = self.config.get('noise_std', 0.1)
        return torch.randn_like(action) * noise_std

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


# ==============================================================================
# --- Example Usage ---
# ==============================================================================

if __name__ == "__main__":
    
    # --- 1. Setup Configuration and Instantiate Model ---
    print("--- 1. Initializing Model and Configuration ---")
    
    config = {
        "state_dim": 8,
        "action_dim": 7,
        "clip_model_name": "openai/clip-vit-base-patch32",
        "gru_hidden_dim": 256,
        "n_heads": 4,
        "reward_k": 1.0,
        "reward_alpha": 0.1,
        "reward_beta": 0.05,
        "noise_std": 0.1,
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        model = FusionNoiseModel(config).to(device)
        print("✅ FusionNoiseModel instantiated successfully.")
    except NameError:
        print("\n❌ ERROR: Could not find 'NoiseModel'.")
        exit()
    except Exception as e:
        print(f"\n❌ An error occurred during model initialization: {e}")
        exit()

    # --- 2. Simulate an Episode Rollout (3 steps) ---
    print("\n--- 2. Simulating a 3-step episode rollout ---")
    model.reset()  # Reset recurrent state at the start of the episode
    
    B = 1 # Using a batch size of 1 for simplicity
    for step in range(3):
        print(f"\n>>> Step {step + 1} <<<")
        # Create dummy data for the current time step
        current_state = torch.randn(B, config["state_dim"]).to(device)
        current_action = torch.randn(B, config["action_dim"]).to(device)
        image = torch.rand(B, 3, 224, 224).to(device)
        text_instruction = ["pick up the green cup and place it on the plate"]
        
        # The 'sample' method is called at each step to get a noise decision and manage internal state
        delta, mask, log_prob = model.sample(current_state, current_action, image, text_instruction)
        
        print(f"   - Sampled Mask: {mask.item()}")
        print(f"   - Log Probability: {log_prob.item():.4f}")
        
        if mask.item() == 1:
            print("   - Decision: ✅ Inject noise.")
            print(f"   - Noise Delta (first 3 dims): {delta.squeeze()[:3].cpu().numpy()}")
        else:
            print("   - Decision: ❌ Do not inject noise.")
            
    # --- 3. Demonstrate Reward and Loss Calculation ---
    print("\n\n--- 3. Demonstrating Reward and Loss Calculation ---")
    
    # A. Reward Calculation Logic
    print("\n[A. Reward Calculation]")
    # Case 1: Noise was injected and the agent FAILED -> HIGHEST reward for noise policy
    reward_failure = model.compute_reward(success=False, delta=delta, mask=mask)
    print(f"   - Reward (when agent fails with noise): {reward_failure.item():.4f}")
    
    # Case 2: Noise was injected but the agent SUCCEEDED -> LOWEST reward for noise policy
    reward_success = model.compute_reward(success=True, delta=delta, mask=mask)
    print(f"   - Reward (when agent succeeds with noise): {reward_success.item():.4f}")

    # B. Loss Calculation Logic
    print("\n[B. Policy Loss Calculation]")
    # This would typically use data collected from a full rollout and computed by a Critic
    dummy_log_probs = torch.tensor([-0.693, -0.105, -1.386], device=device)
    dummy_advantages = torch.tensor([2.5, -0.5, 3.0], device=device) # Positive advantage means action was better than average
    
    loss = model.compute_loss(dummy_log_probs, dummy_advantages)
    print(f"   - Example Policy Loss: {loss.item():.4f}")
    print("      (A positive loss means the optimizer will try to make actions with positive advantages more likely).")