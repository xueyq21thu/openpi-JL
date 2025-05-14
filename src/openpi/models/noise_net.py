import numpy as np
import torch
import torch.nn as nn
from noise import NoiseModel


class NetworkNoiseModel(NoiseModel, nn.Module):
    def __init__(self, config):
        NoiseModel.__init__(self, config)
        nn.Module.__init__(self)
        cfg = self.config
        # Config parameters
        self.mode = cfg.get('mode', 'history')  # mode: 'history' or 'state_image'
        self.state_dim = cfg.get('state_dim')
        self.action_dim = cfg.get('action_dim')
        self.image_dim = cfg.get('image_dim')
        self.output_dim = cfg.get('output_dim')

        # Check if output_dim is provided
        if self.output_dim is None:  
            if self.mode == 'history':
                if self.action_dim is None:
                    raise ValueError("History: Need to provide action_dim or output_dim.")
                self.output_dim = self.action_dim
            else:
                if self.state_dim is None:
                    raise ValueError("state_image: Need to provide state_dim or output_dim.")
                self.output_dim = self.state_dim

        # Config parameters for the network
        self.history_length = cfg.get('history_length', 4)
        d_model = cfg.get('d_model', 64)
        nhead = cfg.get('nhead', 4)                       
        num_layers = cfg.get('num_layers', 2)             
        dim_ff = cfg.get('dim_feedforward', 128)          
        dropout = cfg.get('dropout', 0.1)

        # If in history mode
        if self.mode == 'history':
            if self.state_dim is None or self.action_dim is None:
                raise ValueError("History mode requires state_dim and action_dim.")

            # Configure the Transformer Encoder
            self.state_encoder = nn.Linear(self.state_dim, d_model)
            self.action_encoder = nn.Linear(self.action_dim, d_model)
            self.delta_encoder = nn.Linear(self.action_dim, d_model)

            # Configure learnable position embedding
            max_seq_len = self.history_length + 1
            self.pos_embedding = nn.Embedding(max_seq_len, d_model)
            # Transformer Encoder
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                       dim_feedforward=dim_ff, dropout=dropout,
                                                       batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            
            # Output layer to map to action space
            self.output_layer = nn.Linear(d_model, self.output_dim)
            # List to store the history of (state, action, delta) tuples
            self.history = []
        else:  # In state_image mode

            if self.state_dim is None:
                raise ValueError("state_image mode requires state_dim.")
            self.use_image = (self.image_dim is not None) and (self.image_dim > 0)
            hidden_size = cfg.get('hidden_size', 128)

            if self.use_image:
                # If image is provided, concatenate state and image features
                self.mlp = nn.Sequential(
                    nn.Linear(self.state_dim + self.image_dim, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, self.output_dim)
                )
            else:
                # If no image is provided, only use state features
                self.mlp = nn.Sequential(
                    nn.Linear(self.state_dim, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, self.output_dim)
                )

    def reset(self):
        if self.mode == 'history':
            self.history = []  # Clear the history of (state, action, delta) tuples

        else:  # state_image mode
            pass  # No specific reset needed for MLP

    def _encode_sequence(self, seq):
        """
        Internal utility: Encodes a sequence of (state, action, delta) tuples into a Transformer input tensor.
        Args:
            seq: A list of tuples containing (state_vec, action_vec, delta_vec).
        Returns:
            A tensor of shape (1, L, d_model), representing the embedded sequence (batch_size=1).
        """
        L = len(seq)
        # Stack the states, actions, and deltas into lists
        states = []
        actions = []
        deltas = []
        for (s, a, d) in seq:
            # Convert inputs to tensors and flatten to 1D vectors
            s_tensor = torch.tensor(s, dtype=torch.float32) if not isinstance(s, torch.Tensor) else s.float()
            a_tensor = torch.tensor(a, dtype=torch.float32) if not isinstance(a, torch.Tensor) else a.float()
            d_tensor = torch.tensor(d, dtype=torch.float32) if not isinstance(d, torch.Tensor) else d.float()
            states.append(s_tensor.view(1, -1))
            actions.append(a_tensor.view(1, -1))
            deltas.append(d_tensor.view(1, -1))
        # Concatenate into matrices with shapes (1, L, state_dim), etc.
        state_mat = torch.cat(states, dim=1)   # (1, L, state_dim)
        action_mat = torch.cat(actions, dim=1) # (1, L, action_dim)
        delta_mat = torch.cat(deltas, dim=1)   # (1, L, action_dim)
        # Encode each component
        state_emb = self.state_encoder(state_mat)    # (1, L, d_model)
        action_emb = self.action_encoder(action_mat) # (1, L, d_model)
        delta_emb = self.delta_encoder(delta_mat)    # (1, L, d_model)
        # Combine state/action/delta embeddings to get the integrated representation for each timestep
        seq_emb = state_emb + action_emb + delta_emb  # (1, L, d_model)
        # Add positional encoding
        pos_idx = torch.arange(L).unsqueeze(0)        # (1, L)
        pos_enc = self.pos_embedding(pos_idx)         # (1, L, d_model)
        seq_emb = seq_emb + pos_enc
        return seq_emb  # (1, L, d_model)

    def compute_noise(self, state, action=None, image=None):
        """
        Compute noise increment delta based on the current input.
        In history mode, state and action must be provided; in state_image mode, state must be provided, and image if applicable.
        """
        if self.mode == 'history':
            if state is None or action is None:
                raise ValueError("History mode requires state and action to compute noise.")
            # Construct Transformer input sequence: history + current step (delta initialized to 0)
            seq = list(self.history)  # Copy existing history
            # Use a zero vector as a placeholder for the current step's delta
            zero_delta = torch.zeros(self.action_dim) if isinstance(action, torch.Tensor) \
                         else np.zeros(self.action_dim)
            seq.append((state, action, zero_delta))
            # Truncate the oldest part if the sequence length exceeds the set history length + 1
            if len(seq) > self.history_length + 1:
                seq = seq[-(self.history_length + 1):]
            # Encode the sequence as Transformer input
            seq_emb = self._encode_sequence(seq)        # shape: (1, L, d_model)
            # Process the sequence through the Transformer Encoder
            trans_out = self.transformer(seq_emb)       # Output shape: (1, L, d_model)
            # Extract the output feature corresponding to the last time step (current step)
            current_feature = trans_out[:, -1, :]       # shape: (1, d_model)
            # Map to the noise output vector
            delta_pred = self.output_layer(current_feature)  # shape: (1, output_dim)
            delta_pred = delta_pred.view(-1)            # Remove batch dimension -> (output_dim,)
            # Add the current step's state, action, and delta to the history
            # Note: delta_pred is a tensor; detach it from the computation graph when saving as numerical history
            delta_val = delta_pred.detach() if isinstance(delta_pred, torch.Tensor) else delta_pred
            self.history.append((state, action, delta_val))
            if len(self.history) > self.history_length:
                # Maintain the history list length as history_length (recent k steps, excluding the current step)
                self.history = self.history[-self.history_length:]
            return delta_pred  # Return the noise tensor for the current step

        else:  # state_image mode
            if state is None:
                raise ValueError("state_image mode requires state.")
            # Concatenate state and image features as MLP input
            state_t = state if isinstance(state, torch.Tensor) else torch.tensor(state, dtype=torch.float32)
            if self.use_image:
                if image is None:
                    raise ValueError("state_image mode requires image input.")
                image_t = image if isinstance(image, torch.Tensor) else torch.tensor(image, dtype=torch.float32)
                # Concatenate state and image vectors
                inp = torch.cat([state_t.view(1, -1), image_t.view(1, -1)], dim=1)  # (1, state_dim+image_dim)
            else:
                inp = state_t.view(1, -1)  # (1, state_dim)
            # Forward pass to compute noise output
            delta_pred = self.mlp(inp)   # (1, output_dim)
            delta_pred = delta_pred.view(-1)  # Flatten to remove batch dimension -> (output_dim,)
            return delta_pred

    def sample(self, state, action=None, image=None):
        # Directly use compute_noise to calculate and return noise
        return self.compute_noise(state, action, image)

    def forward(self, state, action=None, image=None):
        # Define forward to be compatible with nn.Module (equivalent to calling sample/compute_noise)
        return self.compute_noise(state, action, image)

    def compute_loss(self, predicted_delta, target_delta=None, reward=None):
        """
        Compute training loss:
        - If target_delta is provided (supervised learning target), use mean squared error loss.
        - If reward is provided (reinforcement learning reward), use its negative as the loss (minimizing this loss maximizes the reward).
        """
        if target_delta is not None:
            # Supervised learning: MSE loss
            return nn.functional.mse_loss(predicted_delta, target_delta)
        elif reward is not None:
            # Reinforcement learning: Negative mean reward as loss
            reward_val = reward
            if not isinstance(reward_val, torch.Tensor):
                reward_val = torch.tensor(reward_val, dtype=torch.float32)
            return -reward_val.mean()
        else:
            raise ValueError("compute_loss requires either target_delta or reward parameter.")

    def compute_reward(self, success: bool, delta: torch.Tensor):
        """
        Compute reward for the disturbance model:
        r_succ = -1 if success else 0
        r_amp = alpha * ||delta||^2
        r = r_succ + r_amp

        Args:
            success (bool): Whether the task succeeded
            delta (Tensor): Noise applied at this timestep

        Returns:
            float: Reward signal
        """
        r_succ = -1.0 if success else 0.0
        r_amp = self.alpha * delta.pow(2).sum().item()
        reward = r_succ + r_amp
        return reward
