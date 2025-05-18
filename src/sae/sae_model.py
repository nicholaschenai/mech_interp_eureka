import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from typing import Dict, Tuple


class SAEModel(nn.Module):
    """
    Sparse Autoencoder model implementation with optional tied weights for neuron interpretation.
    
    This model is designed for mechanistic interpretability of neural networks,
    particularly focusing on extracting interpretable features from activation layers.
    The tied weights architecture ensures that each feature (decoder row) corresponds
    directly to a pattern in the original activation space, making interpretation
    more straightforward.
    
    For robot control networks, features may represent motor primitives, spatial awareness,
    phase-specific strategies, or other interpretable control elements.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int,
                 l1_coef: float = 0.2, tied_weights: bool = True, weight_normalize_eps: float = 1e-8):
        """
        Initialize SAE with input dimension, hidden dimension, sparsity target and L1 coefficient.
        
        Args:
            input_dim: Dimension of input activations (neurons in the layer being analyzed)
            hidden_dim: Dimension of sparse feature space (typically 2-3x input_dim)
            l1_coef: L1 regularization coefficient to encourage sparsity
            tied_weights: Whether to use tied weights (decoder = encoder.T)
            weight_normalize_eps: Small constant for numerical stability in weight normalization
        """
        super(SAEModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.l1_coef = l1_coef
        self.tied_weights = tied_weights
        self.weight_normalize_eps = weight_normalize_eps
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.encoder = nn.Linear(input_dim, hidden_dim)
        nn.init.kaiming_normal_(self.encoder.weight, nonlinearity='relu')
        
        self.encoder_bias = nn.Parameter(torch.zeros(hidden_dim))
        # Initialize decoder bias (we'll manage decoder weights separately for normalization)
        self.decoder_bias = nn.Parameter(torch.zeros(input_dim))
        
        if not tied_weights:
            self.decoder_weight = nn.Parameter(
                nn.init.kaiming_normal_(torch.empty((input_dim, hidden_dim)), nonlinearity='linear')
            )
        else:
            self.decoder_weight = None
        
        self.mse_loss = nn.MSELoss(reduction='none')
        
        self.to(self.device)
        
    @property
    def W_dec(self) -> torch.Tensor:
        if self.tied_weights:
            return self.encoder.weight.t()
        else:
            return self.decoder_weight
            
    @property
    def W_dec_normalized(self) -> torch.Tensor:
        """
        Returns decoder weights normalized over the input dimension.
        
        Normalizing the decoder weights ensures that each feature captures
        a normalized direction in activation space, making features more 
        interpretable and comparable.
        """
        return self.W_dec / (self.W_dec.norm(dim=1, keepdim=True) + self.weight_normalize_eps)
    
    def normalize_decoder_weights(self) -> None:
        self.W_dec.data = self.W_dec_normalized.data

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input activations (batch_size, input_dim)
            
        Returns:
            Encoded sparse activations after ReLU (batch_size, hidden_dim)
        """
        # Center the input by subtracting decoder bias
        x_centered = x - self.decoder_bias
        
        # Compute latent activations
        acts_pre = self.encoder(x_centered) + self.encoder_bias
        acts_post = F.relu(acts_pre)
        
        return acts_post
    
    def decode(self, acts_post: torch.Tensor) -> torch.Tensor:
        x_reconstructed = F.linear(acts_post, self.W_dec_normalized) + self.decoder_bias
        return x_reconstructed
    
    def forward(self, x: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input activations (batch_size, input_dim)
            
        Returns:
            Tuple containing:
                - loss_dict: Dictionary of loss components
                - loss: Combined loss
                - acts_post: Hidden activations after ReLU
                - x_reconstructed: Reconstructed input
        """
        acts_post = self.encode(x)
        x_reconstructed = self.decode(acts_post)
        
        L_reconstruction = self.mse_loss(x_reconstructed, x).mean(dim=1)
        L_sparsity = F.l1_loss(acts_post, torch.zeros_like(acts_post), reduction='none').mean(dim=1) * self.l1_coef
        
        loss_dict = {
            "L_reconstruction": L_reconstruction,
            "L_sparsity": L_sparsity
        }
        loss = L_reconstruction + L_sparsity
        
        return loss_dict, loss, acts_post, x_reconstructed
    
    def save(self, path: str) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: str) -> 'SAEModel':
        with open(path, 'rb') as f:
            model = pickle.load(f)
        return model 
