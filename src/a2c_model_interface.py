"""
A2C model interface for mechanistic interpretability.

This module provides an interface to the A2C model structure,
abstracting the details of how to access different components
of the model architecture.

Responsibilities:
- Access the nested structure of the A2C network
- Provide standardized access to actor_mlp, mu layer, and value layer
- Handle observation normalization
- Provide a unified forward pass interface
"""
import torch
from typing import Dict, Any


class A2CModelInterface:
    def __init__(self, model: Any):
        """
        Args:
            model: The A2C model to interface with
        """
        self.model = model
        self.model.eval()

    def get_actor_mlp(self) -> torch.nn.Module:
        return self.model.a2c_network.actor_mlp
    
    def get_mu_layer(self) -> torch.nn.Module:
        return self.model.a2c_network.mu
    
    def get_value_layer(self) -> torch.nn.Module:
        return self.model.a2c_network.value
    
    def normalize_observation(self, observation: torch.Tensor) -> torch.Tensor:
        return self.model.norm_obs(observation)
    
    def forward(self, observation: torch.Tensor) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            normalized_obs = self.normalize_observation(observation)

            input_dict = {"obs": normalized_obs}
            
            # Run forward pass through a2c_network
            mu, logstd, value, states = self.model.a2c_network(input_dict)
            
            return {'mu': mu, 'value': value}
