"""
Model hook utilities for activation extraction.

This module is responsible for hooking into model layers and 
extracting activations during forward passes.

Responsibilities:
- Hook into model layers to extract activations
- Process observations and run model forward passes
- Capture activations from specified model layers
- Manage the hook lifecycle (register and remove hooks)
"""
import torch
from typing import Dict, List, Tuple, Any

from .a2c_model_interface import A2CModelInterface


class ModelHook:
    def __init__(self, model: Any):
        """Initialize with A2C model.
        
        Args:
            model: The A2C model to hook into for activation extraction
        """
        self.model = model
        self.hooks = []  # List to store hook handles
        self.activations = {}  # Dictionary to store activations
        self.model_interface = A2CModelInterface(model)
    
    def register_hooks(self, layers: List[str] = None) -> None:
        """Set up hooks on model components.
        
        Args:
            layers: List of layer names to hook into. If None, hooks all available layers.
                   Options: 'actor_mlp', 'mu', 'value'
        """
        self.clear_hooks()  # Remove any existing hooks
        
        if layers is None:
            # Note: activations for value and mu are Identity so we don't need to hook them
            layers = ['actor_mlp', 'mu', 'value']
        
        # Register hooks for actor_mlp layers
        if 'actor_mlp' in layers:
            actor_mlp = self.model_interface.get_actor_mlp()
            for i, module in enumerate(actor_mlp):
                hook = module.register_forward_hook(
                    lambda mod, inp, out, name=f"actor_mlp_{i}": self.activations.update({name: out.detach()})
                )
                self.hooks.append(hook)
        
        # Register hook for mu layer
        if 'mu' in layers:
            mu_layer = self.model_interface.get_mu_layer()
            hook = mu_layer.register_forward_hook(
                lambda mod, inp, out, name="mu": self.activations.update({name: out.detach()})
            )
            self.hooks.append(hook)
        
        # Register hook for value layer
        if 'value' in layers:
            value_layer = self.model_interface.get_value_layer()
            hook = value_layer.register_forward_hook(
                lambda mod, inp, out, name="value": self.activations.update({name: out.detach()})
            )
            self.hooks.append(hook)
    
    def get_activations_and_outputs(self, observation: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Run forward pass and return both activations and model outputs.
        
        Args:
            observation: Tensor containing the observation data
            
        Returns:
            Tuple of (activations_dict, outputs_dict)
        """
        self.activations = {}  # Clear previous activations
        
        # Run forward pass and get outputs (normalization handled inside forward)
        outputs = self.model_interface.forward(observation)
        
        # Create a copy of activations
        activations = {k: v.clone() for k, v in self.activations.items()}
        
        return activations, outputs
    
    def clear_hooks(self) -> None:
        """Remove all hooks from model."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
