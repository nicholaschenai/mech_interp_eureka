"""
Wrapper for applying SAE-based causal interventions to neural network models.

This module provides a wrapper that enables SAE-based feature ablation
at specific layers in neural networks, without modifying the original model.
TODO: the modified features thing is not used anywhere, remove it
"""
import torch.nn as nn

from typing import Dict, List, Optional

from .sae_model import SAEModel
from .sae_intervention import SAEIntervention


class SAENetworkWrapper(nn.Module):
    """
    Wrapper for neural networks that applies SAE-based interventions
    to specific layers during forward passes.
    
    This wrapper preserves the original network structure but adds SAE-based
    feature ablation at configurable points in the network.
    """
    def __init__(self, 
                a2c_network: nn.Module, 
                sae_models: Dict[str, SAEModel],
                layer_names: List[str],
                layer_mapping: Optional[Dict[str, str]] = None):
        """
        Args:
            a2c_network: The original A2C network to wrap
            sae_models: Dictionary mapping layer names to their SAE models
            layer_names: List of layer names to apply interventions to
            layer_mapping: Dict mapping layer names to attribute paths in the a2c_network
        """
        super().__init__()
        
        self.a2c_network = a2c_network
        self.sae_manager = SAEIntervention(self, layer_names, sae_models)
        
    def forward(self, input_dict):
        """
        Forward pass that delegates to the wrapped network.
        
        The SAE hooks will automatically be applied during the a2c_network's forward pass
        because they're registered to the appropriate submodules.
        
        Args:
            input_dict: Input dictionary as expected by the a2c_network
            
        Returns:
            Same output structure as the original a2c_network
        """
        return self.a2c_network(input_dict)
    
    def ablate_feature(self, layer_name: str, feature_idx: int):
        self.sae_manager.ablate_feature(layer_name, feature_idx)
    
    def clear_ablation(self):
        self.sae_manager.clear_ablation()
    
    # Feature accumulation methods
    def start_accumulating_features(self):
        """Start accumulating features for all layers"""
        self.sae_manager.start_accumulating_features()
    
    def stop_accumulating_features(self):
        """Stop accumulating features for all layers"""
        self.sae_manager.stop_accumulating_features()
    
    def clear_accumulated_features(self):
        """Clear all accumulated features"""
        self.sae_manager.clear_accumulated_features()
    
    def get_accumulated_features(self, layer_name=None):
        """Get accumulated features for all layers or specific layer"""
        return self.sae_manager.get_accumulated_features(layer_name)
    
    def get_accumulated_modified_features(self, layer_name=None):
        """Get accumulated modified features for all layers or specific layer"""
        return self.sae_manager.get_accumulated_modified_features(layer_name)


def wrap_model_with_sae(model: nn.Module, 
                      sae_models: Dict[str, SAEModel],
                      layer_names: List[str],
                      layer_mapping: Optional[Dict[str, str]] = None) -> nn.Module:
    """
    Helper function to wrap an existing model with SAE interventions.
    
    Args:
        model: The model instance to wrap
        sae_models: Dictionary mapping layer names to their SAE models
        layer_names: List of layer names to apply interventions to
        layer_mapping: Dict mapping layer names to attribute paths in model.a2c_network
        
    Returns:
        Modified model with SAE hooks applied
    """
    # Access the network inside the model
    if hasattr(model, 'a2c_network'):
        model.a2c_network = SAENetworkWrapper(
            model.a2c_network,
            sae_models=sae_models,
            layer_names=layer_names,
            layer_mapping=layer_mapping
        )
        
        # Add convenience methods to the model
        model.ablate_feature = lambda layer, idx: model.a2c_network.ablate_feature(layer, idx)
        model.clear_ablation = lambda: model.a2c_network.clear_ablation()
        model.start_accumulating_features = lambda: model.a2c_network.start_accumulating_features()
        model.stop_accumulating_features = lambda: model.a2c_network.stop_accumulating_features()
        model.clear_accumulated_features = lambda: model.a2c_network.clear_accumulated_features()
        model.get_accumulated_features = lambda layer=None: model.a2c_network.get_accumulated_features(layer)
        model.get_accumulated_modified_features = lambda layer=None: model.a2c_network.get_accumulated_modified_features(layer)
        
        return model
    else:
        raise ValueError("Model does not have an a2c_network attribute")
