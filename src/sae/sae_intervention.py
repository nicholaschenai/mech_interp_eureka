"""
Manages multiple SAE hooks across the network
Provides interface for ablating specific features
Coordinates activation/deactivation of hooks
Handles registration of hooks to model layers
TODO: the modified features thing is not used anywhere, remove it
"""
import torch

import torch.nn as nn

from typing import Dict, List, Optional

from ..activation_intervention import ActivationIntervention
from .sae_model import SAEModel
from .sae_hook import SAEHook


class SAEIntervention(ActivationIntervention):
    """
    Class for performing causal interventions on SAE features.
    
    This class builds on the base ActivationIntervention to provide specific
    functionality for ablating sparse autoencoder features and measuring their
    effects on downstream layers or model outputs.
    """
    def __init__(self, model: nn.Module, layer_names: List[str], sae_models: Dict[str, SAEModel]):
        super().__init__(layer_names)
        self.hooks: Dict[str, SAEHook] = {}
        self.model = model
        self.sae_models = sae_models
        
        # Register hooks after initialization
        self.register_hooks(model)
        
    def _create_hook_for_layer(self, layer_name: str, module: nn.Module) -> SAEHook:
        if layer_name not in self.sae_models:
            raise ValueError(f"No SAE model found for layer {layer_name}")
            
        hook = SAEHook(
            sae_model=self.sae_models[layer_name],
            layer_name=layer_name,
        )
        
        return hook
    
    def register_hooks(self, model: nn.Module, layer_mapping: Dict[str, str] = None) -> None:
        """
        Register SAE hooks on target layers.
        
        For SAE intervention, we need hooks on all target layers, but
        only one layer will be actively modified during an intervention.
        """
        super().register_hooks(model, layer_mapping)
        # After registering, make sure all hooks start deactivated
        self.clear_ablation()
    
    def ablate_feature(self, 
                      source_layer: str, 
                      feature_idx: int, 
                      ) -> Dict[str, torch.Tensor]:
        """
        Ablate (zero out) a specific SAE feature and observe downstream effects.
        
        Args:
            source_layer: Layer containing the feature to ablate
            feature_idx: Index of the SAE feature to ablate
            inputs: Model input tensor(s)
            
        Returns:
            Dictionary of resulting activations for downstream layers
        """
        # Create an ablation function that zeros out the specific feature
        def ablate_specific_feature(features):
            # Clone to avoid modifying the original
            modified = features.clone()
            # Zero out the specified feature
            modified[:, feature_idx] = 0
            return modified
        
        # Activate only the target hook
        for layer_name, hook in self.hooks.items():
            if layer_name == source_layer:
                hook.activate(ablate_specific_feature)
            else:
                hook.deactivate()
    
    def clear_ablation(self):
        """Deactivate all hooks"""
        for hook in self.hooks.values():
            hook.deactivate()

    def start_accumulating_features(self):
        """Start accumulating features across forward passes for all hooks"""
        for hook in self.hooks.values():
            hook.start_accumulating()
    
    def stop_accumulating_features(self):
        """Stop accumulating features for all hooks"""
        for hook in self.hooks.values():
            hook.stop_accumulating()
    
    def clear_accumulated_features(self):
        """Clear all accumulated features for all hooks"""
        for hook in self.hooks.values():
            hook.clear_accumulated_features()
    
    def get_accumulated_features(self, layer_name: Optional[str] = None):
        """
        Return accumulated features for specified layer or all layers.
        
        Args:
            layer_name: Optional layer name. If provided, returns features
                       only for that layer. Otherwise, returns for all layers.
                       
        Returns:
            Dictionary mapping layer names to accumulated feature tensors,
            or a single tensor if layer_name is specified.
        """
        if layer_name is not None:
            if layer_name in self.hooks:
                return self.hooks[layer_name].get_accumulated_features()
            return None
        
        accumulated = {}
        for layer_name, hook in self.hooks.items():
            features = hook.get_accumulated_features()
            if features is not None:
                accumulated[layer_name] = features
        
        return accumulated
    
    def get_accumulated_modified_features(self, layer_name: Optional[str] = None):
        """
        Return accumulated modified features for specified layer or all layers.
        
        Args:
            layer_name: Optional layer name. If provided, returns features
                       only for that layer. Otherwise, returns for all layers.
                       
        Returns:
            Dictionary mapping layer names to accumulated modified feature tensors,
            or a single tensor if layer_name is specified.
        """
        if layer_name is not None:
            if layer_name in self.hooks:
                return self.hooks[layer_name].get_accumulated_modified_features()
            return None
        
        accumulated = {}
        for layer_name, hook in self.hooks.items():
            features = hook.get_accumulated_modified_features()
            if features is not None:
                accumulated[layer_name] = features
        
        return accumulated
