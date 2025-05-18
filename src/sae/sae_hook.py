import torch

from typing import Optional, Callable

from src.sae.sae_model import SAEModel


class SAEHook:
    """
    Forward hook for PyTorch modules that enables SAE-based interventions.
    
    This hook intercepts the output of a module, allows for modifications to
    the activation through the SAE feature space, and can continue the forward
    pass with the modified activations.
    """
    def __init__(self,
                 sae_model: SAEModel,
                 layer_name: str,
                 intervention_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                 ):
        """
        Args:
            sae_model: Trained SAE model for this layer
            layer_name: Name of the layer for identification
            intervention_fn: Optional function to transform the SAE features
        """
        self.sae_model = sae_model
        self.layer_name = layer_name
        self.intervention_fn = intervention_fn
        
        # For accumulating features across batches
        self.accumulate_features = False
        self.accumulated_features = []
        self.accumulated_modified_features = []
        
        # For accumulating output tensors (previous layer activations)
        self.accumulate_outputs = False
        self.accumulated_outputs = []
        
        # future: reconstruction errors already computed in scripts/sae/train_sae.py, is it possible to reuse them? 
        # but caveat that it is for full dataset, whereas over here we might wanna do some batching so needs reshaping
        # Store reconstruction error once computed
        # self.reconstruction_error = None
        
        # Flag to determine whether to perform intervention
        self.active = False
        
    def activate(self, intervention_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None):
        self.active = True
        if intervention_fn is not None:
            self.intervention_fn = intervention_fn
    
    def deactivate(self):
        self.active = False
        self.intervention_fn = None
        
    def encode(self, activations):
        return self.sae_model.encode(activations)
        
    def decode(self, features):
        return self.sae_model.decode(features)
    
    def start_accumulating(self):
        """Start accumulating features across forward passes"""
        self.accumulate_features = True
        self.accumulated_features = []
        self.accumulated_modified_features = []
    
    def stop_accumulating(self):
        """Stop accumulating features"""
        self.accumulate_features = False
    
    def clear_accumulated_features(self):
        self.accumulated_features = []
        self.accumulated_modified_features = []
    
    def get_accumulated_features(self):
        if not self.accumulated_features:
            return None
        return torch.cat(self.accumulated_features, dim=0)
    
    def get_accumulated_modified_features(self):
        if not self.accumulated_modified_features:
            return None
        return torch.cat(self.accumulated_modified_features, dim=0)
    
    # Similar methods for outputs, only to by used for debugging
    def start_accumulating_outputs(self):
        self.accumulate_outputs = True
        self.accumulated_outputs = []

    def stop_accumulating_outputs(self):
        self.accumulate_outputs = False

    def clear_accumulated_outputs(self):
        self.accumulated_outputs = []
    
    def get_accumulated_outputs(self):
        if not self.accumulated_outputs:
            return None
        return torch.cat(self.accumulated_outputs, dim=0)
        
    def __call__(self, module, input_tensor, output_tensor):
        """
        Apply the SAE transformation and optional intervention.
        Intervention will add back reconstruction error
        
        Args:
            module: The module this hook is registered to
            input_tensor: Input to the module (not used)
            output_tensor: Output from the module
            
        Returns:
            Original or transformed tensor based on hook state
        """
        # Accumulate output tensors if enabled, for testing
        if self.accumulate_outputs:
            self.accumulated_outputs.append(output_tensor.detach().cpu())
            
        # Always encode to get features
        features = self.encode(output_tensor)
        
        # Accumulate features if enabled
        if self.accumulate_features:
            self.accumulated_features.append(features.detach().cpu())
        
        # If not active, return original output without intervention
        if not self.active:
            return output_tensor
        
        # Apply intervention if function is provided
        if self.intervention_fn is not None:
            modified_features = self.intervention_fn(features)
        else:
            modified_features = features
            
        # Decode back to activation space using model's method
        reconstructed = self.decode(modified_features)
        """
        # Add back reconstruction error
        
        # we recompute this each time not efficient 
        # if need be, can store once at start of expt (but this is future TODO)
        
        # MEGA NOTE: reconstruction error refers to that of totally unablated case, 
        # so the below only applies to our scenario of adding back reconstruction error 
        # on source layer assuming the prior layers have not been ablated. 
        # if there are multiple ablated layers, this does not hold!
        """
        reconstruction_error = output_tensor - self.decode(features)
        reconstructed += reconstruction_error
        
        return reconstructed
