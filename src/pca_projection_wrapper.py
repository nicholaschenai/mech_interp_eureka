"""
# Example usage:

# Path to combined PCA components file
pca_components_file = 'pca_results/pca_all_layers.npz'

# Optionally, provide specific layer names to use
# If not provided, all layers in the file will be used
layer_names = [
    'actor_mlp_0',  # First layer of actor_mlp
    'actor_mlp_1',  # Second layer of actor_mlp
    'mu',           # Output layer
    'value'         # Value layer
]

# Specify number of components per layer
pca_n_components = {
    'actor_mlp_0': 20,
    'actor_mlp_1': 15,
    'mu': 5,
    'value': 3
}

# Load your model (either from scratch or from pickle)
model = load_your_model()  # Replace with actual loading code

# Apply PCA
model = wrap_model_with_pca(
    model, 
    pca_components_file=pca_components_file,
    layer_names=layer_names,
    pca_n_components=pca_n_components
)

# Now model will apply PCA bottlenecks during inference
output = model(some_input)

# Experiment with different numbers of components
model.set_n_components('actor_mlp_0', 10)  # Change actor_mlp_0 to use 10 components
model.set_all_components(50)               # Use 50 components for all layers

# Remove PCA if needed
model.remove_pca()
"""
import torch.nn as nn

from typing import Dict, List, Optional

from .pca_proj_manager import PCAProjManager


class PCANetworkWrapper(nn.Module):
    """
    Wrapper for ModelA2CContinuousLogStd.Network that applies PCA bottlenecks
    to specific layers during forward passes.
    
    This wrapper preserves the original network structure but adds PCA projections
    at configurable points in the network.
    """
    def __init__(self, 
                a2c_network, 
                pca_components_file: str,
                layer_names: Optional[List[str]] = None,
                pca_n_components: Optional[Dict[str, int]] = None,
                layer_mapping: Optional[Dict[str, str]] = None):
        """
        Args:
            a2c_network: The original A2C network to wrap
            pca_components_file: Path to the combined .npz file containing PCA components
            layer_names: List of layer names to use from the file. If None, auto-detect.
            pca_n_components: Dict mapping layer names to number of components to use
            layer_mapping: Dict mapping layer names to attribute paths in the a2c_network
        """
        super().__init__()
        
        self.a2c_network = a2c_network
        
        self.pca_manager = PCAProjManager(
            components_file=pca_components_file,
            layer_names=layer_names,
            n_components=pca_n_components
        )
        
        self.pca_manager.apply_to_model(self, layer_mapping)
    
    def forward(self, input_dict):
        """
        Forward pass that delegates to the wrapped network.
        
        The PCA hooks will automatically be applied during the a2c_network's forward pass
        because they're registered to the appropriate submodules.
        
        Args:
            input_dict: Input dictionary as expected by the a2c_network
            
        Returns:
            Same output structure as the original a2c_network
        """
        return self.a2c_network(input_dict)
    
    def set_n_components(self, layer_name: str, n_components: int):
        """
        Set the number of principal components to use for a specific layer.
        
        Args:
            layer_name: Name of the layer to modify
            n_components: Number of components to use
        """
        if self.pca_manager:
            self.pca_manager.update_n_components(layer_name, n_components)
    
    def set_all_components(self, n_components: int):
        """Set the same number of components for all hooks"""
        if self.pca_manager:
            self.pca_manager.update_all_components(n_components)
    
    def remove_pca(self):
        """Remove all PCA hooks from the model"""
        if self.pca_manager:
            self.pca_manager.remove_hooks()


def wrap_model_with_pca(model, 
                      pca_components_file: str,
                      layer_names: Optional[List[str]] = None,
                      pca_n_components: Optional[Dict[str, int]] = None,
                      layer_mapping: Optional[Dict[str, str]] = None):
    """
    Helper function to wrap an existing model with PCA projections.
    
    Args:
        model: The ModelA2CContinuousLogStd instance to wrap
        pca_components_file: Path to the combined .npz file containing PCA components
        layer_names: List of layer names to use from the file. If None, auto-detect.
        pca_n_components: Dict mapping layer names to number of components to use
        layer_mapping: Dict mapping layer names to attribute paths in model.a2c_network
        
    Returns:
        Modified model with PCA hooks applied
    """
    # Access the network inside the model
    if hasattr(model, 'a2c_network'):
        model.a2c_network = PCANetworkWrapper(
            model.a2c_network,
            pca_components_file=pca_components_file,
            layer_names=layer_names,
            pca_n_components=pca_n_components,
            layer_mapping=layer_mapping
        )
        
        # Add convenience methods to the model
        model.set_n_components = lambda layer, n: model.a2c_network.set_n_components(layer, n)
        model.set_all_components = lambda n: model.a2c_network.set_all_components(n)
        model.remove_pca = lambda: model.a2c_network.remove_pca()
        
        return model
    else:
        raise ValueError("Model does not have an a2c_network attribute")
