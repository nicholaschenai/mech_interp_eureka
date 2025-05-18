import numpy as np

from typing import Dict, List, Optional

from .pca_hook import PCAHook
from .activation_intervention import ActivationIntervention


class PCAProjManager(ActivationIntervention):
    """
    Manager class to apply PCA hooks to multiple layers of a model.
    """
    def __init__(self, 
                components_file: str, 
                layer_names: Optional[List[str]] = None,
                n_components: Optional[Dict[str, int]] = None):
        """
        Args:
            components_file: Path to the combined .npz file containing PCA components
            layer_names: List of layer names to use from the file. If None, auto-detect.
            n_components: Dict mapping layer names to number of components to use
        """
        super().__init__(layer_names)
        self.hooks: Dict[str, PCAHook] = {}
        self.components_file = components_file
        self.n_components = n_components or {}
        
        # Auto-detect layers if not specified
        if layer_names is None:
            all_data = np.load(components_file)
            layer_names = []
            
            for key in all_data.keys():
                if key.endswith('_components'):
                    layer = key[:-11]  # Remove '_components' suffix
                    layer_names.append(layer)
            
            if not layer_names:
                raise ValueError(f"Could not find any components in {components_file}")
                
            print(f"Automatically detected layers: {layer_names}")
        
    def _create_hook_for_layer(self, layer_name, target_module):
        n_comp = self.n_components.get(layer_name)
        hook = PCAHook(self.components_file, layer_name=layer_name, n_components=n_comp)
        return hook

    def apply_to_model(self, model, layer_mapping: Dict[str, str] = None):
        """
        Apply PCA hooks to model layers.
        
        Args:
            model: PyTorch model
            layer_mapping: Dict mapping layer names to model attribute paths.
                          For sequential modules, can use format "module[index]".
                          If None, attempts to generate mapping automatically.
                           
        Returns:
            Self (for chaining)
        """
        self.register_hooks(model, layer_mapping)
        return self
    
    def update_n_components(self, layer_name: str, n_components: int):
        """
        Update the number of components for a specific layer.
        
        Args:
            layer_name: Name of the layer to update
            n_components: New number of components
        """
        if layer_name in self.hooks:
            self.hooks[layer_name].set_n_components(n_components)
            self.n_components[layer_name] = n_components
    
    def update_all_components(self, n_components: int):
        """Set the same number of components for all hooks"""
        for layer_name in self.hooks:
            self.update_n_components(layer_name, n_components)

    # Minor note: older version's remove_hooks doesnt reset self.hooks
