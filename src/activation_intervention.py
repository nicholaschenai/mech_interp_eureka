import re
import torch.nn as nn

from typing import Dict, List, Callable, Optional


class ActivationIntervention:
    """
    Base class for performing interventions on neural network activations.
    
    This class provides the foundation for intercepting, modifying, and analyzing
    activations at specific layers in a neural network. It is designed to be extended
    by specific intervention types (e.g., SAE, PCA).
    """
    
    def __init__(self, layer_names: List[str]):
        """
        Args:
            layer_names: List of layer names where interventions will be performed
        """
        self.layer_names = layer_names
        self.hooks = {}
        self.handles = []
        self.activation_cache = {}
        
    def register_hooks(self, model, layer_mapping: Dict[str, str] = None) -> None:
        """
        Register forward hooks on target layers to capture activations.
        
        This method sets up hooks that will store layer outputs for later retrieval
        and modification. Concrete implementations should create hooks of the
        appropriate type and register them using _register_hook_for_module.
        """
        # Default implementation - finds modules by path and registers hooks
        # Concrete subclasses should implement this method to create specific hooks
        if layer_mapping is None:
            layer_mapping = self._create_default_mapping(model, self.layer_names)
        
        for layer_name in self.layer_names:
            if layer_name not in layer_mapping:
                print(f"Warning: No layer mapping for '{layer_name}', skipping")
                continue
                
            # Get the target module
            attr_path = layer_mapping[layer_name]
            target_module = self._get_module_by_path(model, attr_path)
            
            if target_module is None:
                raise ValueError(f"Could not find module at path '{attr_path}'")
            
            # Concrete subclasses need to implement _create_hook_for_layer
            hook = self._create_hook_for_layer(layer_name, target_module)
            self.hooks[layer_name] = hook
            
            # Register the hook
            handle = target_module.register_forward_hook(hook)
            self.handles.append(handle)
    
    def _create_hook_for_layer(self, layer_name: str, module: nn.Module) -> Callable:
        """
        Args:
            layer_name: Name of the layer
            module: The module to hook
            
        Returns:
            A hook object or function
        """
        raise NotImplementedError("Subclasses must implement _create_hook_for_layer")
    
    def _create_default_mapping(self, model: nn.Module, layer_names: List[str]) -> Dict[str, str]:
        """
        Create a default mapping from layer names to model paths.
        Handles special cases like actor_mlp_0 -> a2c_network.actor_mlp[0]
        
        Args:
            model: The model to map against
            layer_names: List of layer names to map
            
        Returns:
            Dict mapping layer names to attribute paths
        """
        mapping = {}
        
        # Regular expression to match indexed layers (e.g., actor_mlp_0)
        indexed_pattern = re.compile(r'^(.+)_(\d+)$')
        
        for layer_name in layer_names:
            # Check if this is an indexed layer
            match = indexed_pattern.match(layer_name)
            if match:
                base_name, index = match.groups()
                # Format: a2c_network.base_name[index]
                if hasattr(model, 'a2c_network'):
                    if hasattr(model.a2c_network, base_name):
                        mapping[layer_name] = f"a2c_network.{base_name}[{index}]"
                    else:
                        print(f"Warning: Could not find base module '{base_name}' for '{layer_name}'")
                else:
                    # Try without a2c_network prefix
                    if hasattr(model, base_name):
                        mapping[layer_name] = f"{base_name}[{index}]"
                    else:
                        print(f"Warning: Could not find base module '{base_name}' for '{layer_name}'")
            else:
                # Regular layer, standard mapping
                if hasattr(model, 'a2c_network'):
                    if hasattr(model.a2c_network, layer_name):
                        mapping[layer_name] = f"a2c_network.{layer_name}"
                    else:
                        print(f"Warning: Could not find module '{layer_name}'")
                else:
                    # Try without a2c_network prefix
                    if hasattr(model, layer_name):
                        mapping[layer_name] = layer_name
                    else:
                        print(f"Warning: Could not find module '{layer_name}'")
        
        return mapping
    
    def _get_module_by_path(self, model: nn.Module, path: str) -> Optional[nn.Module]:
        """
        Get a module by its attribute path, supporting indexed access for Sequential modules.
        
        Args:
            model: The model to search in
            path: Path to the module, with support for indexed access using brackets
                 e.g., "a2c_network.actor_mlp[0]" gets the first layer of actor_mlp
                 
        Returns:
            The requested module or None if not found
        """
        # Handle indexed access (e.g., "actor_mlp[0]")
        if '[' in path and ']' in path:
            # Extract the base path and index
            parts = path.split('[')
            base_path = parts[0]
            index_str = parts[1].split(']')[0]
        
            index = int(index_str)
            # Get the base module
            base_module = self._get_module_by_path(model, base_path)
            
            if base_module is not None and hasattr(base_module, '__getitem__'):
                return base_module[index]
            else:
                raise ValueError(f"Module {base_path} is not indexable")
        
        # Regular attribute path traversal
        parts = path.split('.')
        current = model
        
        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
            else:
                return None
        
        return current
    
    def remove_hooks(self) -> None:
        """Remove all registered hooks from the model."""
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.hooks = {}
