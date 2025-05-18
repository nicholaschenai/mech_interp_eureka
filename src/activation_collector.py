"""
class with activation collection methods

This module is responsible for processing pre-recorded observation trajectories,
collecting activations from the A2C model, and providing a structured way to
store them along with metadata.

Responsibilities:
- Store activations with their corresponding metadata from pre-recorded observations
- Organize data to preserve the relationship between observations and activations
- Maintain tensor structure throughout for efficient operations
"""
import pickle
import numpy as np
import torch

from typing import Dict, Any, Optional

from .activation_dataset import ActivationDataset


class ActivationCollector:
    """
    This class collects PyTorch tensor activations from neural networks during inference
    and converts them to NumPy arrays for more efficient storage and broader compatibility 
    with analysis tools.
    """
    
    def __init__(self):
        self.observations: Optional[np.ndarray] = None  # Will be a numpy array of shape [time, batch, obs_dim]
        self.activations: Dict[str, np.ndarray] = {}     # Dict of {layer_name: numpy array of shape [time, batch, ...]}
        # Dict of {key: numpy array or dict of numpy arrays}. Usually contains features, phase_masks, etc.
        self.metadata: Dict[str, np.ndarray | Dict[str, np.ndarray]] = {}
    
    # Helper methods
    def prepare_for_collection(self):
        """Prepare dataset for timestep-by-timestep collection."""
        self._obs_list = []
        self._activations_dict = {}
    
    # Collection methods
    def add_batch_timestep(self, 
                          observations: torch.Tensor, 
                          activations: Dict[str, torch.Tensor]) -> None:
        """Add a batch of activations for a specific timestep.
        
        Args:
            observations: Tensor of shape [batch, obs_dim]
            activations: Dictionary of activations for this timestep
        """
        if not hasattr(self, '_obs_list'):
            self.prepare_for_collection()
        
        # Ensure tensors are detached and on CPU
        observations = observations.detach().cpu()
        
        # Add to lists in time order (assuming add_batch_timestep is called in order)
        self._obs_list.append(observations)
        
        # Store activations
        for layer_name, activation in activations.items():
            if layer_name not in self._activations_dict:
                self._activations_dict[layer_name] = []
            self._activations_dict[layer_name].append(activation.detach().cpu())
    
    def add_metadata(self, metadata: Dict[str, Any]) -> None:
        """Add metadata for the entire trajectory at once.
        
        This should be called after all timesteps have been added using add_batch_timestep.
        Converts PyTorch tensors to NumPy arrays.
        
        Args:
            metadata: Dictionary of metadata (features, phase_masks, etc.)
        """
        # Store metadata directly
        self.metadata = {}
        for key, value in metadata.items():
            if isinstance(value, dict):
                # Handle nested dictionaries
                self.metadata[key] = {}
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, torch.Tensor):
                        self.metadata[key][subkey] = subvalue.detach().cpu().numpy()
                    elif isinstance(subvalue, np.ndarray):
                        self.metadata[key][subkey] = subvalue
                    else:
                        self.metadata[key][subkey] = np.array(subvalue)
            elif isinstance(value, torch.Tensor):
                self.metadata[key] = value.detach().cpu().numpy()
            elif isinstance(value, np.ndarray):
                self.metadata[key] = value
            else:
                self.metadata[key] = np.array(value)
    
    def finalize_collection(self) -> None:
        """Finalize dataset after collecting batches timestep by timestep.
        
        This combines all the batches collected via add_batch_timestep into tensors,
        then converts them to numpy arrays.
        """
        if not hasattr(self, '_obs_list') or not self._obs_list:
            raise ValueError("No batches have been added yet")
        
        # Get total time steps (no sorting needed since we assume data is in order)
        self.time_steps = len(self._obs_list)
        
        # Stack observations into a tensor then convert to numpy
        self.observations = torch.stack(self._obs_list).numpy()
        
        # Stack activations into tensors then convert to numpy
        self.activations = {}
        for layer, act_list in self._activations_dict.items():
            self.activations[layer] = torch.stack(act_list).numpy()
        
        # Convert any metadata tensors to numpy
        for key, value in self.metadata.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, torch.Tensor):
                        self.metadata[key][subkey] = subvalue.numpy()
            elif isinstance(value, torch.Tensor):
                self.metadata[key] = value.numpy()
        
        # Print summary of collected activations
        print("\nActivation shapes in dataset:")
        for layer_name, activation_array in self.activations.items():
            print(f"{layer_name}: {tuple(activation_array.shape)}")
            
        # Clean up temporary storage
        del self._obs_list, self._activations_dict

    # file methods
    def save(self, filepath: str) -> None:
        # Convert tensors to numpy arrays
        numpy_observations = self.observations.numpy() if isinstance(self.observations, torch.Tensor) else self.observations
        
        numpy_activations = {}
        for layer_name, activation in self.activations.items():
            numpy_activations[layer_name] = activation.numpy() if isinstance(activation, torch.Tensor) else activation
        
        numpy_metadata = {}
        for key, value in self.metadata.items():
            if isinstance(value, dict):
                numpy_metadata[key] = {}
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, torch.Tensor):
                        numpy_metadata[key][subkey] = subvalue.numpy()
                    else:
                        numpy_metadata[key][subkey] = subvalue
            elif isinstance(value, torch.Tensor):
                numpy_metadata[key] = value.numpy()
            else:
                numpy_metadata[key] = value
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'observations': numpy_observations,
                'activations': numpy_activations,
                'metadata': numpy_metadata,
            }, f)
    
    # other
    def to_dataset(self) -> 'ActivationDataset':
        """Convert collector to a dataset for analysis."""
        dataset = ActivationDataset()
        dataset.observations = self.observations
        dataset.activations = self.activations
        dataset.metadata = self.metadata
        return dataset
