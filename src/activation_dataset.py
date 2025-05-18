"""
Activation dataset module for managing neural network activations.

This module is responsible for processing pre-recorded observation trajectories,
and providing a structured way to store and access them along with metadata.

Responsibilities:
- Store activations with their corresponding metadata from pre-recorded observations
- Organize data to preserve the relationship between observations and activations
- Support filtering of data by phase, or other criteria
- Maintain tensor structure throughout for efficient operations
"""
import pickle
import numpy as np

from typing import Dict, Tuple, List, Optional


class ActivationDataset:
    """
    Dataset for storing neural network activations and associated metadata as NumPy arrays.
    Retrieval methods for data analysis
    """
    def __init__(self):
        self.observations: Optional[np.ndarray] = None  # Will be a numpy array of shape [time, batch, obs_dim]
        self.activations: Dict[str, np.ndarray] = {}     # Dict of {layer_name: numpy array of shape [time, batch, ...]}
        # Dict of {key: numpy array or dict of numpy arrays}. Usually contains features, phase_masks, etc.
        self.metadata: Dict[str, np.ndarray | Dict[str, np.ndarray]] = {}
    
    # Retrieval methods / activations / by phase
    def filter_by_phase(self, phase_name: str) -> 'ActivationDataset':
        if 'phase_masks' not in self.metadata or phase_name not in self.metadata['phase_masks']:
            raise ValueError(f"Phase {phase_name} not found in dataset")
        
        phase_mask = self.metadata['phase_masks'][phase_name].astype(bool)
        
        # Create filtered dataset
        filtered = ActivationDataset()
        
        # Filter observations
        filtered.observations = self.observations[phase_mask]
        
        # Filter activations
        filtered.activations = {}
        for layer, activation in self.activations.items():
            filtered.activations[layer] = activation[phase_mask]
        
        # Filter metadata
        filtered.metadata = {}
        for key, value in self.metadata.items():
            if key == 'phase_masks':
                continue
                
            filtered.metadata[key] = {}
            for subkey, subvalue in value.items():
                # Apply mask directly since we know the structure is [time, batch, some_dim]
                filtered.metadata[key][subkey] = subvalue[phase_mask]
            
        return filtered
    
    def get_flattened_observations(self) -> np.ndarray:
        return self.observations.reshape(-1, self.observations.shape[-1])

    # Retrieval methods / activations / by layer
    def get_activation_tensor(self, layer_name: str) -> np.ndarray:
        """
        Get activation tensor for a layer.
        
        Returns:
            np.ndarray: Shape (time, batch, neurons) if unfiltered
                         or (flattened_index, neurons) if filtered
        """
        if layer_name not in self.activations:
            raise ValueError(f"Layer {layer_name} not found in activations")
        return self.activations[layer_name]
    
    def get_activation_matrix(self, layer_name: str) -> np.ndarray:
        """
        Get activations reshaped to [time*batch, neurons] if unfiltered
        or return as is if already in [flattened_index, neurons] format.
        """
        activation = self.get_activation_tensor(layer_name)
        
        # Check if already flattened (2D array)
        if activation.ndim == 2:
            return activation
        
        # Otherwise handle the original [time, batch, ...] format
        # Flatten the time and batch dimensions
        time_steps, batch_size = activation.shape[0], activation.shape[1]
        
        # Reshape to [time*batch, neurons]
        if activation.ndim > 2:
            # Flatten all dimensions after the first two
            rest_dims = activation.shape[2:]
            total_features = np.prod(rest_dims).item() if hasattr(np.prod(rest_dims), 'item') else np.prod(rest_dims)
            flattened = activation.reshape(time_steps, batch_size, total_features)
        else:
            flattened = activation
        
        # Combine time and batch dimensions
        return flattened.reshape(time_steps * batch_size, -1)
    
    # Retrieval methods / activations / all
    def get_combined_layer_activations(
            self, 
            layer_names: Optional[List[str]] = None
        ) -> Tuple[np.ndarray, Dict[str, Tuple[int, int]]]:
        """
        Combine activations from all layers into a single matrix, handling different neuron counts.
        
        Returns:
            Tuple of (combined_matrix, layer_indices) where:
            - combined_matrix is a numpy array of shape [time*batch, total_neurons]
            - layer_indices is a dict mapping layer names to (start_idx, end_idx) tuples
        """
        matrices = []
        layer_indices = {}
        
        current_index = 0
        if layer_names is None:
            print(f"Warning: layer_name and layer_names missing, using all layers")
            layer_names = list(self.activations.keys())
        for layer_name in layer_names:
            activation_matrix = self.get_activation_matrix(layer_name)
            
            # Record indices for this layer
            start_idx = current_index
            end_idx = start_idx + activation_matrix.shape[1]
            layer_indices[layer_name] = (start_idx, end_idx)
            current_index = end_idx
            
            matrices.append(activation_matrix)
        if not matrices:
            return np.array([]), {}
        
        if len(set(matrix.shape[0] for matrix in matrices)) > 1:
            raise ValueError("Not all layers have the same number of samples")
        
        combined_matrix = np.hstack(matrices)
        
        return combined_matrix, layer_indices

    # Retrieval methods / features
    def get_flattened_feature(self, feature_key: str) -> np.ndarray:
        """
        Get a flattened (1D) version of a feature by combining time and batch dimensions.
        Works with both original and already-flattened data formats.
        
        Args:
            feature_key: Key of the feature to get
            
        Returns:
            1D array containing the feature values with time and batch dimensions flattened
        """
        if feature_key not in self.metadata['features']:
            raise ValueError(f"Feature '{feature_key}' not found in metadata")
        
        feature = self.metadata['features'][feature_key]
        
        # If already 1D, return as is
        if feature.ndim == 1:
            return feature
        
        # Check if this is a vector feature with components (like distance_vec)
        if feature.ndim > 2:
            raise ValueError(f"Feature '{feature_key}' has more than 2 dimensions. "
                            f"Please use a scalar feature or specify a component.")
        
        # Flatten time and batch dimensions for original format
        # also handle case If flattened but with dimension (already [flattened_index, dim])
        # if feature.ndim == 2 and self.observations.ndim == 2:
        return feature.flatten()
    
    # Retrieval methods / activations by phase and layer
    def get_phase_activations(self, layer_name: str) -> Tuple[Dict[str, np.ndarray], float]:
        """ Get activations for each phase, and the global minimum activation value """
        global_min = np.inf
        phase_activations = {}
        for phase_name in self.metadata['phase_masks'].keys():
            phase_dataset = self.filter_by_phase(phase_name)
            phase_activations[phase_name] = phase_dataset.get_activation_matrix(layer_name)
            global_min = min(global_min, np.min(phase_activations[phase_name]))

        return phase_activations, global_min

    def get_combined_phase_data(self, layer_name: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Create combined dataset of activations with phase labels using get_phase_activations.
        
        Args:
            dataset: ActivationDataset containing activations and phases
            layer_name: Name of the layer to analyze
            
        Returns:
            Tuple of (all_activations, phase_labels, phase_names)
        """
        # Get activations for each phase using the dataset's method
        phase_activations, _ = self.get_phase_activations(layer_name)
        phase_names = list(phase_activations.keys())
        
        # Combine activations and create labels
        all_activations = []
        phase_labels = []
        
        for i, phase_name in enumerate(phase_names):
            activations = phase_activations[phase_name]
            all_activations.append(activations)
            phase_labels.extend([i] * len(activations))
        
        # Convert to numpy arrays
        if all_activations:
            all_activations = np.vstack(all_activations)
            phase_labels = np.array(phase_labels)
        else:
            all_activations = np.array([])
            phase_labels = np.array([])
        
        return all_activations, phase_labels, phase_names
        
    def get_combined_phase_data_multilayer(self, layer_names: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Create combined dataset of activations with phase labels using multiple layers.
        
        Args:
            layer_names: Names of layers to analyze
            
        Returns:
            Tuple of (all_activations, phase_labels, phase_names)
        """
        phase_names = list(self.metadata['phase_masks'].keys())
        
        # Combine activations and create labels
        all_activations = []
        phase_labels = []
        
        for i, phase_name in enumerate(phase_names):
            # Filter dataset for this phase
            phase_dataset = self.filter_by_phase(phase_name)
            # Get combined activations for this phase across specified layers
            phase_combined_activations, _ = phase_dataset.get_combined_layer_activations(layer_names)
            
            all_activations.append(phase_combined_activations)
            phase_labels.extend([i] * len(phase_combined_activations))
        
        # Convert to numpy arrays
        if all_activations:
            all_activations = np.vstack(all_activations)
            phase_labels = np.array(phase_labels)
        else:
            all_activations = np.array([])
            phase_labels = np.array([])
        
        return all_activations, phase_labels, phase_names

    # file methods
    def load(self, filepath: str) -> None:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.observations = data['observations']
        self.activations = data['activations']
        self.metadata = data['metadata']
