"""
Responsibilities:
- Calculate correlations between neuron activations and various features in pre-recorded data
    - Joint positions
    - Joint velocities
    - X, Y, Z components of distance vector
    - Distance magnitude
    - Drawer position/velocity
- Identify neurons with strong responses to any of the above features
"""
import numpy as np

from typing import List, Tuple

from ..activation_dataset import ActivationDataset


class CorrelationAnalyzer:
    def __init__(self, dataset: ActivationDataset):
        self.dataset = dataset

    def compute_correlation(self, 
                           layer_name: str, 
                           feature_key: str) -> np.ndarray:
        """
        Compute correlation between all neurons in a layer and a metadata feature.
        
        Args:
            layer_name: Name of the layer to analyze
            feature_key: Key of the metadata feature to correlate with
            
        Returns:
            Array of correlation values for each neuron in the layer
        """
        # Get activations matrix [time*batch, neurons]
        activations = self.dataset.get_activation_matrix(layer_name)
        
        # Get flattened feature values [time*batch]
        feature_values = self.dataset.get_flattened_feature(feature_key)
        
        # Check if dimensions match
        if len(activations) != len(feature_values):
            raise ValueError(f"Activation dimensions {len(activations)} don't match "
                            f"feature dimensions {len(feature_values)}")
        
        # Calculate correlation for each neuron
        correlations = np.zeros(activations.shape[1])
        for i in range(activations.shape[1]):
            neuron_activations = activations[:, i]
            # Handle NaN values
            mask = ~np.isnan(neuron_activations) & ~np.isnan(feature_values)
            if np.sum(mask) > 1:
                corr = np.corrcoef(neuron_activations[mask], feature_values[mask])[0, 1]
                correlations[i] = corr if not np.isnan(corr) else 0
        
        return correlations

    def find_correlated_neurons(self, 
                               layer_name: str, 
                               feature_key: str, 
                               threshold: float = 0.7) -> List[Tuple[int, float]]:
        """
        Find neurons in a layer whose activations correlate with a metadata feature.
        
        Args:
            layer_name: Name of the layer to analyze
            feature_key: Key of the metadata feature to correlate with
            threshold: Correlation threshold
            
        Returns:
            List of (neuron_index, correlation) tuples for neurons above threshold
        """
        # Get correlations for all neurons
        correlations = self.compute_correlation(layer_name, feature_key)
        
        # Find neurons with correlation above threshold
        high_correlations = []
        for i, corr in enumerate(correlations):
            if abs(corr) >= threshold:
                high_correlations.append((i, corr))
        
        # Sort by absolute correlation value (highest first)
        return sorted(high_correlations, key=lambda x: abs(x[1]), reverse=True)
    
    def get_layer_correlation_matrix(self, 
                                    layer_name: str, 
                                    feature_keys: List[str], 
                                    sort_neurons: bool = True) -> np.ndarray:
        """
        Generate a correlation matrix for neurons in a layer and a list of features.
        """
        # Build correlation matrix by concatenating results for each feature
        corr_matrix = np.column_stack([
            self.compute_correlation(layer_name, feature) 
            for feature in feature_keys
        ])
            
        # Sort neurons by similarity (optional)
        if sort_neurons:
            # Use hierarchical clustering to reorder neurons
            from scipy.cluster.hierarchy import linkage, dendrogram
            Z = linkage(corr_matrix, 'ward')
            reordered_idx = dendrogram(Z, no_plot=True)['leaves']
            corr_matrix = corr_matrix[reordered_idx, :]
        
        return corr_matrix
    # TODO: notebook used identify_all_phase_neurons in this class but we refactored it out, fix notebook
