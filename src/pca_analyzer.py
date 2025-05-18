"""
PCA Analysis to identify major patterns in neuron activations and correlate them with input features.

Responsibilities:
- Run PCA on neuron activations for any layer
- Project activations onto principal components
- Calculate correlations between principal components and features
- Perform regression analysis to identify feature importance
"""
import os
import numpy as np

from typing import Dict, List, Any, Tuple, Optional

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from .activation_dataset import ActivationDataset
from .base_analyzer import BaseAnalyzer


class PCAAnalyzer(BaseAnalyzer):
    def __init__(self):
        super().__init__()

    # helpers    
    def get_feature_matrix(self, 
                         dataset: ActivationDataset, 
                         features: List[str],
                         projected_activations: np.ndarray) -> np.ndarray:
        """
        Args:
            features: List of feature keys to extract
            projected_activations: array to validate dimensions against
            
        Returns:
            Feature matrix with shape (n_samples, n_features)
        """
        # TODO: maybe this should be a dataset method?
        feature_matrix = np.column_stack([
            dataset.get_flattened_feature(feature) 
            for feature in features
        ])

        feature_matrix = self.handle_nan_values(feature_matrix)
        
        if len(feature_matrix) != len(projected_activations):
            raise ValueError(f"Feature matrix dimensions {feature_matrix.shape} don't match "
                            f"projected activations dimensions {projected_activations.shape}")
        
        return feature_matrix
    
    def get_activation_matrix(
        self,
        dataset: ActivationDataset, 
        layer_name: Optional[str] = None,
        layer_names: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, Dict[str, Tuple[int, int]]]:
        # TODO: maybe this should be a dataset method?
        if layer_name is not None:
            # Single layer (backward compatibility)
            activation_matrix = dataset.get_activation_matrix(layer_name)
            layer_indices = {layer_name: (0, activation_matrix.shape[1])}
        else:
            activation_matrix, layer_indices = dataset.get_combined_layer_activations(layer_names)

        activation_matrix = self.handle_nan_values(activation_matrix)

        return activation_matrix, layer_indices
    
    # main methods
    def run_pca(self, 
               dataset: ActivationDataset, 
               layer_name: Optional[str] = None,
               n_components: Optional[int] = None,
               layer_names: Optional[List[str]] = None, 
               **kwargs) -> Dict[str, Any]:
        """
        Run single PCA on activation data from one layer or multiple layers combined.
        
        Args:
            dataset: ActivationDataset containing activations
            layer_name: Name of single layer to analyze (backward compatibility)
            n_components: Number of components to keep (None means all components)
            layer_names: List of layer names to combine and analyze
            **kwargs: Additional arguments for PCA
            
        Returns:
            Dictionary with PCA results and layer information
            
        Note:
            If both layer_name and layer_names are provided, layer_name takes precedence.
            If neither is provided, all layers in the dataset will be used.
        """
        activation_matrix, layer_indices = self.get_activation_matrix(dataset, layer_name, layer_names)
        
        pca = PCA(n_components=n_components, **kwargs)
        pca.fit(activation_matrix)
        
        results = {
            'pca_model': pca,
            'components': pca.components_,  # Shape: (n_components, n_features)
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'explained_variance': pca.explained_variance_,
            'mean': pca.mean_,
            'layer_indices': layer_indices, # Add layer information for multi-layer analysis
        }
        
        return results
    
    def project_activations(
        self, 
        dataset: ActivationDataset, 
        pca_model: PCA,
        layer_name: Optional[str] = None,
        layer_names: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Project activations onto principal components.
        
        Args:
            dataset: ActivationDataset containing the activations
            layer_name: Name of single layer to project (backward compatibility)
            layer_names: List of layer names to combine and project
            
        Returns:
            Array of projected activations with shape (n_samples, n_components)
        """
        # TODO: repeated computation of activation matrix, will handle later
        activation_matrix, _ = self.get_activation_matrix(dataset, layer_name, layer_names)
        
        projected = pca_model.transform(activation_matrix)
        
        return projected
    
    def regress_features_against_components(self,
                                          dataset: ActivationDataset,
                                          projected_activations: np.ndarray,
                                          features: List[str]) -> Dict[str, Any]:
        """
        Args:
            dataset: ActivationDataset containing the features
            projected_activations: Activations projected onto principal components
            features: List of feature keys to analyze
        """
        feature_matrix = self.get_feature_matrix(dataset, features, projected_activations)
        
        # Initialize regression models for each component
        n_components = projected_activations.shape[1]
        coefficients = np.zeros((len(features), n_components))
        intercepts = np.zeros(n_components)
        r2_scores = np.zeros(n_components)
        
        # Fit regression model for each principal component
        for i in range(n_components):
            component_values = projected_activations[:, i]
            
            # Skip if component values are constant
            if np.std(component_values) < 1e-10:
                print(f"Warning: Component {i} has near-zero variance. Skipping regression.")
                continue
                
            model = LinearRegression()
            model.fit(feature_matrix, component_values)
            
            coefficients[:, i] = model.coef_
            intercepts[i] = model.intercept_
            r2_scores[i] = model.score(feature_matrix, component_values)
        
        return {
            'coefficients': coefficients,
            'intercepts': intercepts,
            'r2_scores': r2_scores,
            'feature_names': features
        }
    
    def compute_feature_component_correlations(self,
                                             dataset: ActivationDataset,
                                             projected_activations: np.ndarray,
                                             features: List[str]) -> np.ndarray:
        """
        Args:
            dataset: ActivationDataset containing the features
            projected_activations: Activations projected onto principal components
            features: List of feature keys to analyze
        """
        feature_matrix = self.get_feature_matrix(dataset, features, projected_activations)
        
        n_features = len(features)
        n_components = projected_activations.shape[1]
        
        # Initialize correlation matrix
        correlation_matrix = np.zeros((n_features, n_components))
        
        # Compute correlation for each feature and component
        for i in range(n_features):
            feature_values = feature_matrix[:, i]
            
            for j in range(n_components):
                component_values = projected_activations[:, j]
                
                # Handle constant values
                if np.std(feature_values) < 1e-10 or np.std(component_values) < 1e-10:
                    correlation_matrix[i, j] = 0
                    continue
                
                # Compute correlation coefficient
                valid_indices = ~np.isnan(feature_values) & ~np.isnan(component_values)
                if np.sum(valid_indices) > 1:
                    corr = np.corrcoef(feature_values[valid_indices], component_values[valid_indices])[0, 1]
                    correlation_matrix[i, j] = corr if not np.isnan(corr) else 0
        
        return correlation_matrix

    # save methods
    def save_pca_results(self, results: Dict[str, Dict], output_path: str) -> str:
        """
        Save all PCA components and metadata in a single file.
        
        Args:
            results: Dict mapping layer_name to PCA results from run_pca_for_multiple_layers
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        combined_data = {}
        
        keys_to_save = ['components', 'explained_variance', 'explained_variance_ratio', 'mean']
        for layer_name, layer_results in results.items():
            for key in keys_to_save:
                combined_data[f"{layer_name}_{key}"] = layer_results[key]
        
        np.savez(output_path, **combined_data)
        
    def run_pca_for_multiple_layers(self, 
                                dataset: ActivationDataset,
                                layer_components_map: Dict[str, int],
                                output_path: str = None) -> Dict[str, Dict]:
        """
        Run PCA independently for multiple layers (one PCA per layer) and save results to a single file.
        
        Args:
            dataset: ActivationDataset containing the activations
            layer_components_map: Dict mapping layer_name to number of components
            
        Returns:
            Dict mapping layer_name to PCA results
        """
        results = {}
        
        for layer_name, n_comp in layer_components_map.items():
            layer_results = self.run_pca(dataset, layer_name, n_comp)
            results[layer_name] = layer_results
        
        if output_path and results:
            self.save_pca_results(results, output_path)
            
        return results
