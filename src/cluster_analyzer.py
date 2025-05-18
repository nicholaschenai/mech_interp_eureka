"""
Cluster analysis of neuron activations, to see if we can identify distinct phases of behavior.

This module applies unsupervised clustering algorithms to neural activations to
identify natural groupings that may correspond to different behavioral phases.

Responsibilities:
- Apply various clustering algorithms to neuron activations
- Compare clustering results with known phase labels
- Evaluate clustering quality with metrics
- Find optimal clustering parameters
"""
import copy

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, confusion_matrix, adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from .activation_dataset import ActivationDataset
from .base_analyzer import BaseAnalyzer


class ClusterAnalyzer(BaseAnalyzer):
    def __init__(self):
        super().__init__()
    
    def _preprocess_activations(self, 
                               activations: np.ndarray, 
                               standardize: bool = True) -> np.ndarray:
        """
        Args:
            activations: Activation matrix with shape (n_samples, n_features)
            standardize: Whether to standardize features
        """
        activations = self.handle_nan_values(activations)
        
        # Standardize features if requested
        if standardize:
            scaler = StandardScaler()
            activations = scaler.fit_transform(activations)
        
        return activations
    
    def compare_with_phases(self,
                          cluster_labels: np.ndarray,
                          phase_labels: np.ndarray,
                          phase_names: List[str]) -> Dict[str, Any]:
        """
        Compare clustering results with known phase labels.
        """
        cm = confusion_matrix(phase_labels, cluster_labels)
        
        # Normalize confusion matrix by row (phase)
        with np.errstate(divide='ignore', invalid='ignore'):
            cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
            cm_norm = np.nan_to_num(cm_norm)
        
        ari = adjusted_rand_score(phase_labels, cluster_labels)
        
        # Return results
        return {
            'confusion_matrix': cm,
            'normalized_confusion_matrix': cm_norm,
            'adjusted_rand_index': ari,
            'phase_names': phase_names
        }
    
    def _create_cluster_model(self, 
                           algorithm: str, 
                           n_clusters: Optional[int] = None, 
                           **kwargs) -> Any:
        """
        Args:
            algorithm: Clustering algorithm ('kmeans', 'hierarchical', or 'dbscan')
            n_clusters: Number of clusters (not used for DBSCAN)
            **kwargs: Additional algorithm-specific parameters
        """
        if algorithm.lower() == 'kmeans':
            return KMeans(n_clusters=n_clusters, random_state=42, **kwargs)
        elif algorithm.lower() == 'hierarchical':
            return AgglomerativeClustering(n_clusters=n_clusters, **kwargs)
        elif algorithm.lower() == 'dbscan':
            return DBSCAN(**kwargs)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    def _perform_clustering(self, 
                           activations: np.ndarray, 
                           algorithm: str, 
                           n_clusters: Optional[int] = None, 
                           **kwargs) -> Dict[str, Any]:
        """
        Args:
            algorithm: Clustering algorithm to use
            **kwargs: Additional algorithm parameters
        """
        # Create and fit model
        cluster_model = self._create_cluster_model(algorithm, n_clusters, **kwargs)
        cluster_labels = cluster_model.fit_predict(activations)
        
        # Calculate silhouette score
        try:
            sil_score = silhouette_score(activations, cluster_labels)
        except Exception as e:
            print(f"Could not calculate silhouette score: {e}")
            sil_score = 0.0
        
        return {
            'model': cluster_model,
            'labels': cluster_labels,
            'silhouette_score': sil_score
        }
            
    def analyze_phases(self,
                      dataset: ActivationDataset,
                      layer_name: Optional[str] = None,
                      algorithm: str = 'kmeans',
                      n_clusters: Optional[int] = None,
                      layer_names: Optional[List[str]] = None,
                      **kwargs) -> Dict[str, Any]:
        """
        Analyze neural activations to identify phase-related clusters.
        
        Args:
            dataset: ActivationDataset containing activations and phases
            layer_name: Name of a single layer to analyze (for backward compatibility)
            algorithm: Clustering algorithm to use ('kmeans', 'hierarchical', 'dbscan')
            n_clusters: Number of clusters to use (if None, uses number of phases)
            layer_names: List of layer names to combine and analyze
            **kwargs: Additional algorithm-specific parameters
            
        Returns:
            Dictionary with analysis results
        """
        # Get combined activations and phase labels
        if layer_name is not None:
            # Single layer (backward compatibility)
            all_activations, phase_labels, phase_names = dataset.get_combined_phase_data(layer_name)
        else:
            # Multiple layers or all layers if layer_names is None
            all_activations, phase_labels, phase_names = dataset.get_combined_phase_data_multilayer(layer_names)
        
        if len(all_activations) == 0:
            return {'error': 'No activations found for any phase'}
        
        # Set number of clusters to match number of phases if not specified
        if n_clusters is None:
            n_clusters = len(phase_names)
        
        # Preprocess activations
        all_activations = self._preprocess_activations(all_activations)
        
        # Perform clustering
        clustering_results = self._perform_clustering(all_activations, algorithm, n_clusters, **kwargs)
        
        # Compare clusters with phases
        comparison = self.compare_with_phases(
            clustering_results['labels'], phase_labels, phase_names
        )
        
        # Return results
        return {
            'model': clustering_results['model'],
            'cluster_labels': clustering_results['labels'],
            'phase_labels': phase_labels,
            'phase_names': phase_names,
            'silhouette_score': clustering_results['silhouette_score'],
            'comparison': comparison,
            'n_samples_per_phase': {phase: np.sum(phase_labels == i) 
                                  for i, phase in enumerate(phase_names)},
            'activations': all_activations
        }
    
    def find_optimal_clusters(self,
                             activations: np.ndarray,
                             max_clusters: int = 10,
                             algorithm: str = 'kmeans',
                             **kwargs) -> Dict[str, Any]:
        """
        Find optimal number of clusters using silhouette scores.
        
        Args:
            dataset: ActivationDataset containing activations
            max_clusters: Maximum number of clusters to try
            algorithm: Clustering algorithm to use ('kmeans' or 'hierarchical')
            **kwargs: Additional algorithm-specific parameters
        """
        if algorithm.lower() not in ['kmeans', 'hierarchical']:
            raise ValueError(f"Algorithm {algorithm} not supported for finding optimal clusters. " 
                           f"Use 'kmeans' or 'hierarchical' instead.")
        
        # Try different numbers of clusters
        results = {}
        for n in range(2, max_clusters + 1):
            clustering_results = self._perform_clustering(activations, algorithm, n, **kwargs)
            
            results[n] = copy.deepcopy(clustering_results)
        
        # Find optimal number of clusters
        if results:
            optimal_n = max(results.keys(), key=lambda n: results[n]['silhouette_score'])
        else:
            optimal_n = 2  # Default if no results
            
        return {
            'all_results': results,
            'optimal_n_clusters': optimal_n,
            'optimal_score': results[optimal_n]['silhouette_score'] if results else 0.0,
            'optimal_labels': results[optimal_n]['labels'] if results else np.array([])
        }
