"""
Numerical analysis functions for SAE causal interventions.

This module provides functions for computing causal matrices, feature importance,
and clustering features based on their causal patterns.
"""
import torch

import numpy as np

from typing import Dict, List, Tuple, Callable, Any
from tqdm import tqdm


def compute_causal_matrices(
    baseline_features: Dict[str, torch.Tensor],
    baseline_mu: np.ndarray,
    baseline_value: np.ndarray,
    feature_loader: Callable[[int], Dict[str, Any]],
    source_layer: str,
    layer_names: List[str],
    num_features: int,
    exclude_mu: bool = False,
    exclude_value: bool = False
) -> Dict[str, np.ndarray]:
    """
    Compute causal matrices showing the effect of ablating features.
    
    Args:
        baseline_features: Original features without ablation
        baseline_mu: Original mu output without ablation (shape: time, batch, dim)
        baseline_value: Original value output without ablation (shape: time, batch, 1)
        feature_loader: Function that loads a feature's ablation data given its index
        source_layer: Layer where features were ablated
        layer_names: Names of all layers in order
        num_features: Total number of features to process
        exclude_mu: Whether to exclude mu from analysis
        exclude_value: Whether to exclude value from analysis
        
    Returns:
        Dictionary mapping target names to causal matrices
    """
    # Find layer index to determine downstream layers
    source_idx = layer_names.index(source_layer)
    downstream_layers = layer_names[source_idx+1:]
    
    # Initialize causal matrices for all potential targets
    causal_matrices = {}
    for target_layer in downstream_layers:
        source_dim = baseline_features[source_layer].shape[1]
        target_dim = baseline_features[target_layer].shape[1]
        causal_matrices[target_layer] = np.zeros((source_dim, target_dim))
    
    # Reshape baseline mu and value to (time*batch, dim)
    if not exclude_mu:
        source_dim = baseline_features[source_layer].shape[1]
        if len(baseline_mu.shape) == 3:  # (time, batch, dim)
            time, batch, mu_dim = baseline_mu.shape
            baseline_mu_reshaped = baseline_mu.reshape(-1, mu_dim)  # (time*batch, dim)
        else:  # Already (time*batch, dim) or (batch, dim)
            baseline_mu_reshaped = baseline_mu
        
        causal_matrices["mu"] = np.zeros((source_dim, baseline_mu_reshaped.shape[1]))
    
    if not exclude_value:
        source_dim = baseline_features[source_layer].shape[1]
        if len(baseline_value.shape) == 3:  # (time, batch, 1)
            time, batch, _ = baseline_value.shape
            baseline_value_reshaped = baseline_value.reshape(-1, 1)  # (time*batch, 1)
        else:  # Already (time*batch, 1) or (batch, 1)
            baseline_value_reshaped = baseline_value
        
        causal_matrices["value"] = np.zeros((source_dim, 1))
    
    # Compute deltas for each ablated feature
    for feature_idx in tqdm(range(num_features), desc=f"Computing causal matrices for {source_layer}"):
        # Load this feature's ablation data on demand
        result = feature_loader(feature_idx)
        if result is None:
            print(f"Warning: Could not load data for feature {feature_idx}, skipping")
            continue
            
        ablated_features = result["features"]
        ablated_mu = result["mu"]
        ablated_value = result["value"]
        
        # Reshape ablated mu and value to match baseline shapes
        if not exclude_mu and "mu" in causal_matrices:
            if len(ablated_mu.shape) == 3:  # (time, batch, dim)
                time, batch, mu_dim = ablated_mu.shape
                ablated_mu_reshaped = ablated_mu.reshape(-1, mu_dim)  # (time*batch, dim)
            else:  # Already (time*batch, dim) or (batch, dim)
                ablated_mu_reshaped = ablated_mu
        
        if not exclude_value and "value" in causal_matrices:
            if len(ablated_value.shape) == 3:  # (time, batch, 1)
                time, batch, _ = ablated_value.shape
                ablated_value_reshaped = ablated_value.reshape(-1, 1)  # (time*batch, 1)
            else:  # Already (time*batch, 1) or (batch, 1)
                ablated_value_reshaped = ablated_value
        
        # Compute deltas for downstream layers
        for target_layer in downstream_layers:
            # Convert to numpy if needed
            if isinstance(baseline_features[target_layer], torch.Tensor):
                baseline_target = baseline_features[target_layer].cpu().numpy()
            else:
                baseline_target = baseline_features[target_layer]
                
            if isinstance(ablated_features[target_layer], torch.Tensor):
                ablated_target = ablated_features[target_layer].cpu().numpy()
            else:
                ablated_target = ablated_features[target_layer]
            
            # Compute mean delta across all samples
            delta = ablated_target - baseline_target
            causal_matrices[target_layer][int(feature_idx), :] = delta.mean(axis=0)
        
        # Compute deltas for mu and value if not excluded
        if not exclude_mu and "mu" in causal_matrices:
            delta_mu = np.clip(ablated_mu_reshaped, -1, 1) - np.clip(baseline_mu_reshaped, -1, 1)
            causal_matrices["mu"][int(feature_idx), :] = delta_mu.mean(axis=0)
        
        if not exclude_value and "value" in causal_matrices:
            delta_value = ablated_value_reshaped - baseline_value_reshaped
            causal_matrices["value"][int(feature_idx), :] = delta_value.mean(axis=0)
    
    return causal_matrices


def compute_feature_importance(causal_matrix: np.ndarray) -> np.ndarray:
    """
    Compute importance scores for features based on their causal effects.
    
    Args:
        causal_matrix: Matrix of causal effects [source_features x target_features]
        
    Returns:
        Array of importance scores for each source feature
    """
    # Use L2 norm across all target features as importance score
    return np.linalg.norm(causal_matrix, axis=1)


def get_top_features(importance_scores: np.ndarray, top_n: int = 20) -> np.ndarray:
    """
    Get indices of top features based on importance scores.
    
    Args:
        importance_scores: Array of importance scores for features
        top_n: Number of top features to return
        
    Returns:
        Indices of top features
    """
    return np.argsort(importance_scores)[::-1][:top_n]


# TODO: havent checked this
def cluster_features(
    causal_matrix: np.ndarray,
    n_clusters: int = 10
) -> Tuple[Dict[int, List[int]], np.ndarray]:
    """
    Cluster features based on their causal patterns.
    
    Args:
        causal_matrix: Matrix of causal effects [source_features x target_features]
        n_clusters: Number of clusters to use
        
    Returns:
        Tuple of (cluster_assignments, cluster_centers)
    """
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        raise ImportError("sklearn is required for clustering. Install with 'pip install scikit-learn'.")
    
    # Standardize the features for clustering
    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(causal_matrix)
    
    # Determine appropriate number of clusters
    if causal_matrix.shape[0] < n_clusters:
        n_clusters = max(2, causal_matrix.shape[0] // 2)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_matrix)
    
    # Group features by cluster
    clusters = {}
    for i, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(i)
    
    return clusters, kmeans.cluster_centers_


# TODO: havent checked this
def get_cluster_patterns(
    causal_matrix: np.ndarray,
    clusters: Dict[int, List[int]]
) -> Tuple[np.ndarray, List[str]]:
    """
    Get average causal patterns for each cluster.
    
    Args:
        causal_matrix: Matrix of causal effects [source_features x target_features]
        clusters: Dictionary mapping cluster IDs to feature indices
        
    Returns:
        Tuple of (cluster_patterns, cluster_labels)
    """
    cluster_patterns = []
    cluster_labels = []
    
    for cluster_id, feature_indices in sorted(clusters.items()):
        # Skip empty clusters
        if not feature_indices:
            continue
        
        # Compute mean pattern for this cluster
        cluster_pattern = causal_matrix[feature_indices].mean(axis=0)
        cluster_patterns.append(cluster_pattern)
        cluster_labels.append(f"Cluster {cluster_id} ({len(feature_indices)} features)")
    
    return np.vstack(cluster_patterns), cluster_labels
