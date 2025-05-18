"""
Utility functions for SAE causal intervention analysis and visualization.

This module provides functions for visualizing causal influence matrices,
computing feature importance, and clustering features based on causal effects.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any
from matplotlib.figure import Figure
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def compute_feature_correlation(
    original_features: torch.Tensor, 
    ablated_features: torch.Tensor
) -> np.ndarray:
    """
    Compute correlation between original and ablated feature representations.
    
    Args:
        original_features: Original encoded features (batch_size, feature_dim)
        ablated_features: Features after intervention (batch_size, feature_dim)
        
    Returns:
        Correlation matrix (feature_dim, feature_dim)
    """
    # Implementation will be added later
    pass


def compute_causal_matrix(
    source_model: Any,
    target_model: Any,
    source_layer: str,
    target_layer: str,
    data_samples: torch.Tensor,
    num_source_features: int,
) -> np.ndarray:
    """
    Compute the causal influence matrix between source and target layer features.
    
    Args:
        source_model: Source layer's SAE model
        target_model: Target layer's SAE model
        source_layer: Name of source layer
        target_layer: Name of target layer
        data_samples: Input data samples to use
        num_source_features: Number of source features to ablate
        
    Returns:
        Causal influence matrix (num_source_features, num_target_features)
    """
    # Implementation will be added later
    pass


def compute_feature_importance(causal_matrix: np.ndarray) -> np.ndarray:
    """
    Compute importance scores for source features based on causal effects.
    
    Args:
        causal_matrix: Matrix of shape (source_features × target_features)
        
    Returns:
        Array of importance scores for each source feature
    """
    # Compute absolute magnitude of effects
    abs_effects = np.abs(causal_matrix)
    
    # Sum across target features to get total effect magnitude
    importance_scores = np.sum(abs_effects, axis=1)
    
    # Normalize to [0, 1] range
    importance_scores = importance_scores / np.max(importance_scores)
    
    return importance_scores


def cluster_similar_features(
    causal_matrix: np.ndarray,
    feature_labels: List[str],
    n_clusters: Optional[int] = None,
    max_clusters: int = 10
) -> Tuple[Dict[int, List[str]], np.ndarray]:
    """
    Cluster features based on similarity of their causal effects.
    
    Args:
        causal_matrix: Matrix of shape (source_features × target_features)
        feature_labels: Labels for features
        n_clusters: Number of clusters (or None to determine automatically)
        max_clusters: Maximum number of clusters to consider if n_clusters is None
        
    Returns:
        Tuple of (clustered features, cluster centers)
    """
    # Determine optimal number of clusters if not specified
    if n_clusters is None:
        best_score = -1
        best_n = 2
        
        for n in range(2, min(max_clusters + 1, causal_matrix.shape[0])):
            kmeans = KMeans(n_clusters=n, random_state=42, n_init='auto')
            labels = kmeans.fit_predict(causal_matrix)
            
            # Compute silhouette score if there are enough samples
            if len(set(labels)) > 1:
                score = silhouette_score(causal_matrix, labels)
                if score > best_score:
                    best_score = score
                    best_n = n
        
        n_clusters = best_n
        print(f"Automatically selected {n_clusters} clusters (silhouette score: {best_score:.3f})")
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(causal_matrix)
    
    # Group features by cluster
    clustered_features = {}
    for i, label in enumerate(labels):
        if label not in clustered_features:
            clustered_features[label] = []
        clustered_features[label].append(feature_labels[i])
    
    return clustered_features, kmeans.cluster_centers_


def visualize_causal_matrix(
    causal_matrix: np.ndarray,
    source_features: List[str],
    target_features: List[str],
    title: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = 'coolwarm'
) -> plt.Figure:
    """
    Visualize a causal influence matrix as a heatmap.
    
    Args:
        causal_matrix: Matrix of shape (source_features × target_features)
        source_features: Labels for source features
        target_features: Labels for target features
        title: Title for the plot
        vmin/vmax: Min/max values for colormap normalization
        cmap: Colormap to use
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # If vmin/vmax not provided, make them symmetric around 0
    if vmin is None and vmax is None:
        abs_max = np.max(np.abs(causal_matrix))
        vmin, vmax = -abs_max, abs_max
    
    # Plot heatmap
    im = ax.imshow(causal_matrix, cmap=cmap, vmin=vmin, vmax=vmax)
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Effect Magnitude')
    
    # Set title
    if title:
        ax.set_title(title, fontsize=14)
    
    # Set axis labels
    ax.set_xlabel('Target Features')
    ax.set_ylabel('Source Features')
    
    # When too many features, don't show all labels
    max_labels = 30
    
    # Set x-ticks for target features
    if len(target_features) <= max_labels:
        ax.set_xticks(range(len(target_features)))
        ax.set_xticklabels(target_features, rotation=90)
    else:
        # Show only a subset of labels
        ax.set_xticks(np.linspace(0, len(target_features)-1, max_labels, dtype=int))
        ax.set_xticklabels(
            [target_features[i] for i in np.linspace(0, len(target_features)-1, max_labels, dtype=int)],
            rotation=90
        )
    
    # Set y-ticks for source features
    if len(source_features) <= max_labels:
        ax.set_yticks(range(len(source_features)))
        ax.set_yticklabels(source_features)
    else:
        # Show only a subset of labels
        ax.set_yticks(np.linspace(0, len(source_features)-1, max_labels, dtype=int))
        ax.set_yticklabels(
            [source_features[i] for i in np.linspace(0, len(source_features)-1, max_labels, dtype=int)]
        )
    
    # Add grid
    ax.grid(False)
    
    plt.tight_layout()
    
    return fig


def plot_feature_importance(
    importance_scores: np.ndarray,
    feature_labels: List[str],
    title: Optional[str] = None,
    top_n: Optional[int] = None
) -> plt.Figure:
    """
    Plot feature importance scores.
    
    Args:
        importance_scores: Array of importance scores
        feature_labels: Labels for features
        title: Title for the plot
        top_n: Only show top N features by importance
        
    Returns:
        Matplotlib figure
    """
    # Sort features by importance
    sorted_indices = np.argsort(importance_scores)[::-1]
    sorted_scores = importance_scores[sorted_indices]
    sorted_labels = [feature_labels[i] for i in sorted_indices]
    
    # Limit to top N if specified
    if top_n is not None and top_n < len(sorted_scores):
        sorted_scores = sorted_scores[:top_n]
        sorted_labels = sorted_labels[:top_n]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot horizontal bars
    bars = ax.barh(range(len(sorted_scores)), sorted_scores, height=0.7)
    
    # Add feature labels
    ax.set_yticks(range(len(sorted_scores)))
    ax.set_yticklabels(sorted_labels)
    
    # Set axis labels and title
    ax.set_xlabel('Importance Score')
    ax.set_ylabel('Feature')
    if title:
        ax.set_title(title, fontsize=14)
    
    # Add grid lines
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    return fig


def compare_activation_distributions(
    original_activations: torch.Tensor,
    modified_activations: torch.Tensor,
    feature_idx: int,
    figsize: Tuple[int, int] = (10, 6),
    title: str = 'Activation Distribution Comparison'
) -> Figure:
    """
    Compare distributions of original and modified activations.
    
    Args:
        original_activations: Original activations
        modified_activations: Activations after intervention
        feature_idx: Index of feature that was ablated
        figsize: Figure size (width, height)
        title: Plot title
        
    Returns:
        Matplotlib figure object
    """
    # Implementation will be added later
    pass 
