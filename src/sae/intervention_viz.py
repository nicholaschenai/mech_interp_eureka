"""
Visualization functions for SAE causal interventions.

This module provides visualization functions for causal matrices, feature importance,
and cluster patterns. Uses only matplotlib for plotting.
"""

import matplotlib.pyplot as plt
import numpy as np

from typing import Tuple, List, Optional, Dict


def visualize_causal_matrix(
    causal_matrix: np.ndarray,
    title: str,
    max_features: int = 50,
    top_indices: Optional[np.ndarray] = None,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10),
    x_axis_label: Optional[str] = None,
    x_tick_labels: Optional[List[str]] = None
) -> plt.Figure:
    """
    Visualize a causal matrix as a heatmap.
    
    Args:
        causal_matrix: Matrix of causal effects [source_features x target_features]
        title: Plot title
        max_features: Maximum number of features to display (for readability)
        top_indices: Pre-computed indices of top features (if None, show all or first max_features)
        output_path: Path to save the figure
        figsize: Figure size
        x_axis_label: Custom label for x-axis (overrides automatic detection)
        x_tick_labels: Custom labels for x-axis ticks (e.g., joint names for actions)
        
    Returns:
        Matplotlib figure
    """
    # Limit to max_features if needed
    if causal_matrix.shape[0] > max_features:
        if top_indices is not None:
            # Use provided top indices
            top_indices = top_indices[:max_features]
            causal_matrix = causal_matrix[top_indices, :]
            index_label = f"Top {max_features} Features"
        else:
            # Just use first max_features if no top indices provided
            causal_matrix = causal_matrix[:max_features, :]
            index_label = f"First {max_features} Features"
    else:
        index_label = "Features"
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Determine appropriate color scaling
    vmax = np.percentile(np.abs(causal_matrix), 99)
    vmin = -vmax
    
    # Create heatmap using matplotlib's imshow
    im = ax.imshow(causal_matrix, cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='auto')
    
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Effect Size')
    
    ax.set_ylabel(index_label)
    
    if x_axis_label is not None:
        ax.set_xlabel(x_axis_label)
        if x_axis_label == "Action Dimensions":
            ax.set_xticks(np.arange(9))
            if x_tick_labels:
                ax.set_xticklabels(x_tick_labels, rotation=45, ha='right')
            else:
                ax.set_xticklabels([f"A{i}" for i in range(9)])

    
    # Add y-tick labels with feature indices if using top indices
    if top_indices is not None:
        ax.set_yticks(np.arange(len(top_indices)))
        ax.set_yticklabels([f"F{int(idx)}" for idx in top_indices])
    
    ax.set_title(title)
    
    # Improve layout
    plt.tight_layout()
    
    # Save figure if output path is provided
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    
    return fig


def plot_feature_importance(
    importance_scores: np.ndarray,
    title: str,
    top_n: int = 20,
    top_indices: Optional[np.ndarray] = None,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Plot feature importance rankings.
    
    Args:
        importance_scores: Array of importance scores
        title: Plot title
        top_n: Number of top features to display
        top_indices: Pre-computed indices of top features (if None, will be calculated)
        output_path: Path to save the figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Get indices of top features
    if top_indices is None:
        # Sort indices by importance score
        top_indices = np.argsort(importance_scores)[::-1][:top_n]
    else:
        # Use provided indices but limit to top_n
        top_indices = top_indices[:top_n]
        
    top_scores = importance_scores[top_indices]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot horizontal bar chart
    y_pos = np.arange(len(top_indices))
    bars = ax.barh(y_pos, top_scores, color="royalblue", alpha=0.8)
    
    # Add value labels
    for i, score in enumerate(top_scores):
        ax.text(
            score + score * 0.05,
            i,
            f"{score:.4f}",
            ha="left",
            va="center"
        )
    
    # Set labels and title
    ax.set_yticks(y_pos)
    # Show feature indices instead of ranks (Feature 1, Feature 2, etc.)
    ax.set_yticklabels([f"Feature {int(idx)}" for idx in top_indices])
    ax.set_xlabel("Importance Score")
    ax.set_title(title)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if output path is provided
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    
    return fig

def compare_layer_effects(
    causal_matrices: Dict[str, Dict[str, np.ndarray]],
    target: str,
    source_layers: List[str],
    top_n_per_layer: int = 10,
    importance_scores: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 10),
    x_tick_labels: Optional[List[str]] = None
) -> plt.Figure:
    """
    Create a "reverse causal" plot showing how top features from different layers affect the same target.
    
    Args:
        causal_matrices: Dict mapping source layers to their causal matrix dict {target: matrix}
        target: Target to analyze (e.g., "mu" or "value")
        source_layers: List of source layers to compare
        top_n_per_layer: Number of top features to include from each layer
        importance_scores: Dict mapping source layers to {target: importance} (if precomputed)
        output_path: Path to save the figure
        figsize: Figure size
        x_tick_labels: Custom labels for x-axis ticks (e.g., joint names for actions)
        
    Returns:
        Matplotlib figure
    """
    # Create figure with subplots for each source layer
    fig, axes = plt.subplots(len(source_layers), 1, figsize=figsize, sharex=True)
    if len(source_layers) == 1:
        axes = [axes]
    
    # Determine global color scale for consistent comparison
    vmax_global = 0
    for layer in source_layers:
        if layer in causal_matrices and target in causal_matrices[layer]:
            matrix = causal_matrices[layer][target]
            vmax_layer = np.percentile(np.abs(matrix), 99)
            vmax_global = max(vmax_global, vmax_layer)
    
    vmin_global = -vmax_global
    
    # Process each layer
    for i, layer in enumerate(source_layers):
        ax = axes[i]
        
        if layer not in causal_matrices or target not in causal_matrices[layer]:
            ax.text(0.5, 0.5, f"No data for {layer} → {target}", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes)
            continue
        
        # Get causal matrix for this layer-target combination
        matrix = causal_matrices[layer][target]
        
        # Get top features for this layer based on importance to target
        if importance_scores and layer in importance_scores and target in importance_scores[layer]:
            importance = importance_scores[layer][target]
            top_indices = np.argsort(importance)[::-1][:top_n_per_layer]
        else:
            # If importance scores not provided, compute them
            importance = np.linalg.norm(matrix, axis=1)
            top_indices = np.argsort(importance)[::-1][:top_n_per_layer]
        
        # Extract matrix for top features only
        top_matrix = matrix[top_indices]
        
        # Create heatmap
        im = ax.imshow(top_matrix, cmap='RdBu_r', vmin=vmin_global, vmax=vmax_global, aspect='auto')
        
        # Add y-axis labels for feature indices
        ax.set_yticks(np.arange(len(top_indices)))
        ax.set_yticklabels([f"F{int(idx)}" for idx in top_indices])
        
        # Add x-axis labels based on target type
        if target == "mu" and matrix.shape[1] == 9 and x_tick_labels:
            ax.set_xticks(np.arange(9))
            if i == len(source_layers) - 1:  # Only on bottom subplot
                ax.set_xticklabels(x_tick_labels, rotation=45, ha='right')
        
        # Add layer label
        ax.set_ylabel(layer)
        
        # Add title to first subplot
        if i == 0:
            ax.set_title(f"Effect of Top Features from Different Layers on {target}")
    
    # Add colorbar to the right of the subplots
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Effect Size')
    
    # Add common x-axis label
    if target == "mu":
        fig.text(0.5, 0.04, "Action Dimensions", ha='center')
    elif target == "value":
        fig.text(0.5, 0.04, "Value", ha='center')
    else:
        fig.text(0.5, 0.04, f"{target} Features", ha='center')
    
    # Improve layout
    plt.tight_layout(rect=[0, 0.05, 0.9, 0.95])
    
    # Save figure if output path is provided
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    
    return fig

# TODO: unused, double check
def plot_feature_ablation_effect(
    causal_matrix: np.ndarray,
    feature_idx: int,
    title: Optional[str] = None,
    target_labels: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot the effect of ablating a specific feature on all targets.
    
    Args:
        causal_matrix: Matrix of causal effects [source_features x target_features]
        feature_idx: Index of the feature to analyze
        title: Plot title (defaults to "Effect of Ablating Feature {feature_idx}")
        target_labels: Labels for target dimensions
        output_path: Path to save the figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Extract the effect of this feature
    feature_effect = causal_matrix[feature_idx, :]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create bar chart
    x = np.arange(len(feature_effect))
    ax.bar(x, feature_effect, color='royalblue', alpha=0.8)
    
    # Add zero line
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # Set labels and title
    if target_labels is None:
        if causal_matrix.shape[1] == 9:  # Assuming 9-dim action space
            target_labels = [f"A{i}" for i in range(9)]
        else:
            target_labels = [str(i) for i in range(len(feature_effect))]
    
    ax.set_xticks(x)
    ax.set_xticklabels(target_labels, rotation=45 if len(target_labels) > 10 else 0)
    
    if title is None:
        title = f"Effect of Ablating Feature {feature_idx}"
    ax.set_title(title)
    
    ax.set_ylabel("Effect Size")
    
    # Improve layout
    plt.tight_layout()
    
    # Save figure if output path is provided
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    
    return fig


# TODO: unused, double check
def compare_feature_effects(
    causal_matrices: Dict[str, np.ndarray],
    feature_idx: int,
    title: Optional[str] = None,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Compare effects of ablating the same feature on different targets.
    
    Args:
        causal_matrices: Dictionary mapping target names to causal matrices
        feature_idx: Index of the feature to analyze
        title: Plot title (defaults to "Comparing Effects of Feature {feature_idx}")
        output_path: Path to save the figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot effects for each target
    for i, (target_name, matrix) in enumerate(causal_matrices.items()):
        # Skip targets with too many dimensions
        if matrix.shape[1] > 20:
            continue
            
        feature_effect = matrix[feature_idx, :]
        
        # For targets with multiple dimensions, compute summary statistics
        if matrix.shape[1] > 1:
            ax.bar(
                i, 
                np.mean(np.abs(feature_effect)),
                yerr=np.std(feature_effect),
                capsize=5,
                label=f"{target_name} (mean abs)",
                alpha=0.7
            )
        else:
            ax.bar(
                i,
                feature_effect[0],
                label=target_name,
                alpha=0.7
            )
    
    # Set labels and title
    ax.set_xticks(np.arange(len(causal_matrices)))
    ax.set_xticklabels(list(causal_matrices.keys()))
    
    if title is None:
        title = f"Comparing Effects of Feature {feature_idx}"
    ax.set_title(title)
    
    ax.set_ylabel("Effect Size")
    
    # Add legend if needed
    if len(causal_matrices) > 1:
        ax.legend()
    
    # Improve layout
    plt.tight_layout()
    
    # Save figure if output path is provided
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    
    return fig


# TODO: havent checked this
def visualize_clusters(
    cluster_matrix: np.ndarray,
    cluster_labels: List[str],
    title: str,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    Visualize clustered feature patterns.
    
    Args:
        cluster_matrix: Matrix of cluster patterns [n_clusters x target_features]
        cluster_labels: Labels for each cluster
        title: Plot title
        output_path: Path to save the figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Determine appropriate color scaling
    vmax = np.percentile(np.abs(cluster_matrix), 99)
    vmin = -vmax
    
    # Create heatmap using matplotlib's imshow
    im = ax.imshow(cluster_matrix, cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='auto')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Effect Size')
    
    # Set labels and title
    ax.set_yticks(np.arange(len(cluster_labels)))
    ax.set_yticklabels(cluster_labels)
    
    if cluster_matrix.shape[1] == 1:
        ax.set_xlabel("Value")
    elif cluster_matrix.shape[1] == 9:  # Assuming 9-dim action space
        ax.set_xlabel("Action Dimensions")
        # Add labels for action dimensions
        ax.set_xticks(np.arange(9))
        ax.set_xticklabels([f"A{i}" for i in range(9)])
    else:
        ax.set_xlabel("Target Features")
    
    ax.set_title(title)
    
    # Improve layout
    plt.tight_layout()
    
    # Save figure if output path is provided
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    
    return fig
