"""
Analyze correlations between input features and SAE features.

This script analyzes how input features correlate with Sparse Autoencoder (SAE) features.
It loads an activation dataset containing input features and SAE feature activations,
then computes correlation metrics between them to identify relationships.

The analysis helps understand how interpretable input features (transformed from raw inputs)
relate to learned SAE features, providing insights into what the SAE has captured.

Key functionalities:
1. Load input feature dataset and SAE feature activations
2. Compute correlation matrices between input and SAE features for each layer
3. Identify top SAE features using both maximum and average absolute correlation metrics
4. Visualize correlation relationships with layer-specific heatmaps
5. Save both visualization results and raw data for further analysis
"""
import os
import torch
import pickle
import argparse

import numpy as np
import matplotlib.pyplot as plt

from typing import Dict, List, Tuple, Optional

from src.activation_dataset import ActivationDataset
from utils.base_feature_extractor import get_all_feature_keys

# Constants
CHECKPOINT_DIR = "./ckpts/2025-02-13_09-26-08"
DEFAULT_INPUT_DATA = os.path.join(CHECKPOINT_DIR, 'strong_activation_dataset.pkl')
DEFAULT_SAE_DIR = os.path.join(CHECKPOINT_DIR, 'sae_interventions', 'data')
DEFAULT_OUTPUT_DIR = os.path.join(CHECKPOINT_DIR, 'input_sae_feature_correlation')
DEFAULT_MODEL_NAME = "strong"
DEFAULT_TOP_N = 20

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze correlations between input features and SAE features")
    
    parser.add_argument("--input_data", type=str, default=DEFAULT_INPUT_DATA,
                        help="Path to input features activation dataset")
    parser.add_argument("--sae_features", type=str, default=None,
                        help="Path to SAE baseline features file")
    parser.add_argument("--sae_dir", type=str, default=DEFAULT_SAE_DIR,
                        help="Directory containing SAE intervention data")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME,
                        help="Model name prefix for SAE models")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Directory to save correlation results and visualizations")
    parser.add_argument("--top_n", type=int, default=DEFAULT_TOP_N,
                        help="Number of top correlated SAE features to select per input feature")
    parser.add_argument("--force", action="store_true",
                        help="Force recomputation even if output files exist")
    
    return parser.parse_args()

def load_input_features(data_path: str) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    Load input features from activation dataset.
    
    Args:
        data_path: Path to activation dataset pickle file
    
    Returns:
        Tuple of (features_dict, feature_names) where:
        - features_dict maps feature names to 1D arrays
        - feature_names is a list of feature names
    """
    print(f"Loading input features from {data_path}")
    dataset = ActivationDataset()
    dataset.load(data_path)
    
    # Get feature names from dataset metadata
    feature_names = get_all_feature_keys()
    
    # Extract features as flattened arrays
    features_dict = {}
    for name in feature_names:
        features_dict[name] = dataset.get_flattened_feature(name)
    
    return features_dict, feature_names

def load_sae_features(sae_dir: str, model_name: str, sae_features_path: Optional[str] = None) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load SAE features from baseline results.
    
    Args:
        sae_dir: Directory containing SAE intervention data
        model_name: Model name prefix for SAE files
        sae_features_path: Optional direct path to SAE features file
    
    Returns:
        Dictionary mapping layer names to feature activations
    """
    if sae_features_path is None:
        sae_features_path = os.path.join(sae_dir, f"{model_name}_sae_features.pkl")
    
    print(f"Loading SAE features from {sae_features_path}")
    with open(sae_features_path, 'rb') as f:
        sae_features = pickle.load(f)
    
    # Convert any torch tensors to numpy arrays
    processed_features = {}
    for layer_name, layer_features in sae_features.items():
        if isinstance(layer_features, torch.Tensor):
            processed_features[layer_name] = layer_features.cpu().numpy()
        else:
            processed_features[layer_name] = layer_features
    
    return processed_features

def process_sae_features(sae_features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Process SAE features into a format suitable for correlation analysis.
    
    Args:
        sae_features: Dictionary mapping layer names to feature tensors
    
    Returns:
        Dictionary mapping layer names to 2D matrices [samples, features]
    """
    processed_features = {}
    
    for layer_name, layer_features in sae_features.items():
        # Skip non-tensor entries
        if not isinstance(layer_features, (np.ndarray, torch.Tensor)):
            continue
            
        # Handle different tensor shapes
        # Expected format is typically [time, batch, features]
        feature_array = layer_features
        
        if isinstance(feature_array, torch.Tensor):
            feature_array = feature_array.cpu().numpy()
            
        # Reshape to 2D matrix [samples, features]
        if feature_array.ndim == 3:  # [time, batch, features]
            time_steps, batch_size, feature_dim = feature_array.shape
            feature_array = feature_array.reshape(time_steps * batch_size, feature_dim)
        elif feature_array.ndim > 3:  # More complex shape
            feature_array = feature_array.reshape(feature_array.shape[0] * feature_array.shape[1], -1)
        
        processed_features[layer_name] = feature_array
    
    return processed_features

def compute_correlations(input_features: Dict[str, np.ndarray], 
                         sae_features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Compute correlation between input features and SAE features for each layer.
    
    Args:
        input_features: Dictionary mapping input feature names to 1D arrays
        sae_features: Dictionary mapping layer names to 2D arrays [samples, features]
    
    Returns:
        Dictionary mapping layer names to correlation matrices [input_features, sae_features]
    """
    correlation_matrices = {}
    
    # Create input feature matrix (samples × input_features)
    input_names = list(input_features.keys())
    input_matrix = np.column_stack([input_features[name] for name in input_names])
    
    # For each SAE layer
    for layer_name, layer_features in sae_features.items():
        print(f"Computing correlations for layer {layer_name}")
        
        # Check if data dimensions match
        if input_matrix.shape[0] != layer_features.shape[0]:
            print(f"Warning: Sample count mismatch for layer {layer_name}. " 
                  f"Input features: {input_matrix.shape[0]}, SAE features: {layer_features.shape[0]}")
            continue
        
        # Compute correlation coefficient between each input feature and each SAE feature
        corr_matrix = np.zeros((len(input_names), layer_features.shape[1]))
        
        for i, input_name in enumerate(input_names):
            for j in range(layer_features.shape[1]):
                # Correlation coefficient (Pearson's r)
                corr = np.corrcoef(input_features[input_name], layer_features[:, j])[0, 1]
                corr_matrix[i, j] = corr if not np.isnan(corr) else 0.0
        
        correlation_matrices[layer_name] = corr_matrix
    
    return correlation_matrices, input_names

def get_top_features(
    correlation_matrices: Dict[str, np.ndarray], 
    score_fn: callable, 
    top_n: int = 10,
    score_name: str = "score"
) -> Dict[str, List[int]]:
    """
    For each layer, find top_n SAE features based on a scoring function applied to the correlation matrix.
    
    Args:
        correlation_matrices: Dictionary mapping layer names to correlation matrices
        score_fn: Function that takes a correlation matrix and returns a 1D score array for each SAE feature
        top_n: Number of top features to select
        score_name: Name of the scoring method (for debugging/logging)
    
    Returns:
        Dictionary mapping layer names to lists of selected feature indices
    """
    top_features = {}
    
    for layer_name, corr_matrix in correlation_matrices.items():
        # Calculate scores for each SAE feature using the provided function
        feature_scores = score_fn(corr_matrix)
        
        # Get indices of top features sorted by score
        top_indices = np.argsort(feature_scores)[-top_n:][::-1]
        top_features[layer_name] = top_indices.tolist()
    
    return top_features

def max_abs_correlation(corr_matrix: np.ndarray) -> np.ndarray:
    return np.max(np.abs(corr_matrix), axis=0)

def avg_abs_correlation(corr_matrix: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(corr_matrix), axis=0)

def visualize_correlations_by_metric(correlation_matrices: Dict[str, np.ndarray],
                                    top_features: Dict[str, List[int]],
                                    input_names: List[str],
                                    output_dir: str,
                                    metric_name: str):
    """
    Create subplot visualization of correlations for all layers based on a specific metric.
    
    Args:
        correlation_matrices: Dictionary mapping layer names to correlation matrices
        top_features: Dictionary mapping layer names to lists of top feature indices
        input_names: List of input feature names
        output_dir: Directory to save visualizations
        metric_name: Name of the metric used (e.g., 'max' or 'avg')
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Sort layer names to ensure consistent ordering
    layer_names = sorted(correlation_matrices.keys())
    n_layers = len(layer_names)
    
    # Skip if no layers
    if n_layers == 0:
        print("No layers to visualize.")
        return
    
    # Create a multi-panel figure
    n_cols = min(3, n_layers)  # Maximum 3 columns
    n_rows = (n_layers + n_cols - 1) // n_cols  # Ceiling division
    
    # Find global min/max for consistent color scale
    global_min = float('inf')
    global_max = float('-inf')
    
    # Create matrices for each layer first (for global min/max calculation)
    filtered_corr_matrices = {}
    feature_indices_by_layer = {}
    
    for layer_name in layer_names:
        if layer_name not in top_features:
            continue
            
        corr_matrix = correlation_matrices[layer_name]
        selected_indices = top_features[layer_name]
        
        # Create filtered correlation matrix
        filtered_corr = corr_matrix[:, selected_indices]
        filtered_corr_matrices[layer_name] = filtered_corr
        feature_indices_by_layer[layer_name] = selected_indices
        
        # Update global min/max
        global_min = min(global_min, np.min(filtered_corr))
        global_max = max(global_max, np.max(filtered_corr))
    
    # Create figure with appropriate size - INCREASED SIZE
    fig = plt.figure(figsize=(20, 6 * n_rows))
    
    # Create subplots for each layer
    for i, layer_name in enumerate(layer_names):
        if layer_name not in filtered_corr_matrices:
            continue
            
        # Get the filtered correlation matrix
        filtered_corr = filtered_corr_matrices[layer_name]
        selected_indices = feature_indices_by_layer[layer_name]
        
        # Create subplot
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        
        # Plot heatmap
        im = ax.imshow(filtered_corr, cmap='coolwarm', vmin=-1, vmax=1)
        
        # Add labels with larger font size
        ax.set_xlabel(f'Top {len(selected_indices)} SAE Features', fontsize=12)
        ax.set_ylabel('Input Features', fontsize=12)
        
        # Add title with larger font
        ax.set_title(f'Layer: {layer_name}', fontsize=14)
        
        # Add y-axis labels for input features with larger font
        ax.set_yticks(range(len(input_names)))
        ax.set_yticklabels(input_names, fontsize=11)
        
        # Add x-axis labels for SAE features with larger font
        feature_labels = [f'{idx}' for idx in selected_indices]
        ax.set_xticks(range(len(selected_indices)))
        ax.set_xticklabels(feature_labels, rotation=90, fontsize=10)
    
    # Add colorbar to the right of the figure
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax, label='Correlation Coefficient')
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('Correlation Coefficient', fontsize=14)
    
    # Add overall title
    fig.suptitle(f'Input vs SAE Feature Correlations (by {metric_name} absolute correlation)', fontsize=18)
    
    # Adjust layout with more space
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    
    # Save figure with higher DPI for better quality
    output_path = os.path.join(output_dir, f'input_sae_feature_correlation_by_{metric_name}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")
    
    # Close figure to free memory
    plt.close()
    
    # Save the raw data
    data_output = {
        'correlation_matrices': correlation_matrices,
        'top_features': top_features,
        'filtered_matrices': filtered_corr_matrices,
        'input_names': input_names,
        'feature_indices': feature_indices_by_layer,
        'metric': metric_name
    }
    
    data_path = os.path.join(output_dir, f'input_sae_feature_correlation_by_{metric_name}.pkl')
    with open(data_path, 'wb') as f:
        pickle.dump(data_output, f)

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load input features
    input_features, input_names = load_input_features(args.input_data)
    
    # Load SAE features
    sae_raw_features = load_sae_features(args.sae_dir, args.model_name, args.sae_features)
    
    # Process SAE features into consistent format
    sae_features = process_sae_features(sae_raw_features)
    print(f"Processing SAE features for {len(sae_features)} layers")
    
    # Compute correlations
    correlation_matrices, input_names = compute_correlations(input_features, sae_features)
    
    for score_fn, score_name in [(max_abs_correlation, "max"), (avg_abs_correlation, "avg")]:
        
        top_features = get_top_features(
            correlation_matrices, 
            score_fn=score_fn,
            top_n=args.top_n,
            score_name=score_name
        )
        
        visualize_correlations_by_metric(
            correlation_matrices, 
            top_features, 
            input_names, 
            args.output_dir,
            metric_name=score_name
        )
    
    print(f"Analysis complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
