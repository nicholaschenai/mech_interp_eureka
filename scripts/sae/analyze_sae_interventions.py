#!/usr/bin/env python3
"""
ROLE: Analysis and Visualization of SAE Causal Intervention Data

This script is responsible for loading, analyzing, and visualizing data from
causal intervention experiments on SAE features.

Responsibilities:
1. Load baseline features and ablation results from files
2. Compute causal matrices (deltas) between original and ablated features/outputs
3. Analyze how feature ablations affect downstream layers and outputs
4. Produce visualizations (causal matrices, feature importance rankings)
5. Identify important features and their effects
6. Optionally cluster features with similar causal patterns

'reverse causal' plots: For each of mu and value, plot these top features from each of the 3 layers side by side but using the same scale for coloring, to see if different layers affect the mu and value differently

This script is used after collect_sae_intervention_data.py has been run to generate the data.
"""
import os
import argparse
import pickle
import torch

import numpy as np
import matplotlib.pyplot as plt

from typing import Dict, List, Any

from src.sae.intervention_analysis import (
    compute_causal_matrices,
    compute_feature_importance, 
    get_top_features,
    cluster_features,
    get_cluster_patterns
)

from src.sae.intervention_viz import (
    visualize_causal_matrix,
    plot_feature_importance,
    visualize_clusters,
    compare_layer_effects
)

from utils.model_utils import unwrap_data, load_data
from utils import joint_names

# Constants
STRONG_MODEL_FILE = "FrankaCabinetGPT_epoch__eval.pth"
CHECKPOINT_DIR = "./ckpts/2025-02-13_09-26-08"
DEFAULT_DATA_DIR = f"{CHECKPOINT_DIR}/sae_interventions/data"

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze SAE causal intervention data")
    
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR,
                        help="Directory containing intervention data")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save analysis results (default: data_dir/analysis)")
    parser.add_argument("--model_name", type=str, default="strong",
                        help="Model name prefix for data files")
    parser.add_argument("--source_layer", type=str, default=None,
                        help="Specific source layer to analyze (default: analyze all available)")
    parser.add_argument("--top_n_features", type=int, default=20,
                        help="Number of top features to highlight in importance plots")
    parser.add_argument("--cluster_features", action="store_true",
                        help="Perform clustering of features with similar causal patterns")
    parser.add_argument("--n_clusters", type=int, default=10,
                        help="Number of clusters for feature clustering")
    parser.add_argument("--exclude_mu", action="store_true",
                        help="Exclude mu (action) from analysis")
    parser.add_argument("--exclude_value", action="store_true",
                        help="Exclude value from analysis")
    parser.add_argument("--model_fname", type=str, default=STRONG_MODEL_FILE,
                        help="Path to trained model checkpoint")
    parser.add_argument("--checkpoint_dir", type=str, default=CHECKPOINT_DIR, 
                        help="Checkpoint directory containing model and data")
    return parser.parse_args()

def load_baseline_features(data_dir: str, model_name: str) -> Dict[str, torch.Tensor]:
    """Load baseline features with no ablation"""
    baseline_path = os.path.join(data_dir, f"{model_name}_sae_features.pkl")
    if not os.path.exists(baseline_path):
        raise FileNotFoundError(f"Baseline features file not found: {baseline_path}")
    
    with open(baseline_path, 'rb') as f:
        return pickle.load(f)

def get_feature_file_path(data_dir: str, source_layer: str, feature_idx: int) -> str:
    """Get the file path for a specific feature's ablation results"""
    return os.path.join(data_dir, source_layer, f"feature_{feature_idx}.pkl")

def count_feature_files(data_dir: str, source_layer: str) -> int:
    """Count the number of feature files for a source layer"""
    layer_dir = os.path.join(data_dir, source_layer)
    if not os.path.exists(layer_dir):
        return 0
    
    feature_files = [f for f in os.listdir(layer_dir) if f.startswith("feature_") and f.endswith(".pkl")]
    return len(feature_files)

def create_feature_loader(data_dir: str, source_layer: str):
    """Create a function that loads feature data on demand"""
    
    def load_feature(feature_idx: int):
        feature_path = get_feature_file_path(data_dir, source_layer, feature_idx)
        if os.path.exists(feature_path):
            with open(feature_path, 'rb') as f:
                return pickle.load(f)
        return None
    
    return load_feature

def get_available_source_layers(data_dir: str, model_name: str) -> List[str]:
    """Find all available source layers with ablation data"""
    source_layers = []
    
    # Check for directories
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path) and any(f.startswith("feature_") and f.endswith(".pkl") for f in os.listdir(item_path)):
            source_layers.append(item)
    
    # If no directories found, check for old-format files
    if not source_layers:
        for filename in os.listdir(data_dir):
            if filename.startswith(f"{model_name}_ablation_results_") and filename.endswith(".pkl"):
                layer = filename.replace(f"{model_name}_ablation_results_", "").replace(".pkl", "")
                source_layers.append(layer)
    
    return source_layers

def create_reverse_causal_plots(
    all_results: Dict[str, Dict[str, Any]],
    source_layers: List[str],
    output_dir: str,
    top_n_per_layer: int = 10
):
    """
    Create reverse causal plots that compare how features from different layers
    affect the same target (mu or value).
    
    Args:
        all_results: Dictionary of analysis results from each source layer
        source_layers: List of source layers to compare
        output_dir: Directory to save the plots
        top_n_per_layer: Number of top features to show from each layer
    """
    # Collect causal matrices and importance scores for each layer
    causal_matrices = {}
    importance_scores = {}
    
    for layer in source_layers:
        # Load the causal matrices for this layer
        matrix_path = os.path.join(output_dir, layer, "causal_matrices.pkl")
        if not os.path.exists(matrix_path):
            print(f"Warning: Causal matrix file not found for {layer}, skipping from reverse causal plots")
            continue
            
        with open(matrix_path, 'rb') as f:
            causal_matrices[layer] = pickle.load(f)
        
        # Collect importance scores from results
        if layer in all_results:
            importance_scores[layer] = all_results[layer]["importance_scores"]
    
    # Create reverse causal plots for mu and value
    for target in ["mu", "value"]:
        # Skip if any layer doesn't have this target
        if any(layer in causal_matrices and target not in causal_matrices[layer] for layer in source_layers):
            print(f"Skipping reverse causal plot for {target} as not all layers have this target")
            continue
        
        # Create the plot
        fig = compare_layer_effects(
            causal_matrices=causal_matrices,
            target=target,
            source_layers=source_layers,
            top_n_per_layer=top_n_per_layer,
            importance_scores=importance_scores,
            output_path=os.path.join(output_dir, f"reverse_causal_{target}.png"),
            x_tick_labels=joint_names if target == "mu" else None
        )
        plt.close(fig)
        
        print(f"Created reverse causal plot for {target}")

def analyze_source_layer(
    baseline_features: Dict[str, torch.Tensor],
    baseline_mu: np.ndarray,
    baseline_value: np.ndarray,
    data_dir: str,
    source_layer: str,
    layer_names: List[str],
    output_dir: str,
    top_n_features: int = 20,
    cluster_features_flag: bool = False,
    n_clusters: int = 10,
    exclude_mu: bool = False,
    exclude_value: bool = False
) -> Dict[str, Any]:
    """
    Analyze the causal effects of ablating features in a source layer.
    
    Args:
        baseline_features: Original features without ablation
        baseline_mu: Original mu output without ablation
        baseline_value: Original value output without ablation
        data_dir: Directory containing feature data files
        source_layer: Layer where features were ablated
        layer_names: Names of all layers in order
        output_dir: Directory to save analysis results
        top_n_features: Number of top features to highlight
        cluster_features_flag: Whether to perform feature clustering
        n_clusters: Number of clusters for feature clustering
        exclude_mu: Whether to exclude mu from analysis
        exclude_value: Whether to exclude value from analysis
        
    Returns:
        Dictionary of analysis results
    """
    print(f"\nAnalyzing effects from {source_layer}...")
    
    # Create layer-specific output directory
    layer_dir = os.path.join(output_dir, source_layer)
    os.makedirs(layer_dir, exist_ok=True)
    
    causal_matrix_path = os.path.join(layer_dir, "causal_matrices.pkl")

    if os.path.exists(causal_matrix_path):
        print(f"Loading causal matrices from {causal_matrix_path}")
        with open(causal_matrix_path, 'rb') as f:
            causal_matrices = pickle.load(f)
    else:
        num_features = count_feature_files(data_dir, source_layer)
        feature_loader = create_feature_loader(data_dir, source_layer)

        causal_matrices = compute_causal_matrices(
            baseline_features=baseline_features,
            baseline_mu=baseline_mu,
            baseline_value=baseline_value,
            feature_loader=feature_loader,
            source_layer=source_layer,
            layer_names=layer_names,
            num_features=num_features,
            exclude_mu=exclude_mu,
            exclude_value=exclude_value
        )
        with open(os.path.join(layer_dir, "causal_matrices.pkl"), "wb") as f:
            pickle.dump(causal_matrices, f)
        
    # Initialize results dictionary
    results = {
        "importance_scores": {},
        "top_features": {},
        "clusters": {}
    }
    
    # Process each target
    for target_name, causal_matrix in causal_matrices.items():
        print(f"  Analyzing {source_layer} → {target_name} (matrix shape: {causal_matrix.shape})")
        
        importance_scores = compute_feature_importance(causal_matrix)
        results["importance_scores"][target_name] = importance_scores
        
        top_indices = get_top_features(importance_scores, top_n_features)
        results["top_features"][target_name] = top_indices
        
        # Create descriptive titles
        if target_name == "mu":
            title_suffix = "Actions (mu)"
            x_axis_label = "Action Dimensions"
        elif target_name == "value":
            title_suffix = "Value Estimate"
            x_axis_label = "Value"
        else:
            title_suffix = f"{target_name}"
            x_axis_label = f"{target_name} Features"
        
        # Visualize causal matrix with all features but prioritizing top ones
        fig = visualize_causal_matrix(
            causal_matrix=causal_matrix,
            title=f"Causal Influence: {source_layer} → {title_suffix}",
            top_indices=top_indices,
            max_features=top_n_features,
            output_path=os.path.join(layer_dir, f"causal_matrix_{target_name}.png"),
            x_axis_label=x_axis_label,
            x_tick_labels=joint_names if target_name == "mu" else None
        )
        plt.close(fig)
        
        # Plot feature importance
        fig = plot_feature_importance(
            importance_scores=importance_scores,
            title=f"Feature Importance: {source_layer} → {title_suffix}",
            top_n=top_n_features,
            top_indices=top_indices,
            output_path=os.path.join(layer_dir, f"feature_importance_{target_name}.png")
        )
        plt.close(fig)
        
        # TODO: havent checked this, leave for another time
        # # Perform clustering if requested
        # if cluster_features_flag and causal_matrix.shape[0] > n_clusters:
        #     clusters, cluster_centers = cluster_features(
        #         causal_matrix=causal_matrix,
        #         n_clusters=n_clusters
        #     )
        #     results["clusters"][target_name] = clusters
            
        #     # Get cluster patterns for visualization
        #     cluster_matrix, cluster_labels = get_cluster_patterns(
        #         causal_matrix=causal_matrix,
        #         clusters=clusters
        #     )
            
        #     # Visualize clusters
        #     fig = visualize_clusters(
        #         cluster_matrix=cluster_matrix,
        #         cluster_labels=cluster_labels,
        #         title=f"Feature Clusters: {source_layer} → {title_suffix}",
        #         output_path=os.path.join(layer_dir, f"clusters_{target_name}.png")
        #     )
        #     plt.close(fig)
            
        #     # Save cluster information
        #     with open(os.path.join(layer_dir, f"clusters_{target_name}.pkl"), "wb") as f:
        #         pickle.dump({
        #             "clusters": clusters,
        #             "cluster_centers": cluster_centers
        #         }, f)
    
    return results

def generate_summary_report(
    all_results: Dict[str, Dict[str, Any]],
    output_dir: str
) -> None:
    """
    Generate a summary report of all analysis results.
    
    Args:
        all_results: Dictionary mapping source layers to their analysis results
        output_dir: Directory to save the summary report
    """
    with open(os.path.join(output_dir, "summary_report.txt"), "w") as f:
        f.write("SAE Causal Intervention Analysis Summary\n")
        f.write("=======================================\n\n")
        
        for source_layer, results in all_results.items():
            f.write(f"Source Layer: {source_layer}\n")
            f.write("="*50 + "\n\n")
            
            for target_name, importance_scores in results["importance_scores"].items():
                f.write(f"Target: {target_name}\n")
                f.write("-"*50 + "\n")
                
                # Top features
                if target_name in results["top_features"]:
                    top_indices = results["top_features"][target_name]
                    f.write("\nTop Features:\n")
                    for i, idx in enumerate(top_indices[:10]):  # Just show top 10 in report
                        f.write(f"  {i+1}. Feature {int(idx)} (importance: {importance_scores[idx]:.4f})\n")
                
                # TODO: havent checked this, leave for another time
                # Cluster information
                # if target_name in results.get("clusters", {}):
                #     clusters = results["clusters"][target_name]
                #     f.write("\nFeature Clusters:\n")
                #     for cluster_id, feature_indices in sorted(clusters.items()):
                #         f.write(f"  Cluster {cluster_id}: {len(feature_indices)} features\n")
                
                f.write("\n")
            
            f.write("\n")
        
        f.write("\nAnalysis completed successfully.\n")

def main():
    """Run the analysis."""
    args = parse_args()
    
    data = load_data(args.model_fname, args.checkpoint_dir)
    _, baseline_mu, baseline_value = unwrap_data(data)

    # Set output directory if not specified
    if args.output_dir is None:
        args.output_dir = os.path.join(args.data_dir, "analysis")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    

    baseline_features = load_baseline_features(args.data_dir, args.model_name)
    print(f"Loaded baseline features: {', '.join(baseline_features.keys())}")

    # Get available source layers
    if args.source_layer:
        source_layers = [args.source_layer]
    else:
        source_layers = get_available_source_layers(args.data_dir, args.model_name)
    
    if not source_layers:
        print(f"Error: No source layers with ablation data found in {args.data_dir}")
        return
    
    print(f"Found data for source layers: {', '.join(source_layers)}")
    
    # Get all layer names
    layer_names = list(baseline_features.keys())
    
    # Analyze each source layer
    all_results = {}
    for source_layer in source_layers:

        # Analyze this source layer using on-demand feature loading
        results = analyze_source_layer(
            baseline_features=baseline_features,
            baseline_mu=baseline_mu,
            baseline_value=baseline_value,
            data_dir=args.data_dir,
            source_layer=source_layer,
            layer_names=layer_names,
            output_dir=args.output_dir,
            top_n_features=args.top_n_features,
            cluster_features_flag=args.cluster_features,
            n_clusters=args.n_clusters,
            exclude_mu=args.exclude_mu,
            exclude_value=args.exclude_value
        )
        
        all_results[source_layer] = results
            
    # Generate summary report
    if all_results:
        generate_summary_report(all_results, args.output_dir)
        
        # Create reverse causal plots comparing effects of different layers on mu and value
        if len(source_layers) > 1 and not (args.exclude_mu and args.exclude_value):
            create_reverse_causal_plots(
                all_results=all_results,
                source_layers=source_layers,
                output_dir=args.output_dir,
                top_n_per_layer=10
            )
        
        print(f"\nAnalysis complete! Results saved to {args.output_dir}")
        print("See summary_report.txt for an overview of the findings")
    else:
        print("No valid data could be analyzed")

if __name__ == "__main__":
    main()
