"""
script to get PCA components and save it
"""
import os

from src.activation_dataset import ActivationDataset
from src.pca_analyzer import PCAAnalyzer

CHECKPOINT_DIR = './ckpts/2025-02-13_09-26-08'
target_layers = ['actor_mlp_0']


def create_layer_components_map(layer_names, default_components=None):
    """
    Create a mapping of layer names to component counts.
    
    Args:
        layer_names: List of layer names to analyze
        default_components: Default number of components (None means all)
        
    Returns:
        Dict mapping layer names to component counts
    """
    return {layer: default_components for layer in layer_names}


if __name__ == "__main__":
    dataset = ActivationDataset()
    dataset.load(os.path.join(CHECKPOINT_DIR, 'strong_activation_dataset.pkl'))
    
    output_path = os.path.join(CHECKPOINT_DIR, 'pca_components.npz')
    
    # Create layer_components_map with None to collect all components
    layer_components_map = create_layer_components_map(target_layers, default_components=None)
    
    pca_analyzer = PCAAnalyzer()
    
    # Run PCA for all specified layers and save to a combined file
    results = pca_analyzer.run_pca_for_multiple_layers(
        dataset=dataset,
        layer_components_map=layer_components_map,
        output_path=output_path
    )
    
    print(f"PCA analysis complete. Results saved to {output_path}")

    for layer_name, layer_results in results.items():
        print(f"Layer: {layer_name}")
        print(f"Number of components: {layer_results['components'].shape[0]}")
        print(f"Total variance explained: {layer_results['explained_variance_ratio'].sum():.4f}")
        print(f"Original feature dimension: {layer_results['components'].shape[1]}")
        print()
