"""
Script to collect raw model outputs when forcing activations through a reduced dimensionality space
using Principal Component Analysis (PCA). Results are saved for later analysis.
"""
import os
import copy
import numpy as np

from src.pca_projection_wrapper import wrap_model_with_pca

from utils import assert_outputs_match
from utils.model_utils import load_model, load_data, unwrap_data, compute_model_outputs

# constants to be defined
tolerance = 1e-5

STRONG_MODEL_FILE = "FrankaCabinetGPT_epoch__eval.pth"
CHECKPOINT_DIR = './ckpts/2025-02-13_09-26-08'

PCA_COMPONENTS_FILE = os.path.join(CHECKPOINT_DIR, 'pca_components.npz')

target_layers = ['actor_mlp_0']
component_counts = list(range(1, 6))


def validate_pca_components_file():
    print("Validating PCA components file...")
    if not os.path.exists(PCA_COMPONENTS_FILE):
        raise FileNotFoundError(
            f"PCA components file not found at {PCA_COMPONENTS_FILE}. "
            f"Run collect_pca.py first to generate the file."
        )
    pca_data = np.load(PCA_COMPONENTS_FILE)
    max_components = {}
    for layer in target_layers:
        component_key = f"{layer}_components"
        mean_key = f"{layer}_mean"
        
        if component_key not in pca_data:
            raise KeyError(f"Missing components for layer {layer}")
            
        if mean_key not in pca_data:
            raise KeyError(f"Missing mean for layer {layer}. This is required for proper PCA reconstruction.")
            
        print(f"Found components and mean for layer {layer}")
        print(f"  Components shape: {pca_data[component_key].shape}")
        print(f"  Mean shape: {pca_data[mean_key].shape}")
        
        # Get the total number of available components
        n_available_components = pca_data[component_key].shape[0]
        max_components[layer] = n_available_components
        # Check total variance explained
        variance_ratio_key = f"{layer}_explained_variance_ratio"
        if variance_ratio_key in pca_data:
            total_variance = pca_data[variance_ratio_key].sum()
            print(f"  Total variance explained by all {n_available_components} components: {total_variance:.4f} ({total_variance*100:.2f}%)")

    return max_components

def run_pca_model(model, data, layer_names, pca_n_components):
    """
    Common function to run a model with PCA projection and compute outputs.
    
    Args:
        model: The base model to wrap
        observations: Input observations
        original_mu, original_value: Original outputs to compare against
        data: Original data dictionary
        layer_names: List of layer names to apply PCA to
        n_components: Number of components to use for each layer (dict or int)
                     If None, use all available components
    
    Returns:
        Dictionary with computed outputs and difference metrics
    """
    observations, original_mu, original_value = unwrap_data(data)

    pca_wrapped_model = wrap_model_with_pca(
        model=copy.deepcopy(model),
        pca_components_file=PCA_COMPONENTS_FILE,
        layer_names=layer_names,
        pca_n_components=pca_n_components
    )

    computed_mu, computed_value, computed_norm_obs = compute_model_outputs(
        pca_wrapped_model, observations, original_mu, original_value
    )

    expected_norm_obs = data['post_norm_obs']
    norm_obs_diff = np.abs(computed_norm_obs - expected_norm_obs).max()

    # check that hooks dont affect normalized observations
    if norm_obs_diff >= tolerance:
        raise ValueError(f"Normalized observations differ: {norm_obs_diff:.6f}")
    
    results = {
        'mu': computed_mu,
        'value': computed_value,
    }
    
    return results


def validate_with_all_components(model, data, max_components):
    """
    Run validation using all available components as a sanity check.
    This should result in a perfect or near-perfect reconstruction.
    
    Returns:
        Results dictionary with full component outputs
    """
    print("\nRunning sanity check with all available components...")
    
    results = run_pca_model(
        model=model,
        data=data,
        layer_names=target_layers,
        pca_n_components=max_components
    )
    
    _, original_mu, original_value = unwrap_data(data)
    assert_outputs_match(results['mu'], results['value'], original_mu, original_value)
    

if __name__ == "__main__":
    max_components = validate_pca_components_file()
    
    model = load_model(STRONG_MODEL_FILE, CHECKPOINT_DIR)
    data = load_data(STRONG_MODEL_FILE, CHECKPOINT_DIR)
    
    validate_with_all_components(model, data, max_components)

    # Initialize results dictionary
    final_results = {
        'component_counts': component_counts,
        'target_layers': target_layers,
    }
    
    for n_components in component_counts:
        print(f"\nProcessing with {n_components} components...")
        pca_n_components = {layer: n_components for layer in target_layers}
        results = run_pca_model(
            model=model,
            data=data,
            layer_names=target_layers,
            pca_n_components=pca_n_components
        )
        
        print(f"Collected outputs for {n_components} components")

        final_results[f'mu_{n_components}'] = results['mu']
        final_results[f'value_{n_components}'] = results['value']
    
    results_file = os.path.join(CHECKPOINT_DIR, 'pca_ablation_outputs.npz')
    np.savez(results_file, **final_results)
    
    print(f"\nDone! All outputs saved to {results_file}")
