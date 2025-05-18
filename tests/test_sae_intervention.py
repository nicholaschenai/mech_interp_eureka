#!/usr/bin/env python3
"""
Sanity checks for scripts/sae/collect_sae_intervention_data.py

This test verifies that ablating SAE features in one layer doesn't affect
the activations in previous layers. 

Specifically:

1. When ablating the middle layer SAE features, first layer normal model activations should be unchanged
2. When ablating the last layer SAE features, middle layer normal model activations should be unchanged
"""
import os
import torch
import numpy as np
import random
from typing import Dict

from src.activation_dataset import ActivationDataset
from src.sae.utils import SAE_LAYERS_DEFAULT, load_sae_models
from src.sae.sae_wrapper import wrap_model_with_sae, SAENetworkWrapper
from utils.model_utils import load_model, load_data, unwrap_data, compute_model_outputs

# Constants
CHECKPOINT_DIR = "./ckpts/2025-02-13_09-26-08"
DATA_DIR = os.path.join(CHECKPOINT_DIR, 'sae_interventions', 'data')
MODEL_NAME = "strong"
MODEL_FILE = "FrankaCabinetGPT_epoch__eval.pth"


def compare_activations(baseline_outputs, ablated_outputs, layer_name, epsilon=1e-5):
    """
    Returns:
        bool: True if activations are the same (within epsilon), False otherwise
    """
    print(baseline_outputs.shape, ablated_outputs.shape)
    # Check values
    max_diff = torch.max(torch.abs(baseline_outputs - ablated_outputs)).item()
    if max_diff > epsilon:
        print(f"Error: Values differ for {layer_name} - max difference: {max_diff}")
        return False
    
    print(f"✓ {layer_name} activations match (max diff: {max_diff:.8f})")
    return True


def collect_activations(
    model: SAENetworkWrapper,
    observations,
    original_mu,
    original_value,
    ablation_layer,
    feature_idx
) -> Dict[str, torch.Tensor]:
    """
    Collect activations when doing feature ablation
    
    Args:
        model: Model with SAE hooks
        observations: Input observations
        original_mu: Original mu outputs
        original_value: Original value outputs
        ablation_layer: Layer to ablate (None for baseline)
        feature_idx: Feature index to ablate
    
    Returns:
        Dict of layer outputs
    """
    model.clear_accumulated_features()
    model.clear_ablation()
    for layer_name, hook in model.a2c_network.sae_manager.hooks.items():
        hook.clear_accumulated_outputs()
    
    model.ablate_feature(ablation_layer, feature_idx)
    
    for layer_name, hook in model.a2c_network.sae_manager.hooks.items():
        hook.start_accumulating_outputs()
    
    # Note: mu, value not relevant! we only care about activations!
    # Run forward pass
    _ = compute_model_outputs(model, observations, original_mu, original_value)
    
    outputs = {}
    for layer_name, hook in model.a2c_network.sae_manager.hooks.items():
        outputs[layer_name] = hook.get_accumulated_outputs()

    for layer_name, hook in model.a2c_network.sae_manager.hooks.items():
        hook.stop_accumulating_outputs()
    
    model.clear_ablation()
    
    return outputs


if __name__ == "__main__":
    print("Running SAE intervention sanity checks...")
    
    activation_dataset_path = os.path.join(CHECKPOINT_DIR, 'strong_activation_dataset.pkl')
    activation_dataset = ActivationDataset()
    activation_dataset.load(activation_dataset_path)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(MODEL_FILE, CHECKPOINT_DIR).to(device)
    data = load_data(MODEL_FILE, CHECKPOINT_DIR)
    observations, original_mu, original_value = unwrap_data(data)
    
    sae_models = load_sae_models(os.path.join(CHECKPOINT_DIR, 'sae_models'), MODEL_NAME)

    first_layer, middle_layer, last_layer = SAE_LAYERS_DEFAULT
    
    # Build wrapped model with SAE hooks
    sae_wrapped_model = wrap_model_with_sae(
        model=model,
        sae_models=sae_models,
        layer_names=SAE_LAYERS_DEFAULT
    )
    
    print(f"\nLayers for testing:\n- First: {first_layer}\n- Middle: {middle_layer}\n- Last: {last_layer}")
    
    # Choose a random feature to ablate for each test
    middle_feature_idx = random.randint(0, sae_models[middle_layer].hidden_dim - 1)
    last_feature_idx = random.randint(0, sae_models[last_layer].hidden_dim - 1)
    
    print(f"\nTest 1: Ablating feature {middle_feature_idx} in {middle_layer}")
    middle_ablation_results = collect_activations(
        model=sae_wrapped_model,
        observations=observations,
        original_mu=original_mu,
        original_value=original_value,
        ablation_layer=middle_layer,
        feature_idx=middle_feature_idx
    )
    
    # Test 1: When ablating middle layer, first layer activations should be unchanged
    baseline_first_outputs = activation_dataset.get_activation_matrix(first_layer)
    
    # Convert numpy array to torch tensor for comparison if needed
    if isinstance(baseline_first_outputs, np.ndarray):
        baseline_first_outputs = torch.from_numpy(baseline_first_outputs)
    
    print("\nTest 1 Results:")
    first_layer_unchanged = compare_activations(
        baseline_first_outputs, 
        middle_ablation_results[first_layer], 
        f"{first_layer} (when ablating {middle_layer})"
    )
    
    print(f"\nTest 2: Ablating feature {last_feature_idx} in {last_layer}")
    last_ablation_results = collect_activations(
        model=sae_wrapped_model,
        observations=observations,
        original_mu=original_mu,
        original_value=original_value,
        ablation_layer=last_layer,
        feature_idx=last_feature_idx
    )
    
    # Test 2: When ablating last layer, middle layer activations should be unchanged
    # Get outputs to middle layer from both baseline and ablated runs
    baseline_middle_outputs = activation_dataset.get_activation_matrix(middle_layer)
    
    # Convert numpy array to torch tensor for comparison if needed
    if isinstance(baseline_middle_outputs, np.ndarray):
        baseline_middle_outputs = torch.from_numpy(baseline_middle_outputs)
    
    print("\nTest 2 Results:")
    middle_layer_unchanged = compare_activations(
        baseline_middle_outputs, 
        last_ablation_results[middle_layer], 
        f"{middle_layer} (when ablating {last_layer})"
    )
    
    # Final summary
    print("\nSanity Check Summary:")
    if first_layer_unchanged and middle_layer_unchanged:
        print("✓ All sanity checks passed!")
    else:
        print("✗ Some sanity checks failed!")
        if not first_layer_unchanged:
            print(f"  - Ablating {middle_layer} affected {first_layer} activations (should be unchanged)")
        if not middle_layer_unchanged:
            print(f"  - Ablating {last_layer} affected {middle_layer} activations (should be unchanged)")
