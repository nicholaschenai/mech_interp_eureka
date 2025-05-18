#!/usr/bin/env python3
"""
ROLE: Data Collection for SAE Causal Interventions

This script is responsible for systematically collecting data from SAE causal interventions.
It does NOT perform analysis or visualization of results.

Responsibilities:
1. Load trained model and SAE models for all layers
2. Systematically ablate SAE features in each source layer
3. Collect effects on all downstream SAE features and model outputs (mu, value)
4. Save raw data in a standardized format for later analysis

The collected data is saved as pickle files containing feature data that can be
loaded and analyzed by the analyze_sae_interventions.py script.
"""
import os
import torch
import argparse
import pickle
import copy

import numpy as np

from typing import Dict, Tuple
from tqdm import tqdm

from src.sae.sae_wrapper import wrap_model_with_sae, SAENetworkWrapper
from src.sae.utils import load_sae_models, SAE_LAYERS_DEFAULT

from utils import get_layer_order, assert_outputs_match
from utils.model_utils import load_model, load_data, unwrap_data, compute_model_outputs

# Constants
STRONG_MODEL_FILE = "FrankaCabinetGPT_epoch__eval.pth"
CHECKPOINT_DIR = "./ckpts/2025-02-13_09-26-08"

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Collect SAE causal intervention data across layers")
    
    parser.add_argument("--model_fname", type=str, default=STRONG_MODEL_FILE,
                        help="Path to trained model checkpoint")
    parser.add_argument("--checkpoint_dir", type=str, default=CHECKPOINT_DIR, 
                        help="Checkpoint directory containing model and data")
    parser.add_argument("--sae_dir", type=str, default=None,
                        help="Directory containing trained SAE models (default: checkpoint_dir/sae_models)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save results (default: checkpoint_dir/sae_interventions)")
    parser.add_argument("--model_name", type=str, default="strong",
                        help="Model name prefix for SAE models")
    parser.add_argument("--source_layers", type=str, nargs='+', default=SAE_LAYERS_DEFAULT,
                        help="Specific source layers to analyze (default: all layers)")
    parser.add_argument("--force", action="store_true",
                        help="Force recomputation even if output files exist")
    
    return parser.parse_args()

def collect_model_features(
    model: SAENetworkWrapper,
    observations: np.ndarray,
    original_mu: np.ndarray,
    original_value: np.ndarray,
    source_layer: str = None,
    feature_idx: int = None
) -> Tuple[Dict[str, torch.Tensor], np.ndarray, np.ndarray]:
    """
    Collect features and outputs from the model with optional feature ablation.
    
    Args:
        model: Model with SAE hooks
        observations: Input observations
        original_mu: Original mu outputs
        original_value: Original value outputs
        source_layer: Layer to ablate (None for no ablation)
        feature_idx: Feature to ablate (None for no ablation)
    
    Returns:
        Tuple of (collected_features, mu_output, value_output)
    """
    # Always clear previously accumulated features and ablations
    model.clear_accumulated_features()
    model.clear_ablation()
    
    # Apply feature ablation if specified
    if source_layer is not None and feature_idx is not None:
        model.ablate_feature(source_layer, feature_idx)
    
    # Start accumulating features
    model.start_accumulating_features()
    
    # Process all data through the model (compute_model_outputs handles batching internally)
    mu, value, _ = compute_model_outputs(
        model, observations, original_mu, original_value
    )
    
    features = model.get_accumulated_features()
    
    # Stop accumulating features
    model.stop_accumulating_features()
    
    # Clean up ablation state
    model.clear_ablation()
    
    return features, mu, value


def main():
    args = parse_args()
    
    if args.sae_dir is None:
        args.sae_dir = os.path.join(args.checkpoint_dir, 'sae_models')
    
    if args.output_dir is None:
        args.output_dir = os.path.join(args.checkpoint_dir, 'sae_interventions', 'data')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading model from {args.model_fname}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(args.model_fname, args.checkpoint_dir).to(device)
    data = load_data(args.model_fname, args.checkpoint_dir)
    observations, original_mu, original_value = unwrap_data(data)
    
    sae_models = load_sae_models(args.sae_dir, args.model_name)
    
    if not sae_models:
        raise ValueError(f"No SAE models found in {args.sae_dir}")
    
    layer_names = get_layer_order(list(sae_models.keys()))
    
    if args.source_layers:
        source_layers = [layer for layer in layer_names if layer in args.source_layers]
        if not source_layers:
            raise ValueError(f"None of the specified source layers {args.source_layers} found in model")
    else:
        source_layers = layer_names
    
    print(f"Model layers in order: {layer_names}")
    print(f"Source layers to analyze: {source_layers}")
    
    sae_wrapped_model = wrap_model_with_sae(
        model=copy.deepcopy(model),
        sae_models=sae_models,
        layer_names=layer_names
    )
    
    base_path = os.path.join(args.output_dir, f"{args.model_name}_sae_features.pkl")
    if os.path.exists(base_path) and not args.force:
        print(f"Output file for baseline features already exists. Skipping.")
    else:
        # Collect baseline features once for all layers
        print("Collecting baseline features (no ablation)...")
        baseline_features, baseline_mu, baseline_value = collect_model_features(
            model=sae_wrapped_model,
            observations=observations,
            original_mu=original_mu,
            original_value=original_value
        )
        assert_outputs_match(baseline_mu, baseline_value, original_mu, original_value)
        
        with open(base_path, 'wb') as f:
            pickle.dump(baseline_features, f)
    
    # For each source layer, process feature ablations
    for source_layer in source_layers:
        # Create a directory for this layer's results
        layer_dir = os.path.join(args.output_dir, source_layer)
        os.makedirs(layer_dir, exist_ok=True)
        
        print(f"\nAnalyzing effects from {source_layer}")
        source_features = sae_models[source_layer].hidden_dim
        
        # Process each active feature
        for feature_idx in tqdm(range(source_features), desc=f"Ablating features in {source_layer}"):
            # Check if output file already exists (skip if not forced)
            feature_output_path = os.path.join(layer_dir, f"feature_{feature_idx}.pkl")
            if os.path.exists(feature_output_path) and not args.force:
                continue
                
            # Process this feature ablation
            ablated_features, ablated_mu, ablated_value = collect_model_features(
                model=sae_wrapped_model,
                observations=observations,
                original_mu=original_mu,
                original_value=original_value,
                source_layer=source_layer,
                feature_idx=feature_idx
            )
            
            # Store results
            ablation_result = {
                "features": ablated_features,
                "mu": ablated_mu,
                "value": ablated_value
            }
            
            # Save this feature's results to its own file
            with open(feature_output_path, 'wb') as f:
                pickle.dump(ablation_result, f)
        
        print(f"Saved results for {source_layer} to {layer_dir}/")
    
    print(f"Done! To analyze these results, run analyze_sae_interventions.py")

if __name__ == "__main__":
    main()
