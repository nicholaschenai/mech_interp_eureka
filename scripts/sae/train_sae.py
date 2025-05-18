#!/usr/bin/env python3
"""
Train Sparse Autoencoder (SAE) models on neural network activations.

This script is the first step in the mechanistic interpretability workflow.
It trains SAE models on activation data from specific layers of a neural network,
with a focus on extracting interpretable features from the robot arm control model.

The trained models can then be used for:
1. Feature visualization (extracting and visualizing decoder weights)
2. Feature activation analysis across different task phases
3. Correlation of features with input observations (distance vectors, joint positions, etc.)

By default, this focuses on the middle activation layer (actor_mlp_3) which typically
contains the most interesting and interpretable features.

Next steps after running this script:
- Extract features from the trained models using an extraction script
- Visualize the features using a notebook or visualization tools
"""
import os
import sys
import argparse
import pickle
import torch

from pathlib import Path
from typing import Dict, Any, Tuple

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.activation_dataset import ActivationDataset
from src.sae.sae_model import SAEModel
from src.sae.sae_trainer import SAETrainer, constant_lr
from src.sae.utils import SAE_LAYERS_DEFAULT


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the SAE training script.
    
    Returns:
        Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(description="Train SAE models on neural network activations")
    parser.add_argument("--dataset_path", type=str, default='./ckpts/2025-02-13_09-26-08/strong_activation_dataset.pkl', help="Path to activation dataset")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save trained models")
    parser.add_argument("--layers", type=str, nargs="+", default=SAE_LAYERS_DEFAULT, 
                        help="Layers to train SAEs for (default: actor_mlp_3, the middle activation layer)")
    parser.add_argument("--hidden_dim", type=int, default=None, 
                        help="Hidden dimension for SAE (default: 2x input dimension)")
    parser.add_argument("--l1_coef", type=float, default=0.4, 
                        help="L1 regularization coefficient")
    parser.add_argument("--learning_rate", type=float, default=1e-3, 
                        help="Learning rate (default: 1e-3)")
    parser.add_argument("--batch_size", type=int, default=256, 
                        help="Batch size for training (default: 256)")
    parser.add_argument("--epochs", type=int, default=200, 
                        help="Number of training epochs")
    parser.add_argument("--tied_weights", action="store_true",
                        help="Whether to use tied weights (decoder = encoder.T)")
    parser.add_argument("--weight_normalize_eps", type=float, default=1e-8,
                        help="Small constant for numerical stability in weight normalization")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use for training (default: automatically detect)")
    parser.add_argument("--log_freq", type=int, default=10,
                        help="Frequency of logging during training (default: 10)")
    parser.add_argument("--model_name", type=str, default="strong", 
                        help="Name of the model (e.g., strong, medium, weak) to use in saved filenames")
    return parser.parse_args()


def train_sae_for_layer(dataset: ActivationDataset, layer_name: str, args: argparse.Namespace) -> Tuple[SAEModel, Dict[str, Any]]:
    """
    Args:
        dataset: ActivationDataset containing network activations
        layer_name: Name of the layer to analyze (e.g., actor_mlp_3)
        args: Command-line arguments with training parameters
        
    Returns:
        Tuple of (trained SAEModel instance, training metrics)
    """
    print(f"Training SAE for layer {layer_name}")
    
    activations_np = dataset.get_activation_matrix(layer_name)
    
    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    activations = torch.tensor(activations_np, dtype=torch.float32, device=device)
    
    hidden_dim = args.hidden_dim
    if hidden_dim is None:
        hidden_dim = activations.shape[1] * 2
        print(f"Setting hidden dimension to {hidden_dim} (2x input dimension)")
    
    trainer = SAETrainer(
        hidden_dim=hidden_dim,
        l1_coef=args.l1_coef,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        tied_weights=args.tied_weights,
        weight_normalize_eps=args.weight_normalize_eps
    )

    sae_model, training_info = trainer.train(
        activations, 
        epochs=args.epochs,
        log_freq=args.log_freq,
        lr_scale=constant_lr,
    )
    
    return sae_model, training_info


def main():
    """
    1. Loads activation data from a provided dataset
    2. Trains SAE models for specified layers (defaults to actor_mlp_3)
    3. Saves the trained models for later analysis
    
    The output models will be saved as pickle files that can be loaded
    for feature extraction and visualization.
    """
    args = parse_args()
    
    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.dataset_path), 'sae_models')

    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading activation dataset from {args.dataset_path}")
    dataset = ActivationDataset()
    dataset.load(args.dataset_path)
    
    # Train SAEs for each specified layer
    for layer_name in args.layers:        
        print(f"\n{'='*50}")
        print(f"Training SAE for layer: {layer_name}")
        print(f"{'='*50}\n")
        
        sae_model, training_info = train_sae_for_layer(dataset, layer_name, args)
        
        # Create base path for all files related to this layer
        base_path = os.path.join(args.output_dir, f"{args.model_name}_sae_{layer_name}")
        
        # Save model
        model_path = f"{base_path}.pkl"
        sae_model.save(model_path)
        print(f"Saved trained model to {model_path}")
        
        data_log = training_info['data_log']
        log_path = f"{base_path}_log.pkl"
        with open(log_path, 'wb') as f:
            pickle.dump(data_log, f)
        print(f"Saved training log to {log_path}")

        final_metrics = training_info['final_metrics']
        metrics_path = f"{base_path}_final_metrics.pkl"
        with open(metrics_path, 'wb') as f:
            pickle.dump(final_metrics, f)
        print(f"Saved training metrics to {metrics_path}")

        print(f"\nTraining summary for {layer_name}:")
        print(f"  Reconstruction loss: {final_metrics['reconstruction_loss']:.6f}")
        print(f"  Sparsity: {final_metrics['sparsity']:.6f}")
        print(f"  Dead features: {final_metrics['dead_features']} ({final_metrics['dead_features_percent']:.2f}%)")
        print(f"  Mean active features per sample: {final_metrics['l0_sparsity']:.2f} ({final_metrics['l0_sparsity_percent']:.2f}%)")
    
    print("\n" + "="*50)
    print("Training complete.")
    print("="*50)


if __name__ == "__main__":
    main()
