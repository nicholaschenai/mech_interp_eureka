"""
Plot correlations between input features and layer 1 neurons for all models
So we can compare qualitatively between models, e.g. what are they missing
"""
import os
import argparse

import numpy as np

from config import DEFAULT_OUTPUT_DIR, DEFAULT_CHECKPOINT_DIR

from src.correlation.correlation_analyzer import CorrelationAnalyzer
from src.correlation.correlation_plots import plot_model_comparison_sensitivities

from utils.activation_dataset_utils import load_model_datasets


def main(args):
    # Create output directory if it doesn't exist
    output_dir = os.path.join(args.output_dir, 'correlations')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load datasets for all models
    model_datasets = load_model_datasets(args.ckpt_dir)
    
    if not model_datasets:
        print("No model datasets found. Make sure you've run collect_activations.py first.")
        return
    
    # Create correlation analyzers for each model
    analyzers = {model_type: CorrelationAnalyzer(dataset) 
                for model_type, dataset in model_datasets.items()}
    
    # Define the key input features based on the Eureka analysis
    # Hardcoded as per Eureka clues - these are the critical features
    # that differentiate the models according to reward function analysis
    key_features = ['distance', 'drawer_position', 'drawer_velocity']
    
    # Save path for the comparison figure
    comparison_fig_path = os.path.join(output_dir, 'model_comparison_sensitivities.png')
    
    # Create the comparison visualization
    print("Generating model comparison visualization...")
    results = plot_model_comparison_sensitivities(
        analyzers,
        input_features=key_features,
        layer_name='actor_mlp_1',
        figsize=(12, 15),
        colors={'strong': 'green', 'medium': 'blue', 'weak': 'red'},
        save_path=comparison_fig_path
    )
    print(f"Saved comparison visualization to {comparison_fig_path}")
    
    # Print some basic statistics about the sensitivities
    print("\nSensitivity Statistics:")
    for feature_name in key_features:
        print(f"\n{feature_name.upper()}:")
        for model_type in results:
            sensitivities = results[model_type][feature_name]
            print(f"  {model_type.capitalize()} model:")
            print(f"    Mean: {np.mean(sensitivities):.4f}")
            print(f"    Max: {np.max(sensitivities):.4f}")
            print(f"    Top 5 neuron indices: {np.argsort(sensitivities)[-5:]}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare correlations between input features and neurons across models")
    parser.add_argument('--ckpt_dir', type=str, default=DEFAULT_CHECKPOINT_DIR,
                        help='Directory containing activation datasets')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='Directory to save output visualizations')

    args = parser.parse_args()
    main(args)
