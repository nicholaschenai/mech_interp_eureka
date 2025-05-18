"""
Save neuron labels (high correlation w input/output, phase selective) to files
"""
import os
import argparse
import pickle

from typing import Dict, List, Tuple
from config import DEFAULT_CHECKPOINT_DIR, DEFAULT_OUTPUT_DIR

from src.activation_dataset import ActivationDataset
from src.correlation.correlation_analyzer import CorrelationAnalyzer
from src.phase_analysis import PhaseAnalyzer
from src.correlation.correlation_print import pretty_print_correlated_neurons, pretty_print_phase_neurons

from utils.base_feature_extractor import get_all_feature_keys
from utils.activation_dataset_utils import load_model_dataset


def compute_correlations(
    dataset: ActivationDataset,
    layer_name: str,
    feature_keys: list,
    correlation_threshold: float = 0.7
) -> Dict[str, List[Tuple[int, float]]]:
    """
    Args:
        feature_keys: List of features to analyze

    Returns:
        Dictionary mapping features to neuron correlations {feature: [(neuron_idx, correlation_score), ...]}
    """
    correlation_analyzer = CorrelationAnalyzer(dataset)
    layer_correlations = pretty_print_correlated_neurons(
        correlation_analyzer,
        layer_name,
        feature_keys,
        correlation_threshold
    )
    
    return layer_correlations


def compute_phase_selectivity(
    dataset: ActivationDataset,
    layer_name: str,
    selectivity_threshold: float = 2.0,
    mode: str = "offset"
) -> dict:
    """
    Returns:
        Dictionary mapping phases to selective neurons {phase: {neuron_idx: selectivity_score}}
    """
    phase_analyzer = PhaseAnalyzer(dataset)
    phase_neurons = phase_analyzer.identify_all_phase_neurons(
        layer_name,
        selectivity_threshold=selectivity_threshold,
        mode=mode
    )
    pretty_print_phase_neurons(phase_neurons)
    # Remove total_neurons key if it exists
    if 'total_neurons' in phase_neurons:
        del phase_neurons['total_neurons']
    return phase_neurons


def save_property_results(results: dict, output_path: str):
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved results to {output_path}")


def save_neuron_labels(
    dataset_path: str,
    main_output_dir: str,
    model_name: str = 'strong',
    correlation_threshold: float = 0.7,
    phase_selectivity_threshold: float = 2.0
):
    """
    Save neuron labels for correlations and phase selectivity
    
    Args:
        dataset_path: Path to activation dataset
        main_output_dir: Directory to save labels
        model_name: Name of the model to analyze (default: 'strong')
        correlation_threshold: Threshold for correlation analysis
        phase_selectivity_threshold: Threshold for phase selectivity analysis
    """
    # Create output directories
    correlation_dir = os.path.join(main_output_dir, 'correlations')
    phase_dir = os.path.join(main_output_dir, 'phase')
    os.makedirs(correlation_dir, exist_ok=True)
    os.makedirs(phase_dir, exist_ok=True)
    
    dataset = load_model_dataset(dataset_path, model_name)
    
    layer_names = list(dataset.activations.keys())
    feature_keys = get_all_feature_keys()
    print(f"Found {len(layer_names)} layers: {layer_names}")
    print(f"Found {len(feature_keys)} features: {feature_keys}")
    
    correlation_results = {}
    phase_results_offset = {}
    phase_results_absolute = {}
    
    for layer_name in layer_names:
        print(f"\nProcessing layer: {layer_name}")
        
        print("Computing correlations...")
        correlation_results[layer_name] = compute_correlations(
            dataset, layer_name, feature_keys, correlation_threshold
        )
        
        # Compute phase selectivity with offset mode (original method)
        print("Computing phase selectivity (offset mode)...")
        phase_results_offset[layer_name] = compute_phase_selectivity(
            dataset, layer_name, phase_selectivity_threshold, mode="offset"
        )
        
        # Compute phase selectivity with absolute mode
        print("Computing phase selectivity (absolute mode)...")
        phase_results_absolute[layer_name] = compute_phase_selectivity(
            dataset, layer_name, phase_selectivity_threshold, mode="absolute"
        )
    
    # Save results with model name in filename
    print("\nSaving results...")
    save_property_results(
        correlation_results,
        os.path.join(correlation_dir, f'{model_name}_feature_neuron_corr.pkl')
    )
    
    # Save phase selectivity results for both modes
    save_property_results(
        phase_results_offset,
        os.path.join(phase_dir, f'{model_name}_phase_selectivity_offset.pkl')
    )
    
    save_property_results(
        phase_results_absolute,
        os.path.join(phase_dir, f'{model_name}_phase_selectivity_absolute.pkl')
    )


def main():
    parser = argparse.ArgumentParser(description="Save neuron labels for correlations and phase selectivity")
    parser.add_argument('--dataset_path', type=str, default=DEFAULT_CHECKPOINT_DIR,
                      help='Path to activation dataset')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                      help='Directory to save labels')
    parser.add_argument('--model_name', type=str, default='strong',
                      choices=['strong', 'medium', 'weak'],
                      help='Name of the model to analyze')
    parser.add_argument('--correlation_threshold', type=float, default=0.7,
                      help='Threshold for correlation analysis')
    parser.add_argument('--phase_selectivity_threshold', type=float, default=2.0,
                      help='Threshold for phase selectivity analysis')
    
    args = parser.parse_args()
    save_neuron_labels(
        args.dataset_path,
        args.output_dir,
        args.model_name,
        args.correlation_threshold,
        args.phase_selectivity_threshold
    )

if __name__ == "__main__":
    main()
