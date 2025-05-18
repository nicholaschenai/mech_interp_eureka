"""
Test script for circuit visualization.
just visualize one example and save
"""
import os
import pickle
import matplotlib.pyplot as plt

from config import DEFAULT_OUTPUT_DIR

import src.neuron_properties.loaders as loaders
from src.circuits.circuit_visualizer import CircuitVisualizer
from src.neuron_properties.manager import NeuronPropertyManager


def load_circuit(file_path: str):
    """Load circuit data from pickle file."""
    with open(file_path, "rb") as f:
        return pickle.load(f)


def load_correlations_and_test(correlation_file: str, manager: NeuronPropertyManager):
    """Load correlations into manager and print summary."""
    loaders.load_correlations_into_manager(correlation_file, manager)
    
    # Print summary for each layer
    print("\nLayer Summaries:")
    print("-" * 50)
    for layer_name in ["actor_mlp_0", "actor_mlp_1", "actor_mlp_2"]:
        summary = manager.get_layer_summary(layer_name)
        print(f"\nLayer: {summary['layer_name']}")
        print(f"Total neurons: {summary['neuron_count']}")
        print(f"Neurons with correlations: {summary['neurons_with_correlations']}")
    
    # Print detailed info for a few example neurons
    print("\nExample Neuron Correlations:")
    print("-" * 50)
    test_neurons = [
        ("actor_mlp_0", 96),
        ("actor_mlp_1", 89),
        ("actor_mlp_2", 30),
    ]
    
    for layer_name, neuron_idx in test_neurons:
        props = manager.get_node_properties(layer_name, neuron_idx)
        if props and props.correlations:
            print(f"\nNeuron {layer_name}[{neuron_idx}]:")
            print("Correlations:")
            for corr in props.correlations:
                print(f"  - {corr.feature_name}: {corr.correlation:.3f}")
        else:
            print(f"\nNo correlations found for {layer_name}[{neuron_idx}]")


def load_phase_selectivity_and_test(phase_file: str, manager: NeuronPropertyManager):
    """Load phase selectivity into manager and print summary."""
    loaders.load_phase_selectivity_into_manager(phase_file, manager)
    
    # Print summary for each layer
    print("\nPhase Selectivity Layer Summaries:")
    print("-" * 50)
    for layer_name in ["actor_mlp_0", "actor_mlp_1", "actor_mlp_2"]:
        summary = manager.get_layer_summary(layer_name)
        print(f"\nLayer: {summary['layer_name']}")
        print(f"Total neurons: {summary['neuron_count']}")
        print(f"Neurons with phase selectivity: {summary['neurons_with_phases']}")
    
    # Print detailed info for a few example neurons
    print("\nExample Neuron Phase Selectivity:")
    print("-" * 50)
    test_neurons = [
        ("actor_mlp_0", 96),
        ("actor_mlp_1", 89),
        ("actor_mlp_2", 30),
    ]
    
    for layer_name, neuron_idx in test_neurons:
        props = manager.get_node_properties(layer_name, neuron_idx)
        if props and props.phase_selectivities:
            print(f"\nNeuron {layer_name}[{neuron_idx}]:")
            print("Phase Selectivities:")
            for phase_selectivity in props.phase_selectivities:
                print(f"  - {phase_selectivity.phase}: {phase_selectivity.selectivity:.3f}")
        else:
            print(f"\nNo phase selectivity found for {layer_name}[{neuron_idx}]")


def main():
    # Load a circuit (using mu_0_positive as an example)
    circuit_file = "results/circuits/mu_0_positive.pkl"
    circuit_data = load_circuit(circuit_file)
    
    manager = NeuronPropertyManager()
    
    correlation_file = f"{DEFAULT_OUTPUT_DIR}/correlations/strong_feature_neuron_corr.pkl"
    load_correlations_and_test(correlation_file, manager)
    
    phase_file = f"{DEFAULT_OUTPUT_DIR}/phase/strong_phase_selectivity_offset.pkl"
    load_phase_selectivity_and_test(phase_file, manager)
    
    visualizer = CircuitVisualizer(manager)
    
    # Create and save the figure using the saved data format directly
    # fig = visualizer.visualize_circuit(circuit_data, use_styling=False)
    fig = visualizer.visualize_circuit(circuit_data, use_styling=True)
    
    output_dir = "results/circuits"
    os.makedirs(output_dir, exist_ok=True)
    
    layer_name = circuit_data['layer']
    neuron_idx = circuit_data['neuron_idx']
    polarity = circuit_data['polarity']
    
    # # Save with descriptive filename
    output_file = f"{output_dir}/{layer_name}_{neuron_idx}_{polarity}.png"
    fig.savefig(output_file, dpi=100, bbox_inches="tight")
    plt.close(fig)
    
    print(f"Saved circuit visualization to {output_file}")


if __name__ == "__main__":
    main()
