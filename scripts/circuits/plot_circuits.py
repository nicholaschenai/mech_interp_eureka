"""
Script to visualize all traced circuits with neuron property styling.
"""
import os
import pickle
import matplotlib.pyplot as plt

from src.circuits.circuit_visualizer import CircuitVisualizer
from src.neuron_properties.manager import NeuronPropertyManager
import src.neuron_properties.loaders as loaders

from config import DEFAULT_OUTPUT_DIR

model_name = 'strong'


def load_circuit(file_path: str):
    """Load circuit data from pickle file."""
    with open(file_path, "rb") as f:
        return pickle.load(f)


def visualize_circuit(
    visualizer: CircuitVisualizer,
    circuit_data: dict,
    output_dir: str
):
    """Visualize a single circuit and save the figure."""
    layer_name = circuit_data['layer']
    neuron_idx = circuit_data['neuron_idx']
    polarity = circuit_data['polarity']
    
    output_file = f"{output_dir}/{layer_name}_{neuron_idx}_{polarity}.png"
    # if os.path.exists(output_file):
    #     print(f"Skipping {output_file} because it already exists")
    #     return
    
    fig = visualizer.visualize_circuit(circuit_data, use_styling=True)
    
    fig.savefig(output_file, dpi=100, bbox_inches="tight")
    plt.close(fig)
    
    print(f"Saved circuit visualization to {output_file}")


def main():
    # Initialize property manager and load data
    manager = NeuronPropertyManager()
    # TODO: allow specification of model name
    # Load correlation and phase data
    correlation_file = f"{DEFAULT_OUTPUT_DIR}/correlations/{model_name}_feature_neuron_corr.pkl"
    loaders.load_correlations_into_manager(correlation_file, manager)
    
    phase_file = f"{DEFAULT_OUTPUT_DIR}/phase/{model_name}_phase_selectivity_offset.pkl"
    loaders.load_phase_selectivity_into_manager(phase_file, manager)
    
    # Initialize visualizer with property manager
    visualizer = CircuitVisualizer(manager)
    
    # Create output directory
    output_dir = f"{DEFAULT_OUTPUT_DIR}/circuits/{model_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and visualize all circuits
    circuits_dir = output_dir
    for filename in os.listdir(circuits_dir):
        if filename.endswith(".pkl"):
            circuit_file = os.path.join(circuits_dir, filename)
            circuit_data = load_circuit(circuit_file)
            visualize_circuit(visualizer, circuit_data, output_dir)
    
    print("Done!")


if __name__ == "__main__":
    main()
