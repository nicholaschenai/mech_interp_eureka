"""
Script to trace circuits for neurons in the 'mu' layer and save the results.
"""
import os
import pickle
from typing import Dict
from src.circuits.circuit_tracer import CircuitTracer

from config import MODEL_CONFIGS, DEFAULT_CHECKPOINT_DIR
from utils.model_utils import load_model
from utils.activation_dataset_utils import load_model_dataset

model_name = 'strong'

def save_circuit_data(
        circuit: Dict, 
        layer_name: str, 
        neuron_idx: int, 
        model_name: str, 
        output_dir="results/circuits"
    ):
    """Save extracted circuit data to JSON files."""
    os.makedirs(output_dir, exist_ok=True)
    
    for polarity, circuit_data in circuit.items():
        circuit_data.update({
            'polarity': polarity,
            'layer': layer_name,
            'neuron_idx': neuron_idx,
        })
        
        output_file = f"{output_dir}/{model_name}/{layer_name}_{neuron_idx}_{polarity}.pkl"
        with open(output_file, "wb") as f:
            pickle.dump(circuit_data, f)
        
        print(f"Saved circuit data to {output_file}")

def main():
    # Set up parameters
    layer_name = "mu"
    neuron_indices = range(9)  # 0-8
    
    model_file = MODEL_CONFIGS[model_name]['file']
    activation_dataset = load_model_dataset(DEFAULT_CHECKPOINT_DIR, model_name)
    model = load_model(model_file, DEFAULT_CHECKPOINT_DIR)
    
    tracer = CircuitTracer(model)
    
    # Trace circuits for each neuron
    for neuron_idx in neuron_indices:
        print(f"\n==== Tracing circuit for {layer_name} neuron {neuron_idx} ====")
        circuit = tracer.trace_circuit(layer_name, neuron_idx, activation_dataset)
        save_circuit_data(circuit, layer_name, neuron_idx, model_name)
        
if __name__ == "__main__":
    main()
