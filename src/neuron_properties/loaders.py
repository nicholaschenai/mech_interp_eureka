"""
Functions for loading neuron property data from files.
"""
import pickle

from .manager import NeuronPropertyManager


def load_correlations_into_manager(file_path: str, manager: NeuronPropertyManager):
    """
    Load correlation data from a pickle file and add it to a manager.
    
    Args:
        file_path: Path to the correlation data file
        manager:  manager to add correlations to.
    """
    with open(file_path, 'rb') as f:
        raw_data = pickle.load(f)
    
    for layer_name, layer_data in raw_data.items():
        for feature_name, neuron_data in layer_data.items():
            for neuron_idx, correlation in neuron_data:
                manager.add_correlation(
                    layer_name,
                    neuron_idx,
                    feature_name,
                    correlation,
                )

def load_phase_selectivity_into_manager(file_path: str, manager: NeuronPropertyManager):
    """
    Load phase selectivity data from a pickle file and add it to a manager.
    
    Args:
        file_path: Path to the phase selectivity data file
        manager: Manager to add phase selectivities to
    """
    with open(file_path, 'rb') as f:
        raw_data = pickle.load(f)
    
    for layer_name, layer_data in raw_data.items():
        for phase_name, neuron_data in layer_data.items():
            for neuron_idx, selectivity in neuron_data.items():
                manager.add_phase_selectivity(
                    layer_name,
                    neuron_idx,
                    phase_name,
                    selectivity,
                    mode="offset"
                )
