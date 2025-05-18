"""
Phase Analysis Module

Identifies neurons that selectively activate during specific phases of computation.
Supports multiple analysis methods:
- offset: Original method offsetting activations to be positive
- absolute: Uses absolute activation values to find neurons with strongest magnitude regardless of sign
"""
import numpy as np

from typing import Dict, List, Tuple, Union

from .activation_dataset import ActivationDataset


class PhaseAnalyzer:
    def __init__(self, dataset: ActivationDataset):
        self.dataset = dataset

    def identify_all_phase_neurons(self,
                                layer_name: str,
                                threshold: Union[float, None] = None,
                                selectivity_threshold: float = 2.0,
                                mode: str = "offset") -> Dict[str, Dict[int, float]]:
        """
        Find neurons that are selectively active for each phase.
        
        Args:
            threshold: Activation threshold for considering a neuron "active"
            selectivity_threshold: Minimum ratio of in-phase/out-of-phase activation
            mode: One of "offset" (original method), "absolute" (uses absolute activation values)
            
        Returns:
            Dictionary mapping phase names to dictionaries of selective neurons
            {phase_name: {neuron_index: selectivity_score, ...}, ...}
            For bidirectional mode, negative values indicate neurons inhibited during the phase
        """
        phase_names = list(self.dataset.metadata['phase_masks'].keys())
        if not phase_names:
            print("Warning: No phases found in dataset metadata")
            return {}
        
        phase_activations, global_min = self.dataset.get_phase_activations(layer_name)
        
        total_neurons = phase_activations[phase_names[0]].shape[1]
        phase_neurons = {'total_neurons': total_neurons}
        
        processed_activations = self._preprocess_activations(phase_activations, global_min, mode)
        phase_means = self._compute_phase_means(processed_activations)
        
        for target_phase in phase_names:
            other_mean = self._compute_other_phases_mean(processed_activations, phase_names, target_phase)
            
            selectivity_fn = self._calculate_positive_selectivity
            selective_indices, selective_values = selectivity_fn(
                phase_means[target_phase], 
                other_mean, 
                threshold, 
                selectivity_threshold
            )
            
            selective_neurons = self._format_selective_neurons(selective_indices, selective_values)
            phase_neurons[target_phase] = selective_neurons
        
        return phase_neurons
    
    def _preprocess_activations(self, 
                              phase_activations: Dict[str, np.ndarray], 
                              global_min: float,
                              mode: str) -> Dict[str, np.ndarray]:
        processed = {}
        if mode == "offset":
            for phase, activations in phase_activations.items():
                processed[phase] = activations - global_min
        elif mode == "absolute":
            for phase, activations in phase_activations.items():
                processed[phase] = np.abs(activations)
        else:
            processed = phase_activations.copy()
        
        return processed
    
    def _compute_phase_means(self, phase_activations: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        phase_means = {}
        for phase_name, activations in phase_activations.items():
            phase_means[phase_name] = np.mean(activations, axis=0)
        return phase_means
    
    def _compute_other_phases_mean(self, phase_activations: Dict[str, np.ndarray], 
                                 phase_names: List[str],
                                 target_phase: str) -> np.ndarray:
        other_phases = [p for p in phase_names if p != target_phase]
        other_activations = np.vstack([phase_activations[p] for p in other_phases])
        return np.mean(other_activations, axis=0)
    
    def _calculate_positive_selectivity(self, 
                                     target_mean: np.ndarray, 
                                     other_mean: np.ndarray,
                                     threshold: Union[float, None],
                                     selectivity_threshold: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate positive-only selectivity (original method).
        Returns indices and values of neurons that are more active in target phase.
        """
        epsilon = 0.01
        selectivity = target_mean / np.maximum(other_mean, epsilon)
        
        mask = selectivity >= selectivity_threshold
        if threshold is not None:
            mask = mask & (target_mean >= threshold)
        
        indices = np.where(mask)[0]
        values = selectivity[mask]
        
        return indices, values
    
    def _format_selective_neurons(self, indices: np.ndarray, values: np.ndarray) -> Dict[int, float]:
        selective_neurons = {int(idx): float(val) for idx, val in zip(indices, values)}
        return dict(sorted(selective_neurons.items(), key=lambda x: abs(x[1]), reverse=True))
