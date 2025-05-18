from typing import Dict, Optional, Tuple, Any
from collections import defaultdict

from .data_structures import NodeProperties, Correlation, PhaseSelectivity


class NeuronPropertyManager:
    def __init__(self, correlation_threshold: float = 0.8):
        self.correlation_threshold = correlation_threshold
        # (layer_name, neuron_idx) -> NodeProperties
        self.nodes: Dict[Tuple[str, int], NodeProperties] = defaultdict(NodeProperties)
    
    def add_correlation(
        self, 
        layer_name: str, 
        neuron_idx: int, 
        feature_name: str, 
        correlation: float, 
    ):
        if self.correlation_threshold is not None:
            if abs(correlation) < self.correlation_threshold:
                return
        node = self.nodes[(layer_name, neuron_idx)]
        node.add_correlation(Correlation(feature_name, correlation))
    
    def add_phase_selectivity(
        self, 
        layer_name: str, 
        neuron_idx: int, 
        phase: str, 
        selectivity: float, 
        mode: str
    ):
        """
        Args:
            layer_name: Name of the layer
            neuron_idx: Index of the neuron
            phase: Phase type
            selectivity: Selectivity value
            mode: Selectivity mode
        """
        node = self.nodes[(layer_name, neuron_idx)]
        node.add_phase_selectivity(PhaseSelectivity(phase, selectivity, mode))
    
    def get_node_properties(self, layer_name: str, neuron_idx: int) -> Optional[NodeProperties]:
        """
        Returns:
            NodeProperties object or None if node doesn't exist
        """
        return self.nodes.get((layer_name, neuron_idx))
    
    def get_layer_properties(self, layer_name: str) -> Dict[int, NodeProperties]:
        """
        Returns:
            Dictionary mapping neuron indices to NodeProperties
        """
        return {idx: props for (l, idx), props in self.nodes.items() if l == layer_name}
    
    def get_layer_summary(self, layer_name: str) -> Dict[str, Any]:
        """
        Returns:
            Dictionary with layer property summaries
        """
        layer_props = self.get_layer_properties(layer_name)
        if not layer_props:
            return {"layer_name": layer_name, "neuron_count": 0}
        
        # Count neurons with properties
        neurons_with_props = sum(1 for props in layer_props.values() if props.has_properties())
        
        # Count neurons with each type of property
        neurons_with_corr = sum(1 for props in layer_props.values() if props.correlations)
        neurons_with_phase = sum(1 for props in layer_props.values() if props.phase_selectivities)
        
        return {
            "layer_name": layer_name,
            "neuron_count": len(layer_props),
            "neurons_with_properties": neurons_with_props,
            "neurons_with_correlations": neurons_with_corr,
            "neurons_with_phases": neurons_with_phase
        }
