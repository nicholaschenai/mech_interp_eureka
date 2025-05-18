from typing import Dict, Any, Optional

from utils import joint_names, get_observation_labels, parse_joint_name

from ..neuron_properties.data_structures import NodeProperties


class CircuitStyler:
    """
    Handles styling and visual attributes for neural circuit visualization.
    
    Special neurons include those 
    - with high correlations (both positive and negative) with inputs/outputs
    - those which are phase selective ('approaching', 'opening', 'deceleration'). 
    These will be labelled and colored differently, and are not mutually exclusive.

    Manages:
    - Node colors and sizes
    - Edge colors and weights
    - Labels and tooltips
    """
    
    def __init__(self):
        self.obs_labels = get_observation_labels()

    def _input_output_styling(self, layer_name: str, neuron_idx: int, style_update: dict):
        """special styles for input and output layers"""
        if layer_name == 'obs' or layer_name is None:
            # Input layer: observations
            # Set color based on observation type
            obs_label = self.obs_labels[neuron_idx]
            obs_color = "lightblue" # joint position
            if "drawer" in obs_label:
                obs_color = "orange"  # Drawer observations
            elif "distance" in obs_label:
                obs_color = "lightyellow"  # Target vector
            elif "velocity" in obs_label:
                obs_color = "lightgreen"  # Joint velocities
            
            style_update.update({
                "color": obs_color,
                "size": 1000,
                "_base_label": obs_label  # Store base label without neuron_idx
            })
        elif layer_name == 'mu':
            # Output layer: joint actions
            style_update.update({
                "color": "lightgreen",
                "size": 1000,
                "_base_label": joint_names[neuron_idx]  # Store base label without neuron_idx
            })
    
    def _get_correlation_color(self, correlation: float) -> str:
        # Convert strongest correlation to RGB color using heatmap scheme
        # corr ranges from -1 to 1, map to RGB:
        # -1 -> (1,0,0) red
        #  0 -> (1,1,1) white
        # +1 -> (0,0,1) blue
        r = max(0, min(1, 1 - correlation))  # 1 at -1, 0 at +1
        g = max(0, min(1, 1 - abs(correlation)))  # 1 at 0, 0 at ±1
        b = max(0, min(1, 1 + correlation))  # 0 at -1, 1 at +1
        
        # Convert to hex color and set node color
        color = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
        return color
        
    def _special_neuron_styling(
            self, 
            layer_name: str, 
            neuron_idx: int, 
            neuron_data: Optional[NodeProperties], 
            style_update: dict
        ):
        """special styles for neurons with high correlations or phase selectivity"""
        if not neuron_data:
            return
            
        labels = []
        
        if neuron_data.correlations:
            strongest_corr = neuron_data.get_strongest_correlation()
            
            color = self._get_correlation_color(strongest_corr.correlation)
            style_update["color"] = color
            
            # Add all correlations to labels
            for corr in neuron_data.correlations:
                corr_feat_name = parse_joint_name(corr.feature_name)
                labels.append(f"{corr_feat_name} corr: {corr.correlation:.2f}")
                # I want size to keep increasing with each special property. DO NOT edit this!
                style_update["size"] = style_update.get("size", 300) + 100
            
            style_update["linewidth"] = 2
        
        if neuron_data.phase_selectivities:
            strongest_phase = neuron_data.get_strongest_phase()
        
            phase_shapes = {
                "approaching": "d",  # diamond
                "opening": "^",      # triangle up
                "deceleration": "s", # square
            }
            style_update["shape"] = phase_shapes[strongest_phase.phase]
            
            # Add phase selectivities to labels
            for phase in neuron_data.phase_selectivities:
                labels.append(f"Phase: {phase.phase} ({phase.selectivity:.2f})")
                # I want size to keep increasing with each special property. DO NOT edit this!
                style_update["size"] = style_update.get("size", 300) + 100
            
            style_update["linewidth"] = 2
        
        # Store labels for later concatenation
        if labels:
            style_update["_special_labels"] = labels

    def get_node_styling(
            self, 
            layer_name: str, 
            neuron_idx: int, 
            neuron_data: Optional[NodeProperties]
        ) -> Dict[str, Any]:
        """
        Determine styling for a neuron node based on its labels.
        
        Args:
            layer_name: Name of the layer
            neuron_idx: Index of the neuron
            neuron_data: NodeProperties object containing special info about the neuron
            
        Returns:
            Dict with styling parameters (color, size, shape, etc.)
        """
        style_update = {}

        self._input_output_styling(layer_name, neuron_idx, style_update)
        self._special_neuron_styling(layer_name, neuron_idx, neuron_data, style_update)
        
        # Build final label by stacking components
        label_parts = [str(neuron_idx)]  # Start with neuron index
        
        # Add base label if it exists
        if "_base_label" in style_update:
            label_parts.append(style_update.pop("_base_label"))
        
        # Add special labels if they exist
        if "_special_labels" in style_update:
            label_parts.extend(style_update.pop("_special_labels"))
        
        # Join all parts with newlines
        style_update["label"] = "\n".join(label_parts)
        
        return style_update
