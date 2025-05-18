import matplotlib.pyplot as plt
import networkx as nx

from typing import Dict, Tuple, Optional

from ..neuron_properties.manager import NeuronPropertyManager

from .circuit_styler import CircuitStyler


class CircuitVisualizer:
    """
    Class for visualizing traced circuits with special handling for labeled neurons.
    """
    def __init__(self, neuron_property_manager: Optional[NeuronPropertyManager] = None):
        self.data_manager = neuron_property_manager or NeuronPropertyManager()
        self.styler = CircuitStyler()
    
    def visualize_circuit(
        self,
        saved_data: Dict,
        figsize: Tuple[int, int] = (20, 10),
        use_styling: bool = False
    ) -> plt.Figure:
        """
        Visualize circuit directly from saved data format.
        
        Args:
            saved_data: Dict containing circuit data in saved format:
                {
                    'nodes': {(layer_name, neuron_idx): activation},
                    'edges': {(src_node, target_node): contribution},
                    'layer': str,
                    'neuron_idx': int,
                    'polarity': str
                }
            figsize: Size of the figure
            use_styling: Whether to apply special styling based on neuron labels
            
        Returns:
            Matplotlib figure
        """
        G = nx.DiGraph()
        
        # Add nodes in reverse order so layer is right
        for (layer_name, neuron_idx), activation in reversed(saved_data['nodes'].items()):
            # Node identifier
            node_id = f"{layer_name}_{neuron_idx}"
            
            node_attrs = {
                "layer": layer_name,
                "neuron_idx": neuron_idx,
                "activation": activation
            }
            
            # default styling
            node_style = {
                "color": "yellow",
                "size": 300,
                "shape": "o",
                "alpha": 0.7,
                "linewidth": 1,
                "label": f"{neuron_idx}"
            }
            
            if use_styling:
                neuron_data = self.data_manager.get_node_properties(layer_name, neuron_idx)
                style_update = self.styler.get_node_styling(layer_name, neuron_idx, neuron_data)
                node_style.update(style_update)

            node_attrs.update(node_style)
            
            G.add_node(node_id, **node_attrs)
        
        for (src_node, target_node), contribution in saved_data['edges'].items():
            src_layer, src_idx = src_node
            target_layer, target_idx = target_node
            
            src_id = f"{src_layer}_{src_idx}"
            target_id = f"{target_layer}_{target_idx}"
            
            G.add_edge(
                src_id,
                target_id,
                weight=abs(contribution),
                sign=1 if contribution > 0 else -1
            )
        
        return self._draw_graph(G, figsize, use_styling)
    
    def _draw_graph(
        self,
        G: nx.DiGraph,
        figsize: Tuple[int, int],
        use_styling: bool
    ) -> plt.Figure:
        # Calculate nodes per layer
        layer_nodes = {}
        for node, attrs in G.nodes(data=True):
            layer = attrs['layer']
            if layer not in layer_nodes:
                layer_nodes[layer] = []
            layer_nodes[layer].append(node)
        
        # Get maximum nodes in any layer
        max_nodes = max(len(nodes) for nodes in layer_nodes.values())
        
        # Adjust figure height based on number of nodes
        adjusted_height = 2 + max_nodes * 0.8
        figsize = (figsize[0], adjusted_height)
        
        # Create figure with adjusted size
        fig, ax = plt.subplots(figsize=figsize)
        
        # Layout the graph by layer
        pos = nx.multipartite_layout(G, subset_key="layer")
        
        # Adjust vertical positions based on layer size
        for layer, nodes in layer_nodes.items():
            # Sort nodes by neuron index for consistent ordering
            nodes.sort(key=lambda n: -G.nodes[n]['neuron_idx'])
            
            # Calculate vertical spacing
            total_height = 1
            spacing = total_height / (len(nodes) + 1)
            
            # Adjust y positions
            for i, node in enumerate(nodes):
                pos[node] = (pos[node][0], -0.4 + (i + 1) * spacing)
        
        # Extract node attributes for visualization
        # node_colors = [G.nodes[n]["color"] for n in G.nodes]
        # node_sizes = [G.nodes[n]["size"] for n in G.nodes]
        node_labels = {n: G.nodes[n].get("label", "") for n in G.nodes}
        
        # Extract edge attributes
        edge_weights = [G.edges[e]["weight"] * 2 for e in G.edges]
        edge_colors = ["green" if G.edges[e]["sign"] > 0 else "red" for e in G.edges]
        
        # Draw the network
        # Group nodes by shape
        nodes_by_shape = {}
        for node in G.nodes():
            shape = G.nodes[node].get('shape', 'o')  # default to circle
            if shape not in nodes_by_shape:
                nodes_by_shape[shape] = []
            nodes_by_shape[shape].append(node)
        
        # Draw nodes for each shape
        for shape, nodes in nodes_by_shape.items():
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=nodes,
                node_color=[G.nodes[n]["color"] for n in nodes],
                node_size=[G.nodes[n]["size"] for n in nodes],
                node_shape=shape,
                alpha=0.8,
                ax=ax
            )
        
        nx.draw_networkx_edges(
            G, pos,
            width=edge_weights,
            edge_color=edge_colors,
            alpha=0.6,
            arrows=True,
            arrowsize=10,
            ax=ax
        )
        
        nx.draw_networkx_labels(
            G, pos,
            labels=node_labels,
            font_size=8,
            ax=ax
        )
        
        # Add layer labels at the bottom of fig
        for layer, nodes in layer_nodes.items():
            x_pos = pos[nodes[0]][0]  # Use x position of first node in layer
            layer_label = layer or 'obs'
            ax.text(x_pos, -0.45, layer_label, 
                   horizontalalignment='center',
                   verticalalignment='top',
                   fontsize=10,
                   fontweight='bold')
        
        ax.set_title("Neural Circuit Visualization", fontsize=14)
        
        plt.tight_layout()
        return fig
