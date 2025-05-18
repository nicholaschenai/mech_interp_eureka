"""
This module traces activation pathways through a neural network to identify
which neurons and connections contribute most to a particular neuron's activation.
"""
import torch
import numpy as np

from typing import Dict, List, Tuple, Optional

from ..activation_dataset import ActivationDataset


class CircuitTracer:
    def __init__(self, model):
        """
        Args:
            model: The trained RL model with actor_mlp network
        """
        self.model = model
        self.layer_weights = self._extract_layer_weights()
        self.to_process = []
        
    def _extract_layer_weights(self) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Extract weights and biases from each layer of the actor_mlp, mu, and value layers.
        
        Returns:
            Dict mapping layer names to (weights, bias) tuples
        """
        weights = {}
        
        actor_mlp = self.model.a2c_network.actor_mlp if hasattr(self.model, 'a2c_network') else self.model.actor_mlp
        for name, module in actor_mlp.named_modules():
            if isinstance(module, torch.nn.Linear):
                weights[f"actor_mlp_{name}"] = (
                    module.weight.detach().cpu().numpy(),
                    module.bias.detach().cpu().numpy() if module.bias is not None else None
                )
        
        if hasattr(self.model, 'a2c_network'):
            if hasattr(self.model.a2c_network, 'mu'):
                weights['mu'] = (
                    self.model.a2c_network.mu.weight.detach().cpu().numpy(),
                    self.model.a2c_network.mu.bias.detach().cpu().numpy() if self.model.a2c_network.mu.bias is not None else None
                )
            if hasattr(self.model.a2c_network, 'value'):
                weights['value'] = (
                    self.model.a2c_network.value.weight.detach().cpu().numpy(),
                    self.model.a2c_network.value.bias.detach().cpu().numpy() if self.model.a2c_network.value.bias is not None else None
                )
                
        return weights
    
    def _get_previous_layer_name(self, layer_name: str) -> Optional[Tuple[str, bool]]:
        """
        Returns (previous_layer_name, is_activation_layer (for current layer))
        """
        if layer_name in ['mu', 'value']:
            return 'actor_mlp_5', False
        
        try:
            parts = layer_name.split('_')
            current_idx = int(parts[-1]) if parts[-1].isdigit() else None
            
            if current_idx is not None and current_idx > 0:
                return f"{'_'.join(parts[:-1])}_{current_idx - 1}", current_idx % 2 == 1
            else:
                return None, None
        except (ValueError, IndexError):
            return None, None
    
    def find_max_activating_examples(
        self, 
        layer_name: str, 
        neuron_idx: int, 
        activation_dataset: ActivationDataset, 
        top_k: int = 10,
        threshold_pct: float = 0.7
    ) -> Dict[str, List[Tuple[int, float]]]:
        """
        Find examples that maximally activate the neuron for each polarity.
        
        Returns:
            Dictionary with 'positive' and 'negative' keys, each containing
            a list of (example_idx, activation_value) tuples
        """
        layer_activations = activation_dataset.get_activation_matrix(layer_name)
        neuron_activations = layer_activations[:, neuron_idx]
        
        sorted_indices = np.argsort(neuron_activations)
        
        bottom_indices = sorted_indices[:top_k]
        top_indices = sorted_indices[-top_k:][::-1]
        
        # Filter top indices to keep only positive values
        positive_examples = []
        for idx in top_indices:
            val = neuron_activations[idx]
            if val > 0:  # Only keep if positive
                positive_examples.append((int(idx), float(val)))
        
        # Filter bottom indices to keep only negative values
        negative_examples = []
        for idx in bottom_indices:
            val = neuron_activations[idx]
            if val < 0:  # Only keep if negative
                negative_examples.append((int(idx), float(val)))
        
        # Apply threshold
        if positive_examples:
            max_pos_val = positive_examples[0][1]
            pos_threshold = threshold_pct * max_pos_val
            positive_examples = [(idx, val) for idx, val in positive_examples if val >= pos_threshold]
        
        if negative_examples:
            max_neg_val = negative_examples[0][1]
            neg_threshold = threshold_pct * max_neg_val  # Will be negative
            negative_examples = [(idx, val) for idx, val in negative_examples if val <= neg_threshold]
        
        return {
            'positive': positive_examples,
            'negative': negative_examples
        }

    def get_important_contributors(
        self, 
        current_layer: str, 
        input_vector: np.ndarray, 
        current_neuron_idx: int,
        current_activation: float,
        threshold: float = 0.5
    ) -> List[Tuple[int, float]]:
        weights, bias_vec = self.layer_weights.get(current_layer, (None, None))
        if weights is None:
            print(f'weights is None for {current_layer}')

        neuron_weights = weights[current_neuron_idx]
        contributions = neuron_weights * input_vector

        manual_sum = np.sum(contributions) 
        bias_val = 0
        if bias_vec is not None:
            bias_val = bias_vec[current_neuron_idx]
            manual_sum += bias_val

        if not np.allclose(manual_sum, current_activation):
            raise ValueError(f"Sum of contributions does not match activation: {manual_sum} != {current_activation}")
        
        total_contribution = current_activation
        # Avoid division by zero
        if total_contribution == 0:
            return []
        
        abs_contributions = np.abs(contributions)
        sorted_indices = np.argsort(abs_contributions)[::-1]
        
        important_contributors = []
        # start from bias value in case it carries a significant proportion of the activation
        cumulative_contribution = bias_val 
        
        for idx in sorted_indices:
            # Note: this works for both signs!
            if cumulative_contribution / total_contribution >= threshold:
                break
            
            important_contributors.append((int(idx), float(contributions[idx])))
            cumulative_contribution += contributions[idx]
                
        return important_contributors

    def trace_circuit(
        self, 
        layer_name: str, 
        neuron_idx: int, 
        activation_dataset: ActivationDataset,
        threshold: float = 0.5
    ) -> Dict:
        """
        Args:
            threshold: Contribution threshold for tracing
            
        Returns:
            Dict representation of the circuit graph
        """
        print(f"Tracing circuit for {layer_name} neuron {neuron_idx}...")
        activating_examples = self.find_max_activating_examples(
            layer_name, neuron_idx, activation_dataset
        )
        result = {}
        
        for polarity in ['positive', 'negative']:
            if not activating_examples[polarity]:
                continue
            
            example_idx, activation_value = activating_examples[polarity][0]
            print(f"Processing {polarity} circuit (example #{example_idx}, activation: {activation_value:.4f})...")
            polarity_result = {'examples': activating_examples[polarity], 'example_idx': example_idx}

            circuit_graph = CircuitGraph(polarity, example_idx)

            self.to_process = [(layer_name, neuron_idx, float(activation_value))]
            while self.to_process:
                current_layer, current_neuron_idx, current_activation = self.to_process.pop(0)
                current_node = (current_layer, current_neuron_idx)
                if current_node in circuit_graph.nodes:
                    continue

                circuit_graph.add_node(current_layer, current_neuron_idx, current_activation)

                if current_layer is None:
                    continue
                
                if circuit_graph.num_nodes() % 50 == 0:
                    print(f"  Processed {circuit_graph.num_nodes()} nodes, queue size: {len(self.to_process)}")
                    
                prev_layer_name, is_activation_layer = self._get_previous_layer_name(current_layer)
                
                if circuit_graph.num_nodes() % 10 == 0 and prev_layer_name:
                    print(f"  Tracing: {current_layer} -> {prev_layer_name}" + 
                          f" ({'activation' if is_activation_layer else 'linear'} layer)")
                
                if prev_layer_name:
                    prev_layer_activations = activation_dataset.get_activation_matrix(prev_layer_name)
                    input_vector = prev_layer_activations[example_idx]
                else:
                    raw_obs = activation_dataset.get_flattened_observations()[example_idx]
                    input_vector = self.model.norm_obs(torch.tensor(raw_obs, dtype=torch.float32)).numpy()
                
                if is_activation_layer:
                    important_contributors = [(current_neuron_idx, input_vector[current_neuron_idx])]
                else:
                    important_contributors = self.get_important_contributors(
                        current_layer,
                        input_vector,
                        current_neuron_idx,
                        current_activation,
                        threshold
                    )

                for component_idx, contribution_value in important_contributors:
                    prev_node = (prev_layer_name, component_idx)
                    circuit_graph.add_edge(prev_node, current_node, float(contribution_value))
                    
                    self.to_process.append((prev_layer_name, component_idx, float(input_vector[component_idx])))
                    
            print(f"Completed {polarity} circuit ({circuit_graph.num_nodes()} nodes)")
            circuit_graph.layer_stats()
            polarity_result['nodes'] = circuit_graph.nodes
            polarity_result['edges'] = circuit_graph.edges
            result[polarity] = polarity_result

        print(f"Circuit tracing complete for {layer_name} neuron {neuron_idx}. ")
        return result


class CircuitGraph:
    def __init__(self, polarity: str, example_idx: int):
        self.polarity = polarity  # 'positive' or 'negative'
        self.example_idx = example_idx
        self.nodes = {}  # (layer_name, neuron_idx) -> activation
        self.edges = {}  # (src_node, target_node) -> contribution
        self.layer_count = {}  # layer_name -> count
        
    def add_node(self, layer_name, neuron_idx, activation):
        node_id = (layer_name, neuron_idx)
        if node_id in self.nodes:
            raise ValueError(f"Node {node_id} already exists")
        self.nodes[node_id] = activation
        self.layer_count[layer_name] = self.layer_count.get(layer_name, 0) + 1

    def add_edge(self, src_node, target_node, contribution):
        if (src_node, target_node) in self.edges:
            raise ValueError(f"Edge {src_node} -> {target_node} already exists")
        self.edges[(src_node, target_node)] = contribution

    def num_nodes(self):
        return len(self.nodes)

    def num_edges(self):
        return len(self.edges)
    
    def layer_stats(self):
        for layer_name, count in self.layer_count.items():
            print(f"{layer_name}: {count}")
