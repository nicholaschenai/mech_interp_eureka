import torch
import numpy as np
from typing import Dict, List, Tuple

from .sae_model import SAEModel
from ..activation_dataset import ActivationDataset


class SAEAnalyzer:
    """Analysis tools for trained SAE models"""
    
    def __init__(self, sae_model: SAEModel):
        """Initialize with trained SAE model"""
        self.sae_model = sae_model
        
    def get_feature_vectors(self) -> np.ndarray:
        """
        Get feature vectors from decoder weights.
        
        Returns:
            Array of shape (hidden_dim, input_dim) where each row is a feature vector
        """
        # Get normalized decoder weights and transpose them
        # The decoder weights have shape (input_dim, hidden_dim)
        # We want each feature as a row, so we transpose to (hidden_dim, input_dim)
        return self.sae_model.W_dec_normalized.detach().cpu().numpy().T
        
    def analyze_feature_activations(self, activations: np.ndarray) -> Dict:
        """Analyze how features activate on dataset"""
        # Use the model's encode method to get encoded features
        activations = torch.from_numpy(activations).to(self.sae_model.device)
        with torch.no_grad():
            encoded_features = self.sae_model.encode(activations)
            
        # Convert to numpy for analysis
        encoded_acts = encoded_features.cpu().numpy()
        
        # Calculate activation statistics
        stats = {
            "mean": encoded_acts.mean(axis=0),
            "std": encoded_acts.std(axis=0),
            "max": encoded_acts.max(axis=0),
            "sparsity": (np.abs(encoded_acts) > 1e-8).mean(axis=0)
        }
        
        return stats
        
    def compute_feature_correlations(self, features: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[str]]:
        """Compute correlations between SAE features and input features"""
        # Get feature vectors (decoder weights)
        feature_vectors = self.get_feature_vectors()
        
        # Initialize correlation matrix
        n_features = feature_vectors.shape[0]
        n_input_features = len(features)
        correlation_matrix = np.zeros((n_features, n_input_features))
        
        # Compute correlations
        feature_names = list(features.keys())
        for i, feature_name in enumerate(feature_names):
            for j in range(n_features):
                correlation_matrix[j, i] = np.corrcoef(feature_vectors[j], features[feature_name])[0, 1]
                
        return correlation_matrix, feature_names
        
    def extract_feature_data(self, dataset: ActivationDataset, layer_name: str) -> Dict:
        """Extract feature weights and statistics for visualization"""
        # Get feature vectors (decoder weights)
        feature_vectors = self.get_feature_vectors()
        
        # Get activations for this layer
        activations = dataset.get_activation_matrix(layer_name)
        
        # Analyze feature activations
        activation_stats = self.analyze_feature_activations(activations)
        
        # Debug dimension info
        print(f"Model hidden dimension: {self.sae_model.hidden_dim}")
        print(f"Feature vectors shape: {feature_vectors.shape}")
        print(f"Activation stats mean shape: {activation_stats['mean'].shape}")
        
        # Ensure feature vectors and activation stats have consistent dimensions
        if feature_vectors.shape[0] != self.sae_model.hidden_dim:
            print(f"Warning: Feature vectors dimension {feature_vectors.shape[0]} doesn't match model hidden dimension {self.sae_model.hidden_dim}")
            # Truncate or pad feature vectors if needed
            if feature_vectors.shape[0] > self.sae_model.hidden_dim:
                feature_vectors = feature_vectors[:self.sae_model.hidden_dim]
        
        # Rank features by activation
        ranked_indices = self.rank_features_by_activation(activation_stats["mean"])
        
        # Ensure ranked indices are within bounds of feature vectors
        valid_indices = [idx for idx in ranked_indices if idx < feature_vectors.shape[0]]
        if len(valid_indices) < len(ranked_indices):
            print(f"Warning: {len(ranked_indices) - len(valid_indices)} ranked indices were out of bounds and removed")
        
        # Prepare data for saving
        feature_data = {
            "feature_vectors": feature_vectors,
            "activation_stats": activation_stats,
            "ranked_indices": valid_indices,
            "layer_name": layer_name,
            "input_dim": self.sae_model.input_dim,
            "hidden_dim": self.sae_model.hidden_dim
        }
        
        return feature_data
        
    def rank_features_by_activation(self, activations: np.ndarray) -> List[int]:
        """
        Rank features by their mean activation across the dataset
        
        Args:
            activations: Array of feature activations, shape (hidden_dim,)
            
        Returns:
            List of feature indices sorted by activation magnitude (highest first)
        """
        # Ensure we're not exceeding the model's feature count
        if len(activations) > self.sae_model.hidden_dim:
            print(f"Warning: Activation vector length {len(activations)} exceeds model hidden dimension {self.sae_model.hidden_dim}")
            activations = activations[:self.sae_model.hidden_dim]
        elif len(activations) < self.sae_model.hidden_dim:
            print(f"Warning: Activation vector length {len(activations)} is less than model hidden dimension {self.sae_model.hidden_dim}")
        
        # Sort features by their average activation magnitude (higher = more important)
        return list(np.argsort(-np.abs(activations)))
        
    def analyze_phase_specific_features(self, activations_by_phase: Dict[str, np.ndarray]) -> Dict:
        """Identify features that activate strongly in specific phases"""
        # Get the list of phases
        phases = list(activations_by_phase.keys())
        
        # Analyze activations for each phase
        phase_stats = {}
        for phase in phases:
            phase_stats[phase] = self.analyze_feature_activations(activations_by_phase[phase])
            
        # Find phase-specific features
        # A feature is phase-specific if it activates much more strongly in one phase than others
        n_features = self.sae_model.hidden_dim
        phase_specific_features = {phase: [] for phase in phases}
        
        for feature_idx in range(n_features):
            # Get mean activation for this feature in each phase
            activations_across_phases = [phase_stats[phase]["mean"][feature_idx] for phase in phases]
            
            # Find phase with maximum activation
            max_phase_idx = np.argmax(activations_across_phases)
            max_phase = phases[max_phase_idx]
            
            # Check if this feature is specific to this phase
            # Criterion: activation in top phase is at least 2x higher than others
            max_activation = activations_across_phases[max_phase_idx]
            other_activations = [act for i, act in enumerate(activations_across_phases) if i != max_phase_idx]
            
            if other_activations and max_activation > 2.0 * max(other_activations):
                phase_specific_features[max_phase].append({
                    "feature_idx": feature_idx,
                    "activation_ratio": max_activation / max(other_activations) if max(other_activations) > 0 else float('inf')
                })
                
        return phase_specific_features 

    def get_feature_distributions_by_phase(self, dataset: ActivationDataset, layer_name: str) -> Dict:
        """
        Compute feature activation distributions for each phase.
        
        Args:
            dataset: ActivationDataset containing the activations
            layer_name: Name of the layer to analyze
            
        Returns:
            Dict mapping feature_idx -> phase_name -> list of activation values
            Example: {0: {'approaching': array([...]), 'grasping': array([...])}, ...}
        """
        # Get activations for all phases
        phase_activations, _ = dataset.get_phase_activations(layer_name)
        phase_names = list(phase_activations.keys())
        
        # Create feature activation distributions by phase
        feature_distributions = {}
        
        for phase_name in phase_names:
            # Get activations for this phase
            phase_acts = phase_activations[phase_name]
            
            # Encode with SAE to get sparse features
            phase_acts_tensor = torch.from_numpy(phase_acts).to(self.sae_model.device)
            with torch.no_grad():
                encoded_features = self.sae_model.encode(phase_acts_tensor)
            
            # Convert to numpy for easier analysis
            encoded_acts = encoded_features.cpu().numpy()
            
            # Store activations by feature and phase
            n_features = encoded_acts.shape[1]
            for feature_idx in range(n_features):
                feature_activations = encoded_acts[:, feature_idx]
                
                # Store activation values for histogram creation
                if feature_idx not in feature_distributions:
                    feature_distributions[feature_idx] = {}
                feature_distributions[feature_idx][phase_name] = feature_activations
        
        return feature_distributions
    
    def find_phase_specific_features(self, phase_distributions: Dict) -> Dict:
        """
        Identify features that are specific to certain phases.
        
        Args:
            phase_distributions: Dict mapping feature_idx -> phase_name -> activation values
            
        Returns:
            Dict mapping phase_name -> list of {feature_idx, specificity_score} dicts
        """
        if not phase_distributions:
            return {}
        
        # Get list of phases from the first feature's distributions
        first_feature = next(iter(phase_distributions.values()))
        phase_names = list(first_feature.keys())
        
        # Initialize result structure
        phase_specific_features = {phase: [] for phase in phase_names}
        
        # For each feature, check if it's specific to a phase
        for feature_idx, phase_data in phase_distributions.items():
            # Calculate mean activation in each phase
            mean_activations = {phase: np.mean(np.abs(acts)) for phase, acts in phase_data.items()}
            
            # Find phase with maximum activation
            max_phase = max(mean_activations, key=mean_activations.get)
            max_activation = mean_activations[max_phase]
            
            # Calculate ratio of max activation to other phases
            other_phases = [p for p in phase_names if p != max_phase]
            if not other_phases:
                continue
                
            other_activations = [mean_activations[p] for p in other_phases]
            max_other = max(other_activations) if other_activations else 0
            
            # If activation in best phase is at least 2x the second best,
            # consider it phase-specific
            if max_other > 0 and max_activation / max_other >= 2.0:
                specificity_score = max_activation / max_other
                phase_specific_features[max_phase].append({
                    "feature_idx": feature_idx,
                    "specificity_score": specificity_score
                })
        
        # Sort features by specificity score
        for phase in phase_names:
            phase_specific_features[phase] = sorted(
                phase_specific_features[phase],
                key=lambda x: x["specificity_score"],
                reverse=True
            )
        
        return phase_specific_features 
