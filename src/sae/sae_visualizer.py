import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any

from .sae_model import SAEModel
from ..activation_dataset import ActivationDataset


class SAEVisualizer:
    """Visualization tools for SAE models"""
    
    def __init__(self, sae_model: SAEModel, dataset: ActivationDataset, layer_name: str):
        """Initialize with model, dataset and layer name"""
        self.sae_model = sae_model
        self.dataset = dataset
        self.layer_name = layer_name
        
    def plot_feature_activation_by_phase(self, n_top_features: int = 10) -> None:
        """Plot top feature activations by task phase"""
        # This is for part 2, we'll implement it for the simpler approach
        pass
        
    def plot_feature_correlation_with_observations(self) -> None:
        """Plot correlation heatmap between features and observations"""
        pass
        
    def plot_feature_vectors(self, feature_vectors: np.ndarray, feature_indices: List[int], 
                            title: str = "Top SAE Features") -> plt.Figure:
        """
        Plot feature vectors for selected features
        
        Args:
            feature_vectors: Array of feature vectors (n_features x input_dim)
            feature_indices: Indices of features to plot
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        n_features = len(feature_indices)
        rows = int(np.ceil(np.sqrt(n_features)))
        cols = int(np.ceil(n_features / rows))
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        if rows * cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for i, idx in enumerate(feature_indices):
            if i >= len(axes):
                break
                
            # Get feature vector
            feature = feature_vectors[idx]
            
            # Plot as 1D heatmap
            im = axes[i].imshow(feature.reshape(1, -1), cmap='coolwarm', vmin=-0.5, vmax=0.5)
            axes[i].set_title(f"Feature {idx}")
            axes[i].set_yticks([])
            
        # Hide unused subplots
        for i in range(n_features, len(axes)):
            axes[i].axis('off')
            
        plt.tight_layout()
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Weight Value')
        
        fig.suptitle(title, fontsize=16, y=0.98)
        plt.subplots_adjust(top=0.9)
        
        return fig
        
    def plot_reconstruction_quality(self) -> None:
        """Plot original vs reconstructed activations"""
        pass
        
    def visualize_features(self, feature_data: Dict, top_k: int = 16, 
                          use_stacked_layout: bool = True) -> plt.Figure:
        """
        Visualize top features from feature data
        
        Args:
            feature_data: Dictionary with feature data
            top_k: Number of top features to visualize
            use_stacked_layout: Whether to use stacked layout (True) or grid layout (False)
            
        Returns:
            Matplotlib figure
        """
        feature_vectors = feature_data["feature_vectors"]
        metadata = {k: v for k, v in feature_data.items() if k != "feature_vectors"}
        
        # Ensure indices are within bounds
        num_features = feature_vectors.shape[0]
        ranked_indices = metadata["ranked_indices"]
        valid_indices = [idx for idx in ranked_indices if idx < num_features]
        top_indices = valid_indices[:min(top_k, len(valid_indices))]

        # Determine layout
        if use_stacked_layout:
            # For stacked layout, create a figure with stacked features
            fig = self.create_stacked_feature_vis(feature_vectors, top_indices, metadata)
        else:
            # For grid layout, determine grid size
            grid_size = (int(np.ceil(np.sqrt(len(top_indices)))), int(np.ceil(np.sqrt(len(top_indices)))))
            # Create visualization
            fig = self.create_feature_grid(feature_vectors, top_indices, metadata, grid_size)
        
        return fig
        
    def create_stacked_feature_vis(self, feature_weights: np.ndarray,
                                 feature_indices: List[int],
                                 metadata: Dict) -> plt.Figure:
        """
        Create a stacked visualization of features (one above the other)
        
        Args:
            feature_weights: Array of feature weight vectors
            feature_indices: Indices of features to visualize
            metadata: Additional metadata about features
            
        Returns:
            Matplotlib figure
        """
        n_features = len(feature_indices)
        # Create figure with enough height for all features
        fig, axes = plt.subplots(n_features, 1, figsize=(12, n_features * 1.2))
        
        if n_features == 1:
            axes = [axes]
        
        # Create activation stats labels if available
        activation_labels = {}
        if "activation_stats" in metadata:
            stats = metadata["activation_stats"]
            for i, idx in enumerate(feature_indices):
                mean_act = stats["mean"][idx] if "mean" in stats else "?"
                sparsity = stats["sparsity"][idx] if "sparsity" in stats else "?"
                activation_labels[idx] = f"Mean: {mean_act:.2f}, Sparsity: {sparsity:.2f}"
        
        # Plot each feature
        for i, idx in enumerate(feature_indices):
            # Get feature weights
            feature = feature_weights[idx]
            
            # Display as 1D vector with aspect ratio that makes it clearly visible
            im = axes[i].imshow(feature.reshape(1, -1), aspect='auto', 
                               cmap='coolwarm', vmin=-0.5, vmax=0.5)
            
            # Add title with feature index
            title = f"Feature {idx}"
            if idx in activation_labels:
                title += f" | {activation_labels[idx]}"
            axes[i].set_title(title, fontsize=10)
            axes[i].set_yticks([])
            
            # Add x-label on the bottom axis only
            if i == n_features - 1:
                axes[i].set_xlabel("Input Neurons")
        
        # Adjust spacing between subplots
        plt.tight_layout()
        
        # Add horizontal colorbar at the top
        # First, adjust the figure to make room for the colorbar
        fig.subplots_adjust(top=0.9, right=0.95, left=0.05)
        
        # Add colorbar axes above the plots
        cbar_ax = fig.add_axes([0.15, 0.93, 0.7, 0.02])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('Weight Value')
        
        # Add title
        layer_name = metadata.get("layer_name", "Unknown Layer")
        fig.suptitle(f"Top SAE Features for {layer_name}", fontsize=16, y=0.98)
        
        return fig
        
    def create_feature_grid(self, feature_weights: np.ndarray, 
                           feature_indices: List[int], 
                           metadata: Dict,
                           grid_size: Tuple[int, int] = (4, 4)) -> plt.Figure:
        """
        Create a grid visualization of features
        
        Args:
            feature_weights: Array of feature weight vectors
            feature_indices: Indices of features to visualize
            metadata: Additional metadata about features
            grid_size: Tuple of (rows, cols) for grid layout
            
        Returns:
            Matplotlib figure
        """
        rows, cols = grid_size
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        axes = axes.flatten()
        
        # Create activation stats labels if available
        activation_labels = {}
        if "activation_stats" in metadata:
            stats = metadata["activation_stats"]
            for i, idx in enumerate(feature_indices):
                if i >= len(axes):
                    break
                mean_act = stats["mean"][idx] if "mean" in stats else "?"
                sparsity = stats["sparsity"][idx] if "sparsity" in stats else "?"
                activation_labels[idx] = f"Mean: {mean_act:.2f}, Sparsity: {sparsity:.2f}"
        
        # Plot each feature
        for i, idx in enumerate(feature_indices):
            if i >= len(axes):
                break
                
            # Get feature weights
            feature = feature_weights[idx]
            
            # Determine if we need to reshape
            if "input_shape" in metadata:
                input_shape = metadata["input_shape"]
                # If input is multidimensional (like an image), reshape
                try:
                    feature_reshaped = feature.reshape(input_shape)
                    im = axes[i].imshow(feature_reshaped, cmap='coolwarm', vmin=-0.5, vmax=0.5)
                except ValueError:
                    # If reshape fails, display as 1D
                    im = axes[i].imshow(feature.reshape(1, -1), cmap='coolwarm', vmin=-0.5, vmax=0.5)
            else:
                # Display as 1D vector
                im = axes[i].imshow(feature.reshape(1, -1), cmap='coolwarm', vmin=-0.5, vmax=0.5)
            
            # Add title with feature index
            title = f"Feature {idx}"
            if idx in activation_labels:
                title += f"\n{activation_labels[idx]}"
            axes[i].set_title(title, fontsize=10)
            axes[i].set_yticks([])
            
        # Hide unused subplots
        for i in range(len(feature_indices), len(axes)):
            axes[i].axis('off')
            
        # Add colorbar
        plt.tight_layout()
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Weight Value')
        
        # Add title
        layer_name = metadata.get("layer_name", "Unknown Layer")
        fig.suptitle(f"Top SAE Features for {layer_name}", fontsize=16, y=0.98)
        plt.subplots_adjust(top=0.9)
        
        return fig 

    def plot_feature_distributions(self, 
                               phase_distributions: Dict, 
                               phase_specific_features: Dict,
                               top_k: int = 10) -> plt.Figure:
        """
        Plot histograms of feature activations across different phases.
        
        Args:
            phase_distributions: Dict mapping feature_idx -> phase_name -> activation values
            phase_specific_features: Dict mapping phase_name -> list of features with their specificity
            top_k: Number of top features to show for each phase
            
        Returns:
            Matplotlib figure with distribution plots
        """
        if not phase_distributions or not phase_specific_features:
            print("No data to visualize")
            return plt.figure()
            
        # Get list of phases
        phase_names = list(phase_specific_features.keys())
        n_phases = len(phase_names)
        
        # Determine number of features to show (limited by top_k and available features)
        features_per_phase = {}
        for phase in phase_names:
            phase_features = phase_specific_features[phase]
            features_per_phase[phase] = phase_features[:min(top_k, len(phase_features))]
        
        # Count total features to display
        total_features = sum(len(features) for features in features_per_phase.values())
        
        if total_features == 0:
            print("No phase-specific features found")
            return plt.figure()
        
        # Create a figure with subplots for each feature
        fig, axes = plt.subplots(total_features, 1, figsize=(12, 3 * total_features),
                                constrained_layout=True)
        
        # If only one feature, make axes iterable
        if total_features == 1:
            axes = [axes]
        
        # Plot each phase-specific feature
        plot_idx = 0
        colors = plt.cm.tab10.colors
        
        for phase_idx, phase in enumerate(phase_names):
            phase_color = colors[phase_idx % len(colors)]
            
            for feature_info in features_per_phase[phase]:
                feature_idx = feature_info["feature_idx"]
                specificity = feature_info["specificity_score"]
                
                # Get the subplot for this feature
                ax = axes[plot_idx]
                
                # For each phase, plot a histogram of feature activations
                for p_idx, p_name in enumerate(phase_names):
                    if feature_idx not in phase_distributions or p_name not in phase_distributions[feature_idx]:
                        continue
                        
                    # Get activations for this feature in this phase
                    activations = phase_distributions[feature_idx][p_name]
                    
                    # Plot histogram with some transparency
                    color = colors[p_idx % len(colors)]
                    ax.hist(activations, bins=30, alpha=0.6, color=color, label=p_name)
                
                # Highlight the phase this feature is specific to
                ax.set_title(f"Feature {feature_idx} - Specific to {phase} (Score: {specificity:.2f})")
                ax.legend()
                
                # Add grid for visibility
                ax.grid(True, linestyle='--', alpha=0.7)
                
                # Move to next subplot
                plot_idx += 1
        
        # Add overall title
        fig.suptitle(f"Phase-Specific Feature Distributions for {self.layer_name}", fontsize=16)
        
        return fig 
