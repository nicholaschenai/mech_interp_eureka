"""
PCA Visualization module for creating visual representations of PCA analysis results.

This module provides tools to visualize principal components, projected activations,
and their relationships with features for better interpretation of dimensionality
reduction results.

Responsibilities:
- Visualize principal components and their explained variance
- Plot projected activations in 2D/3D spaces
- Create heatmaps of feature-component correlations
- Visualize regression coefficients between features and components
"""
import numpy as np
import matplotlib.pyplot as plt

from typing import Dict, List, Any, Optional, Tuple
from matplotlib.figure import Figure

from .base_visualizer import BaseVisualizer


class PCAVisualizer(BaseVisualizer):
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 100):
        super().__init__(figsize, dpi)
    
    # PCA related
    def plot_pca_components(self, 
                           components: np.ndarray, 
                           layer_name: str,
                           title: Optional[str] = None) -> Figure:
        """
        Visualize the principal components as a heatmap.
        
        Args:
            components: Principal component vectors with shape (n_components, n_features)
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        im = ax.imshow(components, cmap='RdBu_r', aspect='auto', interpolation='none')
        im.set_clim(-np.max(np.abs(components)), np.max(np.abs(components)))  # Center colormap at 0
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Component Weight')
        
        ax.set_ylabel('Principal Components')
        ax.set_xlabel(f'Neurons in {layer_name}')
        
        ax.set_yticks(range(components.shape[0]))
        ax.set_yticklabels([f'PC{i+1}' for i in range(components.shape[0])])
        ax.set_xticks([])  # Hide x-ticks for cleaner visualization
        
        if title is None:
            title = f'Principal Components for {layer_name}'
        ax.set_title(title)
        
        self.figures['components'] = fig
        
        return fig
    
    def plot_explained_variance(self, 
                               explained_variance_ratio: np.ndarray,
                               title: Optional[str] = None) -> Figure:
        """
        Args:
            explained_variance_ratio: Proportion of variance explained by each component
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        ax.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
        ax.plot(range(1, len(explained_variance_ratio) + 1), 
                np.cumsum(explained_variance_ratio), 'ro-', label='Cumulative')
        
        ax.set_xlabel('Principal Component')
        ax.set_ylabel('Proportion of Variance Explained')
        
        if title is None:
            title = 'Explained Variance by Principal Component'
        ax.set_title(title)
        
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        self.figures['explained_variance'] = fig
        
        return fig
    
    # raw feature-activation related
    def plot_projected_activations_by_feature(self,
                                            projected_activations: np.ndarray,
                                            feature_values: np.ndarray,
                                            components: Tuple[int, int] = (0, 1),
                                            title: Optional[str] = None) -> Figure:
        """
        Plot projected activations colored by a feature value.
        
        Args:
            projected_activations: Activations projected onto principal components
            feature_values: Feature values to use for coloring points
            components: Tuple of component indices to plot (default: first two components)
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Extract components to plot
        comp1, comp2 = components
        x = projected_activations[:, comp1]
        y = projected_activations[:, comp2]
        
        # Plot scatter with color based on feature values
        scatter = ax.scatter(x, y, c=feature_values, cmap='viridis', alpha=0.7)
        
        plt.colorbar(scatter, ax=ax, label='Feature Value')
        
        ax.set_xlabel(f'Principal Component {comp1+1}')
        ax.set_ylabel(f'Principal Component {comp2+1}')
        
        if title is None:
            title = f'Activations Projected onto PC{comp1+1} and PC{comp2+1}'
        ax.set_title(title)
        
        ax.grid(True, alpha=0.3)
        
        self.figures[f'projected_pc{comp1+1}_pc{comp2+1}'] = fig
        
        return fig
    
    def plot_3d_projected_activations(self,
                                    projected_activations: np.ndarray,
                                    feature_values: np.ndarray,
                                    components: Tuple[int, int, int] = (0, 1, 2),
                                    title: Optional[str] = None) -> Figure:
        """
        Create a 3D scatter plot of activations projected onto three principal components.
        basically 3d version of plot_projected_activations_by_feature
        
        Args:
            projected_activations: Activations projected onto principal components
            feature_values: Feature values to use for coloring points
            components: Tuple of component indices to plot (default: first three components)
        """
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        ax = fig.add_subplot(111, projection='3d')
        
        comp1, comp2, comp3 = components
        x = projected_activations[:, comp1]
        y = projected_activations[:, comp2]
        z = projected_activations[:, comp3]
        
        # Plot 3D scatter with color based on feature values
        scatter = ax.scatter(x, y, z, c=feature_values, cmap='viridis', alpha=0.7)
        
        plt.colorbar(scatter, ax=ax, label='Feature Value')
        
        ax.set_xlabel(f'Principal Component {comp1+1}')
        ax.set_ylabel(f'Principal Component {comp2+1}')
        ax.set_zlabel(f'Principal Component {comp3+1}')
        
        if title is None:
            title = f'3D Projection onto PC{comp1+1}, PC{comp2+1}, and PC{comp3+1}'
        ax.set_title(title)
        
        self.figures[f'3d_projected_pc{comp1+1}_pc{comp2+1}_pc{comp3+1}'] = fig
        
        return fig
    
    # feature-component correlations
    def plot_feature_component_correlation_heatmap(self,
                                                feature_correlations: np.ndarray,
                                                features: List[str],
                                                title: Optional[str] = None) -> Figure:
        """
        Args:
            feature_correlations: Correlation matrix with shape (n_features, n_components)
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        component_labels = [f'PC{i+1}' for i in range(feature_correlations.shape[1])]
        
        im = ax.imshow(feature_correlations, cmap='RdBu_r', aspect='auto', interpolation='none')
        im.set_clim(-1, 1)  # Correlation values range from -1 to 1
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Correlation')
        
        ax.set_xticks(range(len(component_labels)))
        ax.set_xticklabels(component_labels)
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        
        for i in range(feature_correlations.shape[0]):
            for j in range(feature_correlations.shape[1]):
                ax.text(j, i, f'{feature_correlations[i, j]:.2f}', 
                       ha='center', va='center', 
                       color='white' if abs(feature_correlations[i, j]) > 0.5 else 'black')
        
        ax.set_xlabel('Principal Components')
        ax.set_ylabel('Features')
        
        if title is None:
            title = 'Feature-Component Correlation Heatmap'
        ax.set_title(title)
        
        self.figures['feature_correlation'] = fig
        
        return fig
    
    # feature-component regression related
    def plot_regression_coefficients(self,
                                   regression_results: Dict[str, Any],
                                   features: List[str],
                                   n_components: Optional[int] = None,
                                   title: Optional[str] = None) -> Figure:
        coefficients = regression_results['coefficients']
        r2_scores = regression_results['r2_scores']
        
        # Limit number of components if specified
        if n_components is not None:
            coefficients = coefficients[:, :n_components]
            r2_scores = r2_scores[:n_components]
        else:
            n_components = coefficients.shape[1]
        
        # Create figure with subplots for each component
        fig, axes = plt.subplots(1, n_components, 
                                figsize=(self.figsize[0] * n_components // 2, self.figsize[1]),
                                dpi=self.dpi)
        
        if n_components == 1:
            axes = [axes]
        
        # Plot coefficients for each component
        for i in range(n_components):
            ax = axes[i]
            
            # Sort coefficients by absolute value
            sorted_indices = np.argsort(np.abs(coefficients[:, i]))[::-1]
            sorted_features = [features[j] for j in sorted_indices]
            sorted_coeffs = coefficients[sorted_indices, i]
            
            # Plot horizontal bar chart
            bars = ax.barh(range(len(sorted_features)), sorted_coeffs)
            
            # Color bars based on sign
            for j, bar in enumerate(bars):
                bar.set_color('royalblue' if sorted_coeffs[j] > 0 else 'crimson')
            
            # Make sure each subplot shows its own labels
            ax.set_yticks(range(len(sorted_features)))
            ax.set_yticklabels(sorted_features)
            
            # Optional: explicitly show y-labels for all subplots
            if i > 0:  # For non-first plots
                ax.tick_params(axis='y', which='both', labelleft=True)
            
            ax.set_xlabel('Coefficient')
            
            ax.set_title(f'PC{i+1} (R²={r2_scores[i]:.2f})')
            
            ax.grid(True, alpha=0.3)
        
        # Set common y-label for all subplots
        fig.text(0.01, 0.5, 'Features', va='center', rotation='vertical')
        
        if title is None:
            title = 'Regression Coefficients for Principal Components'
        fig.suptitle(title)
        
        plt.tight_layout(rect=[0.03, 0, 1, 0.95])
        
        self.figures['regression_coefficients'] = fig
        
        return fig
