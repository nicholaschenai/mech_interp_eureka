"""
PCA Projection Plots module for visualizing the effects of PCA bottlenecks on model performance.

This module provides visualization tools to analyze and interpret how PCA bottlenecking
affects model outputs, with a focus on action fidelity metrics.

Responsibilities:
- Plot action fidelity metrics across different component counts
- Visualize per-dimension errors in bottlenecked actions
- Create summary visualizations highlighting critical thresholds
- Generate comparative visualizations of original vs. bottlenecked outputs
- Analyze time dimension effects in PCA bottlenecking
"""
import os

import numpy as np
import matplotlib.pyplot as plt

from typing import Dict, List, Tuple, Optional, Union

from .base_visualizer import BaseVisualizer


class PCAProjectionPlotter(BaseVisualizer):
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 100):
        super().__init__(figsize, dpi)

    def create_summary_plot(self,
                          metrics: Dict[int, Dict],
                          output_dir: Optional[str] = None,
                          title: Optional[str] = None) -> plt.Figure:
        """
        Args:
            metrics: Dict mapping component counts to metric dictionaries
        """
        component_counts = sorted(metrics.keys())
        
        fig, ax1 = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        ax2 = ax1.twinx()
        
        # Plot MSE metrics on primary y-axis (lower is better)
        ax1.set_xlabel('Number of PCA Components')
        ax1.set_ylabel('Normalized MSE', color='tab:blue')
        ax1.plot(component_counts, [metrics[c]['mu_nmse'] for c in component_counts], 
                'o-', linewidth=2, label='Action NMSE', color='tab:blue')
        ax1.plot(component_counts, [metrics[c]['value_nmse'] for c in component_counts], 
                's-', linewidth=2, label='Value NMSE', color='tab:cyan')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        
        # Add horizontal line at 0 for reference
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        
        # Plot cosine similarity on secondary y-axis (higher is better)
        ax2.set_ylabel('Cosine Similarity', color='tab:red')
        ax2.plot(component_counts, [metrics[c]['mu_cosine'] for c in component_counts], 
                'x-', linewidth=2, label='Cosine Similarity', color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        
        # Add horizontal line at 1 for reference
        ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.3)
        
        # Use log scale for x-axis if there are many component counts
        if len(component_counts) > 3:
            ax1.set_xscale('log')
        
        ax1.set_xticks(component_counts)
        ax1.set_xticklabels(component_counts)
        
        ax1.grid(True, alpha=0.3)
        
        if title is None:
            title = 'Summary of PCA Bottleneck Effects on Model Outputs'
        plt.title(title)
        
        # Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = "pca_bottleneck_summary.png"
            plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
        
        # Store and return figure
        self.figures['summary'] = fig
        return fig
    
    def plot_metrics_vs_components(self, 
                                 metrics: Dict[int, Dict], 
                                 metric_names: Union[str, List[str]],
                                 output_dir: Optional[str] = None,
                                 title: Optional[str] = None) -> plt.Figure:
        """
        Plot metrics against number of PCA components.
        
        Args:
            metrics: Dict mapping component counts to metric dictionaries
            metric_names: Single metric name or list of metric names to plot
            output_dir: Directory to save plot (if None, plot is not saved)
            title: Optional custom title
            
        Returns:
            Matplotlib Figure object
        """
        if isinstance(metric_names, str):
            metric_names = [metric_names]
            
        components = sorted(metrics.keys())
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Define color map for multiple metrics
        cmap = plt.cm.get_cmap('tab10', len(metric_names))
        
        # Define displayable metric names and their behavior
        metric_display = {
            'mu_mse': ('Action MSE', 'lower'),
            'mu_nmse': ('Normalized MSE', 'lower'),
            'clipped_mu_mse': ('Clipped Action MSE', 'lower'),
            'mu_max_dev': ('Max Deviation', 'lower'),
            'mu_cosine': ('Cosine Similarity', 'higher'),
            'clipped_mu_cosine': ('Clipped Cosine Similarity', 'higher'),
            'value_mse': ('Value MSE', 'lower'),
            'value_nmse': ('Value NMSE', 'lower'),
        }
        
        for i, metric in enumerate(metric_names):
            # Extract values and plot
            values = [metrics[c][metric] for c in components]
            
            # Get display name and preferred direction
            display_name, direction = metric_display.get(metric, (metric, 'lower'))
            
            ax.plot(components, values, 'o-', 
                   label=display_name,
                   color=cmap(i),
                   linewidth=2, 
                   markersize=8)
            
            # If this is a maximizing metric (like cosine similarity), mark the best point
            if direction == 'higher':
                best_idx = np.argmax(values)
                ax.plot(components[best_idx], values[best_idx], 'o', 
                       color=cmap(i), markersize=12,
                       markeredgecolor='black', markeredgewidth=2)
            else:  # Minimizing metric (like MSE)
                best_idx = np.argmin(values)
                ax.plot(components[best_idx], values[best_idx], 'o', 
                       color=cmap(i), markersize=12,
                       markeredgecolor='black', markeredgewidth=2)
        
        ax.set_xlabel('Number of PCA Components')
        ax.set_ylabel('Metric Value')
        
        if title is None:
            title = 'Action Fidelity Metrics vs. PCA Components'
        ax.set_title(title)
        
        if len(components) > 3:
            ax.set_xscale('log')
        
        ax.set_xticks(components)
        ax.set_xticklabels(components)
        
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        metric_str = '_'.join(metric_names)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = f"metrics_vs_components_{metric_str}.png"
            plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
        
        # Store and return figure
        self.figures[f'metrics_{metric_str}'] = fig
        return fig
    
    def plot_dimension_errors(self, 
                            dimension_errors: Dict[int, Dict], 
                            component_count: int,
                            output_dir: Optional[str] = None,
                            n_dims: int = 9,
                            show_clipped: str = 'both') -> plt.Figure:
        """
        Plot error metrics for each action dimension.
        
        Args:
            dimension_errors: Dictionary with per-dimension error metrics by component count
            component_count: Component count to visualize
            output_dir: Directory to save plot (if None, plot is not saved)
            n_dims: Number of action dimensions to show (if None, show all)
            show_clipped: Which version to show: 'unclipped', 'clipped', or 'both'
            
        Returns:
            Matplotlib Figure object
        """
        # Get data for the specified component count
        errors = dimension_errors[component_count]
        
        # Create figure - size depends on whether we're showing both or not
        if show_clipped == 'both':
            fig, axes = plt.subplots(2, 2, figsize=(self.figsize[0], self.figsize[1]*1.5), 
                                     dpi=self.dpi, sharex='col')
            axes = axes.flatten()
        else:
            fig, axes = plt.subplots(2, 1, figsize=self.figsize, dpi=self.dpi, sharex=True)
            axes = axes.flatten()
        
        # Determine which prefixes to use
        prefixes = []
        if show_clipped == 'unclipped' or show_clipped == 'both':
            prefixes.append('')
        if show_clipped == 'clipped' or show_clipped == 'both':
            prefixes.append('clipped_')
        
        # Process each version (unclipped and/or clipped)
        for i, prefix in enumerate(prefixes):
            # Get number of dimensions
            dim_count = len(errors[f'{prefix}per_dim_mse'])
            
            # Limit to specified number of dimensions if needed
            if n_dims is not None and n_dims < dim_count:
                # Use the n_dims most affected dimensions
                indices = errors[f'{prefix}most_affected_dims'][:n_dims]
            else:
                indices = np.arange(dim_count)
                n_dims = dim_count
            
            # Prepare data
            dimensions = np.arange(len(indices))
            mse_values = errors[f'{prefix}per_dim_mse'][indices]
            rel_error_values = errors[f'{prefix}per_dim_relative_error'][indices]
            
            # Calculate axis indices based on whether showing both or not
            if show_clipped == 'both':
                mse_ax_idx = i
                rel_err_ax_idx = i + 2
            else:
                mse_ax_idx = 0
                rel_err_ax_idx = 1
            
            # Plot MSE by dimension
            ax_mse = axes[mse_ax_idx]
            version_label = "Unclipped" if prefix == '' else "Clipped"
            bars1 = ax_mse.bar(dimensions, mse_values, color='royalblue', alpha=0.7)
            ax_mse.set_ylabel('Mean Squared Error')
            ax_mse.set_title(f'{version_label} Per-Dimension MSE with {component_count} Components')
            ax_mse.grid(True, alpha=0.3, axis='y')
            
            # Add dimension indices as text on top of bars
            for j, bar in enumerate(bars1):
                dim_idx = indices[j]
                ax_mse.text(j, bar.get_height() * 1.01, f'Dim {dim_idx}', 
                         ha='center', va='bottom', rotation=90, fontsize=8)
            
            # Plot relative error by dimension
            ax_rel = axes[rel_err_ax_idx]
            bars2 = ax_rel.bar(dimensions, rel_error_values, color='firebrick', alpha=0.7)
            ax_rel.set_xlabel('Action Dimension')
            ax_rel.set_ylabel('Relative Error')
            ax_rel.set_title(f'{version_label} Per-Dimension Relative Error')
            ax_rel.grid(True, alpha=0.3, axis='y')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if output directory is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            clipped_str = 'both' if show_clipped == 'both' else show_clipped
            filename = f"dimension_errors_comp{component_count}_{clipped_str}.png"
            plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
        
        # Store and return figure
        fig_key = f'dim_errors_{component_count}_{show_clipped}'
        self.figures[fig_key] = fig
        return fig
    
    def plot_dimension_errors_heatmap(self,
                                    dimension_errors: Dict[int, Dict],
                                    metric: str = 'per_dim_mse',
                                    output_dir: Optional[str] = None,
                                    title: Optional[str] = None,
                                    use_clipped: bool = False) -> plt.Figure:
        """
        Create a heatmap of errors across dimensions and component counts.
        
        Args:
            dimension_errors: Dictionary with per-dimension error metrics by component count
            metric: Which metric to use ('per_dim_mse', 'per_dim_relative_error', etc.)
            output_dir: Directory to save plot (if None, plot is not saved)
            title: Optional custom title
            use_clipped: Whether to use clipped version of metrics
            
        Returns:
            Matplotlib Figure object
        """
        # Apply the clipped_ prefix if requested
        prefixed_metric = f"clipped_{metric}" if use_clipped else metric
        metric_label_suffix = " (Clipped)" if use_clipped else ""
        
        # Extract component counts and dimensions
        component_counts = sorted(dimension_errors.keys())
        n_dims = len(dimension_errors[component_counts[0]][prefixed_metric])
        
        # Create data matrix for heatmap
        data = np.zeros((len(component_counts), n_dims))
        
        # Fill data matrix
        for i, comp in enumerate(component_counts):
            data[i] = dimension_errors[comp][prefixed_metric]
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Plot heatmap
        im = ax.imshow(data, cmap='viridis', aspect='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        
        # Set labels for the colorbar based on metric
        base_metric = metric.replace('clipped_', '')  # Remove prefix if present
        metric_labels = {
            'per_dim_mse': 'Mean Squared Error',
            'per_dim_max_error': 'Maximum Error',
            'per_dim_nmse': 'Normalized MSE',
            'per_dim_relative_error': 'Relative Error'
        }
        cbar_label = metric_labels.get(base_metric, base_metric) + metric_label_suffix
        cbar.set_label(cbar_label)
        
        # Set labels and title
        ax.set_xlabel('Action Dimension')
        ax.set_ylabel('Number of PCA Components')
        
        if title is None:
            metric_display = metric_labels.get(base_metric, base_metric) + metric_label_suffix
            title = f'Error Heatmap: {metric_display} by Dimension and Component Count'
        ax.set_title(title)
        
        # Set ticks
        ax.set_xticks(np.arange(n_dims))
        ax.set_yticks(np.arange(len(component_counts)))
        ax.set_yticklabels(component_counts)
        
        # Save if output directory is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            clipped_str = "clipped" if use_clipped else "unclipped"
            filename = f"dimension_heatmap_{base_metric}_{clipped_str}.png"
            plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
        
        # Store and return figure
        clipped_str = "clipped" if use_clipped else "unclipped"
        self.figures[f'heatmap_{base_metric}_{clipped_str}'] = fig
        return fig
