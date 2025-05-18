"""
This module provides visualization tools for clustering results,
allowing for exploration of cluster properties and comparison with known phase labels.

Responsibilities:
- Visualize clusters in 2D using dimensionality reduction
- Compare clustering results with known phase labels
- Visualize cluster metrics across different parameters
- Plot cluster centers and distributions
"""
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.figure import Figure
from typing import Dict, List, Any, Tuple, Optional

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from .base_visualizer import BaseVisualizer


class ClusterVisualizer(BaseVisualizer):
    def __init__(self, figsize: Tuple[int, int] = (10, 8), dpi: int = 100):
        super().__init__(figsize, dpi)
        # Cache for reduced dimensions to avoid redundant computations
        self._dim_reduction_cache = {}

    # helpers
    def _get_cache_key(self, activations_id: int, method: str, n_components: int, **kwargs) -> str:
        """Generate a cache key for dimension reduction results"""
        # Use the id of the activations array and method parameters to create a unique key
        param_str = "_".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
        return f"{id(activations_id)}_{method}_{n_components}_{param_str}"

    def clear_cache(self):
        """Clear the dimension reduction cache to free memory"""
        self._dim_reduction_cache = {}
        return f"Cache cleared ({len(self._dim_reduction_cache)} items)"

    def _reduce_dimensions(self, 
                          activations: np.ndarray, 
                          method: str = 'pca',
                          n_components: int = 2,
                          use_cache: bool = True,
                          **kwargs) -> np.ndarray:
        """
        Reduce dimensionality of activations for visualization.
        
        Args:
            activations: Activation matrix with shape (n_samples, n_features)
            method: Dimensionality reduction method ('pca' or 'tsne')
            **kwargs: Additional parameters for the reduction method
            
        Returns:
            Reduced activations with shape (n_samples, n_components)
        """
        if use_cache:
            cache_key = self._get_cache_key(id(activations), method, n_components, **kwargs)
            if cache_key in self._dim_reduction_cache:
                return self._dim_reduction_cache[cache_key]
        
        if np.isnan(activations).any():
            print("Warning: NaN values found in activations. Replacing with zeros.")
            activations = np.nan_to_num(activations)
        
        if method.lower() == 'pca':
            reducer = PCA(n_components=n_components, **kwargs)
        elif method.lower() == 'tsne':
            reducer = TSNE(n_components=n_components, random_state=42, **kwargs)
        else:
            raise ValueError(f"Unsupported dimensionality reduction method: {method}")
        
        reduced = reducer.fit_transform(activations)
        
        if use_cache:
            cache_key = self._get_cache_key(id(activations), method, n_components, **kwargs)
            self._dim_reduction_cache[cache_key] = reduced
            
        return reduced
    
    # summaries
    def plot_phase_distribution(self,
                               analysis_results: Dict[str, Any],
                               title: Optional[str] = None) -> Figure:
        """
        Plot the distribution of samples across phases.
        
        Args:
            analysis_results: Results from analyze_phases method
            title: Optional title override
            
        Returns:
            Matplotlib Figure object
        """
        phase_names = analysis_results['phase_names']
        n_samples = analysis_results['n_samples_per_phase']
        
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
    
        bars = ax.bar(range(len(phase_names)), [n_samples[phase] for phase in phase_names])
        
        # Add data labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                  f'{height}',
                  ha='center', va='bottom')
        
        ax.set_xlabel('Phase')
        ax.set_ylabel('Number of Samples')
        if title is None:
            title = 'Sample Distribution Across Phases'
        ax.set_title(title)
        
        ax.set_xticks(range(len(phase_names)))
        ax.set_xticklabels(phase_names)
        ax.grid(True, axis='y', alpha=0.3)
        
        self.figures['phase_distribution'] = fig
        return fig

    # clusters
    def plot_clusters_2d(self,
                        analysis_results: Dict[str, Any],
                        layer_name: str,
                        method: str = 'tsne',
                        title: Optional[str] = None,
                        **kwargs) -> Figure:
        """
        Plot clusters in 2D using dimensionality reduction.
        
        Args:
            analysis_results: Results from analyze_phases method
            method: Dimensionality reduction method ('pca' or 'tsne')
            title: Optional title override
            **kwargs: Additional parameters for dimensionality reduction
            
        Returns:
            Matplotlib Figure object
        """
        cluster_labels = analysis_results['cluster_labels']
        activations = analysis_results['activations']

        reduced = self._reduce_dimensions(activations, method=method, **kwargs)
        
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Get unique clusters and assign colors
        unique_clusters = np.unique(cluster_labels)
        n_clusters = len(unique_clusters)
        cmap = plt.cm.get_cmap('tab10' if n_clusters <= 10 else 'tab20')
        
        # Plot each cluster
        for i, cluster in enumerate(unique_clusters):
            mask = cluster_labels == cluster
            ax.scatter(
                reduced[mask, 0], 
                reduced[mask, 1], 
                s=30, 
                c=[cmap(i / n_clusters)], 
                label=f'Cluster {cluster}',
                alpha=0.7
            )
        
        if title is None:
            title = f'Clusters of {layer_name} activations ({method.upper()} projection)'
        ax.set_title(title)
        ax.set_xlabel(f'{method.upper()} Component 1')
        ax.set_ylabel(f'{method.upper()} Component 2')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        self.figures['clusters_2d'] = fig
        return fig
    
    # comparing clusters to phase labels
    def plot_clusters_by_phase(
            self,
            analysis_results: Dict[str, Any],
            layer_name: str,
            method: str = 'tsne',
            title: Optional[str] = None,
            **kwargs) -> Figure:
        """
        Plot clusters in 2D colored by both cluster and phase.
        
        Args:
            analysis_results: Results from analyze_phases method
            layer_name: Name of the layer to visualize
            method: Dimensionality reduction method ('pca' or 'tsne')
            title: Optional title override
            **kwargs: Additional parameters for dimensionality reduction
            
        Returns:
            Matplotlib Figure object
        """
        cluster_labels = analysis_results['cluster_labels']
        phase_labels = analysis_results['phase_labels']
        phase_names = analysis_results['phase_names']
        activations = analysis_results['activations']
        
        reduced = self._reduce_dimensions(activations, method=method, **kwargs)
        
        # Get unique clusters for coloring
        unique_clusters = np.unique(cluster_labels)
        n_clusters = len(unique_clusters)
        cmap = plt.cm.get_cmap('tab10' if n_clusters <= 10 else 'tab20')
        
        # Create figure with subplots for each phase
        fig, axes = plt.subplots(1, len(phase_names), 
                                figsize=(self.figsize[0] * len(phase_names) // 2, self.figsize[1]),
                                sharey=True, sharex=True, dpi=self.dpi)
        
        if len(phase_names) == 1:
            axes = [axes]
        
        # Create handles for ALL clusters upfront for the legend
        legend_handles = []
        legend_labels = []
        for j, cluster in enumerate(unique_clusters):
            # Create a scatter proxy artist for each cluster
            legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                           markerfacecolor=cmap(j / n_clusters), markersize=10))
            legend_labels.append(f'Cluster {cluster}')
        
        # Plot each phase in a separate subplot
        for i, phase_name in enumerate(phase_names):
            ax = axes[i]
            phase_mask = phase_labels == i
            
            # Track if we found any clusters in this phase
            clusters_in_phase = False
            
            # Plot each cluster within this phase
            for j, cluster in enumerate(unique_clusters):
                mask = (cluster_labels == cluster) & phase_mask
                if np.sum(mask) > 0:  # Only plot if there are points
                    clusters_in_phase = True
                    ax.scatter(
                        reduced[mask, 0], 
                        reduced[mask, 1], 
                        s=30, 
                        c=[cmap(j / n_clusters)], 
                        alpha=0.7
                    )
            
            # Add a note if no clusters found in this phase
            if not clusters_in_phase:
                ax.text(0.5, 0.5, "No data points", ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12, color='gray')
            
            ax.set_title(f'Phase: {phase_name} (n={np.sum(phase_mask)})')
            ax.set_xlabel(f'{method.upper()} Component 1')
            if i == 0:
                ax.set_ylabel(f'{method.upper()} Component 2')
            ax.grid(True, alpha=0.3)
        
        fig.legend(legend_handles, legend_labels, loc='lower center', 
                 bbox_to_anchor=(0.5, 0.02), ncol=min(n_clusters, 5))
            
        if title is None:
            title = f'Clusters by Phase for {layer_name} ({method.upper()} projection)'
        fig.suptitle(title)
        
        plt.tight_layout(rect=[0, 0.08, 1, 0.95])
        
        self.figures['clusters_by_phase'] = fig
        return fig
    
    def plot_cluster_phase_confusion(self,
                                  analysis_results: Dict[str, Any],
                                  title: Optional[str] = None) -> Figure:
        """
        Plot confusion matrix between clusters and phases.
        
        Args:
            analysis_results: Results from analyze_phases method
        """
        cluster_labels = analysis_results['cluster_labels']
        phase_names = analysis_results['phase_names']
        unique_clusters = np.unique(cluster_labels)
        
        cm_norm = analysis_results['comparison']['normalized_confusion_matrix']

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        im = ax.imshow(cm_norm, cmap='Blues', aspect='auto')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Proportion of Phase')
        
        ax.set_xticks(np.arange(len(unique_clusters)))
        ax.set_yticks(np.arange(len(phase_names)))
        ax.set_xticklabels([f'Cluster {c}' for c in unique_clusters])
        ax.set_yticklabels(phase_names)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        for i in range(len(phase_names)):
            for j in range(len(unique_clusters)):
                ax.text(j, i, f'{cm_norm[i, j]:.2f}',
                       ha="center", va="center", 
                       color="white" if cm_norm[i, j] > 0.5 else "black")
        
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Phase')
        if title is None:
            title = 'Confusion Matrix: Phases vs. Clusters'
        ax.set_title(title)
        
        fig.tight_layout()
        self.figures['confusion_matrix'] = fig
        return fig
        
    def plot_cluster_composition(self,
                               analysis_results: Dict[str, Any],
                               title: Optional[str] = None) -> Figure:
        """
        Plot pie charts showing the composition of each cluster by phase.
        
        Args:
            analysis_results: Results from analyze_phases method
        """
        cluster_labels = analysis_results['cluster_labels']
        phase_labels = analysis_results['phase_labels']
        phase_names = analysis_results['phase_names']
        
        unique_clusters = np.unique(cluster_labels)
        n_clusters = len(unique_clusters)
        
        fig, axes = plt.subplots(1, n_clusters, 
                                figsize=(self.figsize[0] * n_clusters // 2, self.figsize[1]),
                                dpi=self.dpi)
        
        if n_clusters == 1:
            axes = [axes]
        
        cmap = plt.cm.get_cmap('tab10' if len(phase_names) <= 10 else 'tab20')
        colors = [cmap(i / len(phase_names)) for i in range(len(phase_names))]
        
        # Create legend handles for all phases
        legend_handles = []
        for i, phase_name in enumerate(phase_names):
            legend_handles.append(plt.Rectangle((0, 0), 1, 1, fc=colors[i]))
        
        # Calculate phase composition for each cluster
        for i, cluster in enumerate(unique_clusters):
            ax = axes[i]
            cluster_mask = cluster_labels == cluster
            
            # Calculate count for each phase within this cluster
            phase_counts = []
            for j in range(len(phase_names)):
                phase_mask = phase_labels == j
                phase_counts.append(np.sum(cluster_mask & phase_mask))
            
            # Plot pie chart if there are any samples
            if sum(phase_counts) > 0:
                # Only plot non-empty slices but maintain color mapping
                non_empty_indices = [j for j, count in enumerate(phase_counts) if count > 0]
                sizes = [phase_counts[j] for j in non_empty_indices]
                non_empty_colors = [colors[j] for j in non_empty_indices]
                
                ax.pie(sizes, colors=non_empty_colors, autopct='%1.1f%%', startangle=90)
            else:
                ax.text(0.5, 0.5, "Empty Cluster", ha='center', va='center')
            
            ax.set_title(f'Cluster {cluster} (n={np.sum(cluster_mask)})')
        
        fig.legend(legend_handles, phase_names, loc='lower center', 
                  ncol=min(len(phase_names), 5))
        
        if title is None:
            title = 'Phase Composition of Each Cluster'
        fig.suptitle(title)
        
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])
        self.figures['cluster_composition'] = fig
        return fig
    
    def plot_silhouette_scores(self,
                             results: Dict[int, Dict[str, Any]],
                             title: Optional[str] = None) -> Figure:
        """
        Plot silhouette scores for different numbers of clusters.
        
        Args:
            results: Dictionary mapping n_clusters to results (from find_optimal_clusters)
        """
        n_values = sorted(results.keys())
        scores = [results[n]['silhouette_score'] for n in n_values]
        
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Plot line and find optimal number of clusters
        ax.plot(n_values, scores, 'o-', linewidth=2, markersize=8)
        optimal_n = max(n_values, key=lambda n: results[n]['silhouette_score'])
        ax.plot(optimal_n, results[optimal_n]['silhouette_score'], 'ro', markersize=12, 
               label=f'Optimal: {optimal_n} clusters')
        
        ax.set_xlabel('Number of Clusters')
        ax.set_ylabel('Silhouette Score')
        if title is None:
            title = 'Silhouette Scores for Different Cluster Counts'
        ax.set_title(title)
        ax.set_xticks(n_values)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        self.figures['silhouette_scores'] = fig
        return fig
