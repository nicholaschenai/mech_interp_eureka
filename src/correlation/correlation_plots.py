import matplotlib.pyplot as plt
import numpy as np

from typing import List, Dict

from .correlation_analyzer import CorrelationAnalyzer

from utils.base_feature_extractor import get_all_feature_keys


# TODO: plotting is abit off for aspect ratio, fix
def plot_feature_correlation_heatmap(
    analyzer: CorrelationAnalyzer, 
    layer_name: str,
    feature_keys: List[str] = None,
    figsize=(15, 10),
    cmap='coolwarm',
    sort_neurons=True
):
    """
    Compute and plot a heatmap of correlations between neurons in a layer and all 1D features.
    
    Args:
        analyzer: CorrelationAnalyzer instance
        layer_name: Name of layer to analyze
        feature_keys: List of feature keys to analyze (uses default list if None)
        figsize: Size of the figure (width, height)
        cmap: Colormap for the heatmap
        sort_neurons: Whether to cluster neurons by correlation similarity
    """
    if feature_keys is None:
        feature_keys = get_all_feature_keys()
    
    print(f"Computing correlations for features...")
    
    # Get correlation matrix
    corr_matrix = analyzer.get_layer_correlation_matrix(layer_name, feature_keys, sort_neurons)
    
    # Get dimensions
    num_neurons, num_features = corr_matrix.shape
    
    # Prepare feature labels (shortened if needed)
    feature_labels = [f.replace('joint', 'j').replace('position', 'pos').replace('velocity', 'vel') 
                    for f in feature_keys]
    
    # Calculate appropriate figure size to maintain square cells
    # Base the figure width on the number of features
    width = max(8, min(20, num_features * 0.8))
    # Adjust height to maintain square cells based on the ratio of neurons to features
    height = width * (num_neurons / num_features)
    # Cap the height to a reasonable value if needed
    height = min(height, 30)
    
    # Create figure and axes with adjusted size
    fig, ax = plt.subplots(figsize=(width, height))
    
    # Plot heatmap using imshow with equal aspect ratio for square cells
    im = ax.imshow(
        corr_matrix,
        cmap=cmap,
        vmin=-1, 
        vmax=1,
        aspect='equal'  # Ensure the cells are square
    )
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, label='Correlation')
    
    # Add x-axis ticks and labels
    ax.set_xticks(np.arange(len(feature_labels)))
    ax.set_xticklabels(feature_labels)
    
    # Don't show y-axis ticks/labels (too many neurons)
    ax.set_yticks([])
    
    # Add title and labels
    ax.set_title(f'Neuron-Feature Correlations for Layer: {layer_name}', fontsize=14)
    ax.set_xlabel('Features', fontsize=12)
    ax.set_ylabel('Neurons', fontsize=12)
    
    # Rotate x-axis labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Make plot more compact
    plt.tight_layout()
    
    # Add a note about sorting if applicable
    if sort_neurons:
        plt.figtext(0.01, 0.01, 'Note: Neurons are clustered by correlation pattern similarity', 
                   fontsize=8, style='italic')
    
    # Show the plot
    plt.show()
    
    return corr_matrix

def plot_model_comparison_sensitivities(
    analyzers_dict: Dict[str, CorrelationAnalyzer],
    input_features: List[str],
    layer_name='actor_mlp_1',
    figsize=(12, 15),
    colors={'strong': 'green', 'medium': 'blue', 'weak': 'red'},
    save_path=None
):
    """
    Create visualization comparing feature sensitivities across models.
    
    Args:
        analyzers_dict: Dict mapping model names to CorrelationAnalyzer instances
        input_features: List of feature names to analyze
        layer_name: Name of the layer to analyze (default: actor_mlp_1)
        figsize: Size of the figure
        colors: Dict mapping model names to colors for plotting
        save_path: Path to save the figure (optional)
    """
    fig, axes = plt.subplots(len(input_features), 1, figsize=figsize)
    if len(input_features) == 1:
        axes = [axes]  # Handle case of single subplot
        
    # Calculate sensitivities for each model and feature
    results = {}
    
    for model_name, analyzer in analyzers_dict.items():
        results[model_name] = {}
        
        for feature_name in input_features:
            # Use the correlation_analyzer to compute correlations
            corrs = analyzer.compute_correlation(layer_name, feature_name)
            
            # Store results (take absolute values for plotting)
            results[model_name][feature_name] = np.abs(corrs)
    
    # Define fixed y-positions for annotations to avoid overlap
    y_positions = {
        'strong': 0.9,  # Top position for strong model
        'medium': 0.5,  # Middle position for medium model
        'weak': 0.1     # Bottom position for weak model
    }
    
    # Plot histograms for each feature
    for i, feature_name in enumerate(input_features):
        ax = axes[i]
        
        # Set fixed bins between 0 and 1
        bins = np.linspace(0, 1, 11)  # 10 bins from 0 to 1
        
        # Plot histogram for each model with distinct outlines
        for model_name in results:
            sensitivities = results[model_name][feature_name]
            
            # Use step histogram with edge color for better distinction
            ax.hist(sensitivities, bins=bins, alpha=0.25,
                   label=model_name, color=colors[model_name],
                   edgecolor=colors[model_name], linewidth=2,
                   histtype='stepfilled', linestyle='-.')
            
            # Calculate and mark the mean value instead of top 10 neurons
            mean_value = np.mean(sensitivities)
            ax.axvline(mean_value, color=colors[model_name], 
                      linestyle='--', linewidth=2)
            
            # Add text annotation for the mean with fixed position
            y_max = ax.get_ylim()[1]
            y_pos = y_max * y_positions[model_name]
            
            ax.text(mean_value, y_pos, f'{model_name}\nmean: {mean_value:.3f}', 
                   color=colors[model_name], ha='center', va='top',
                   bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
            
        ax.set_title(f'Neuron sensitivity to {feature_name}', fontsize=14)
        ax.set_xlabel('Absolute correlation', fontsize=12)
        ax.set_ylabel('Number of neurons', fontsize=12)
        
        # Set x-axis limits to 0-1 to match bins
        ax.set_xlim(0, 1)
        
        ax.legend()
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return results
