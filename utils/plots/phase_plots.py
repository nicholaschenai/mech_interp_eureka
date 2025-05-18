"""
plotting with phase information
"""
import matplotlib.pyplot as plt
import numpy as np

from itertools import cycle

from utils import joint_names

def prepare_joint_velocity_comparison_data(
        obs_list, 
        max_timesteps=100,
        epsilon=1e-5,
        num_actions=9
    ):
    """
    Prepares joint velocity vs distance data for multiple models during the approaching phase.
    This function handles all the data extraction and phase mask application for multiple models
    to facilitate comparison plotting.
    
    Parameters:
    -----------
    obs_list : list of ndarrays
        List of observation arrays, each with shape (time, batch, obs_dim) for different models
    max_timesteps : int, optional
        Maximum number of timesteps to consider, defaults to 100
    epsilon : float, optional
        Threshold for considering drawer position/velocity as zero/positive
    num_actions : int, optional
        Number of action dimensions (typically 9 for Franka robot)
        
    Returns:
    --------
    dict
        Dictionary containing processed data for each model:
        {
            'filtered_velocities_list': List of filtered joint velocities arrays,
            'filtered_distances_list': List of filtered distances arrays,
            'approaching_counts': List of counts of timesteps in approaching phase,
            'model_masks': List of booleans indicating if model has data in this phase
        }
    """
    # Initialize output containers
    filtered_velocities_list = []
    filtered_distances_list = []
    approaching_counts = []
    model_masks = []  # Indicates which models have valid data for this phase
    
    # Process each model's observations
    for model_idx, obs in enumerate(obs_list):
        # Extract observation components
        _, velocities, rel_pos_to_handle, drawer_pos, drawer_vel = segment_observation(obs, num_actions)

        # Get phase masks
        approaching_mask, approaching_count, _, _, _, _ = get_all_phase_masks(
            drawer_pos, drawer_vel, epsilon, max_timesteps
        )
        
        # Check if this model has meaningful data for the approaching phase
        has_approaching_data = approaching_count > 0
        model_masks.append(has_approaching_data)
        approaching_counts.append(approaching_count)
        
        if has_approaching_data:
            # Apply mask to get filtered data
            filtered_velocities, filtered_distances = apply_approaching_mask(
                velocities, rel_pos_to_handle, approaching_mask
            )
            
            filtered_velocities_list.append(filtered_velocities)
            filtered_distances_list.append(filtered_distances)
        else:
            # Add empty arrays if no data for this phase
            filtered_velocities_list.append(np.array([]))
            filtered_distances_list.append(np.array([]))
    
    return {
        'filtered_velocities_list': filtered_velocities_list,
        'filtered_distances_list': filtered_distances_list,
        'approaching_counts': approaching_counts,
        'model_masks': model_masks
    }

def compare_joint_velocities_across_models(
        obs_list,
        labels=None,
        colors=None,
        joint_dims=None,
        max_timesteps=100,
        num_bins=20,
        epsilon=1e-5,
        line_styles=None,
        line_width=2.5,
        auto_scale_y=True
    ):
    """
    Create comparison plots of mean joint velocities vs distance for multiple models during the approach phase.
    Each joint gets its own subplot with different models represented by different colors.

    Parameters:
    -----------
    obs_list : list of ndarrays
        List of observation arrays, each with shape (time, batch, obs_dim) for different models
    labels : list of str, optional
        Labels for each model, defaults to "Model 1", "Model 2", etc.
    colors : list of str, optional
        Colors for each model, defaults to standard colors
    joint_dims : list of int, optional
        Which joint dimensions to plot. If None, plots all joints
    max_timesteps : int, optional
        Maximum number of timesteps to consider, defaults to 100
    num_bins : int, optional
        Number of distance bins for statistics calculation
    epsilon : float, optional
        Threshold for phase detection
    line_styles : list of str, optional
        Line styles for each model, defaults to solid lines
    line_width : float, optional
        Width of the plotted lines, defaults to 2.5
    auto_scale_y : bool, optional
        Whether to automatically scale y-axis based on data range, defaults to True
    """
    # Prepare default values
    num_models = len(obs_list)
    num_actions = len(joint_names)

    if joint_dims is None:
        joint_dims = list(range(num_actions))

    if labels is None:
        labels = [f'Model {i+1}' for i in range(num_models)]

    if colors is None:
        default_colors = ['g', 'b', 'r', 'c', 'm', 'y', 'orange']
        color_cycle = cycle(default_colors)
        colors = [next(color_cycle) for _ in range(num_models)]

    if line_styles is None:
        line_styles = ['-'] * num_models

    # Get the prepared data for all models
    data = prepare_joint_velocity_comparison_data(
        obs_list,
        max_timesteps=max_timesteps,
        epsilon=epsilon
    )

    filtered_velocities_list = data['filtered_velocities_list']
    filtered_distances_list = data['filtered_distances_list']
    model_masks = data['model_masks']
    approaching_counts = data['approaching_counts']

    # Create figure with subplots for each joint
    fig, axes = plt.subplots(len(joint_dims), 1, figsize=(14, 4*len(joint_dims)), sharex=True)

    # If only one joint, wrap axes in a list
    if len(joint_dims) == 1:
        axes = [axes]

    # Collect all means for y-axis scaling if auto_scale_y is True
    all_means_by_joint = {}
    if auto_scale_y:
        for joint_idx in joint_dims:
            all_means_by_joint[joint_idx] = []

    # Plot each joint
    for i, joint_idx in enumerate(joint_dims):
        ax = axes[i]

        # Add horizontal line at y=0 first so it's behind the data
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.4, zorder=1)

        # Plot each model's data for this joint
        for model_idx in range(num_models):
            # Skip models with no data in this phase
            if not model_masks[model_idx]:
                continue

            filtered_velocities = filtered_velocities_list[model_idx]
            filtered_distances = filtered_distances_list[model_idx]

            if len(filtered_velocities) == 0:
                continue

            # Get velocities for this joint
            joint_vel = filtered_velocities[:, joint_idx]

            # Define bin edges based on distance range for this model
            min_dist = np.min(filtered_distances)
            max_dist = np.max(filtered_distances)
            bin_edges = np.linspace(min_dist, max_dist, num_bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            means = []

            # Calculate mean for each bin
            for bin_idx in range(num_bins):
                lower_bound = bin_edges[bin_idx]
                upper_bound = bin_edges[bin_idx+1]

                # Find velocities in this distance bin
                mask = (filtered_distances >= lower_bound) & (filtered_distances < upper_bound)
                bin_velocities = joint_vel[mask]

                # Only compute mean if there are enough samples
                if len(bin_velocities) > 0:
                    mean = np.mean(bin_velocities)
                else:
                    # If no samples, use NaN
                    mean = np.nan

                means.append(mean)

            # Convert to numpy array
            means = np.array(means)

            # Add to collection for y-axis scaling if needed
            if auto_scale_y:
                valid_means = means[~np.isnan(means)]
                if len(valid_means) > 0:
                    all_means_by_joint[joint_idx].append(valid_means)

            # Plot mean line
            ax.plot(bin_centers, means,
                    color=colors[model_idx],
                    linestyle=line_styles[model_idx],
                    label=f"{labels[model_idx]} (n={approaching_counts[model_idx]})",
                    linewidth=line_width,
                    zorder=2)

        # Set y-axis limits - either auto-scaled or fixed
        if auto_scale_y and all_means_by_joint[joint_idx]:
            # Combine all valid mean values for this joint
            all_data = np.concatenate(all_means_by_joint[joint_idx])
            if len(all_data) > 0:
                # Calculate max absolute value and scale by 1.1 for padding
                y_max = np.max(np.abs(all_data))
                ax.set_ylim(-y_max*1.1, y_max*1.1)

        # Set title and labels
        ax.set_title(f'Joint {joint_idx}: {joint_names[joint_idx]}')
        ax.set_ylabel('Joint Velocity')
        ax.grid(True, alpha=0.3, zorder=0)  # Ensure grid is behind everything
        ax.legend()

        # Show x-axis labels on all subplots
        ax.tick_params(labelbottom=True)

    # Set common x-axis label
    axes[-1].set_xlabel('Distance to Handle')

    plt.suptitle('Joint Velocities vs Distance During Approach Phase', fontsize=16)

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Make room for suptitle
    plt.show()


def plot_joint_velocities_stats_by_distance(
        joint_velocities,
        rel_pos_to_handle,
        approaching_mask,
        num_bins=20,
        percentile_range=(10, 90),
        show_legend=True
    ):
    """
    Plot the mean and percentile range of joint velocities with respect 
    to relative distance to handle, binned by distance.

    Parameters:
    -----------
    joint_velocities : ndarray
        Joint velocities array of shape (time, batch, num_actions)
    rel_pos_to_handle : ndarray
        Relative coordinates to handle of shape (time, batch, 3)
    approaching_mask : ndarray
        Boolean mask of shape (time, batch) for the approaching phase
    num_bins : int, optional
        Number of distance bins
    percentile_range : tuple of (low, high), optional
        Percentile range to display, defaults to (5, 95) which captures 90% of data points
    show_legend : bool, optional
        Whether to show the legend, defaults to True
    """
    # Get filtered velocities and distances
    filtered_velocities, filtered_distances = apply_approaching_mask(
        joint_velocities,
        rel_pos_to_handle,
        approaching_mask
    )

    # Define bin edges based on distance range
    min_dist = np.min(filtered_distances)
    max_dist = np.max(filtered_distances)
    bin_edges = np.linspace(min_dist, max_dist, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Define colors for each joint
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'orange', 'purple', 'brown']

    # Create figure
    plt.figure(figsize=(14, 10))

    # Unpack percentile range
    low_percentile, high_percentile = percentile_range

    # Loop through each joint
    for joint_idx in range(filtered_velocities.shape[1]):
        joint_vel = filtered_velocities[:, joint_idx]

        means = []
        percentile_lower = []
        percentile_upper = []

        # Calculate statistics for each bin
        for i in range(num_bins):
            lower_bound = bin_edges[i]
            upper_bound = bin_edges[i+1]

            # Find velocities in this distance bin
            mask = (filtered_distances >= lower_bound) & (filtered_distances < upper_bound)
            bin_velocities = joint_vel[mask]

            # Only compute statistics if there are enough samples
            if len(bin_velocities) > 1:
                mean = np.mean(bin_velocities)
                # Calculate percentile range to show distribution
                lower = np.percentile(bin_velocities, low_percentile)
                upper = np.percentile(bin_velocities, high_percentile)
            else:
                # If no or too few samples, use NaN
                mean, lower, upper = np.nan, np.nan, np.nan

            means.append(mean)
            percentile_lower.append(lower)
            percentile_upper.append(upper)

        # Convert to numpy arrays
        means = np.array(means)
        percentile_lower = np.array(percentile_lower)
        percentile_upper = np.array(percentile_upper)

        # Plot mean line
        plt.plot(bin_centers, means, color=colors[joint_idx],
                 label=joint_names[joint_idx], linewidth=2)

        # Plot percentile range as shaded region
        valid_idx = ~np.isnan(means)
        if np.any(valid_idx):
            plt.fill_between(
                bin_centers[valid_idx],
                percentile_lower[valid_idx],
                percentile_upper[valid_idx],
                color=colors[joint_idx],
                alpha=0.2
            )

    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    range_str = f"{low_percentile}th-{high_percentile}th percentile"
    plt.title(f'Joint Velocities vs Distance to Handle During Approach Phase\n(Shaded area: {range_str})')
    plt.xlabel('Distance to Handle')
    plt.ylabel('Joint Velocity')
    plt.grid(True, alpha=0.3)

    if show_legend:
        plt.legend()

    plt.tight_layout()
    plt.show()


def plot_joint_velocities_vs_distance(
        joint_velocities, 
        rel_pos_to_handle, 
        approaching_mask, 
        show_legend=True
    ):
    """
    Create a scatter plot of joint velocities with respect to relative distance to handle.
    This is to study the approach phase of the task.
    Each joint is represented by a different color.
    
    Parameters:
    -----------
    filtered_velocities : ndarray
        Filtered joint velocities of shape (num_filtered, num_actions)
    filtered_distances : ndarray
        Filtered distances to handle of shape (num_filtered,)
    show_legend : bool, optional
        Whether to show the legend, defaults to True
    """
    filtered_velocities, filtered_distances = apply_approaching_mask(
        joint_velocities, 
        rel_pos_to_handle, 
        approaching_mask
    )
    plt.figure(figsize=(12, 8))
    
    # Define colors for each joint
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'orange', 'purple', 'brown']
    
    # Loop through each joint and create scatter plot
    for joint_idx in range(filtered_velocities.shape[1]):
        joint_vel = filtered_velocities[:, joint_idx]
        plt.scatter(
            filtered_distances, 
            joint_vel, 
            c=colors[joint_idx], 
            alpha=0.2, 
            label=joint_names[joint_idx],
            edgecolors='none'  # Removes edges to make dense plots more readable
        )
    
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.title('Joint Velocities vs Distance to Handle During Approach Phase')
    plt.xlabel('Distance to Handle')
    plt.ylabel('Joint Velocity')
    plt.grid(True, alpha=0.3)
    
    if show_legend:
        plt.legend()
    
    plt.tight_layout()
    plt.show()
