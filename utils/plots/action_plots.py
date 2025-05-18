"""
TODO: Needs refactoring. esp with base feature extractor

Plotting tools to analyze actions

used in base_action_analysis and output_comparison.ipynb
"""
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from utils import joint_names
from utils.base_feature_extractor import extract_pos_vel_from_obs_list


def plot_action_heatmap(mu_strong, mu_weak, mu_medium, batch_idx=0, title="Action Difference Heatmap"):
    """
    Create a heatmap showing differences between model actions over time.
    
    Args:
        mu_strong: Strong model's actions (time, batch, action_dim)
        mu_weak: Weak model's actions (time, batch, action_dim)
        mu_medium: Medium model's actions (time, batch, action_dim)
        batch_idx: Which batch to visualize
        title: Plot title
    """
    # Extract a single batch
    strong_actions = mu_strong[:, batch_idx, :]
    weak_actions = mu_weak[:, batch_idx, :]
    medium_actions = mu_medium[:, batch_idx, :]
    
    # Calculate differences
    weak_diff = strong_actions - weak_actions
    medium_diff = strong_actions - medium_actions
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Find global min/max for consistent color scaling
    vmin = min(weak_diff.min(), medium_diff.min())
    vmax = max(weak_diff.max(), medium_diff.max())
    
    # Plot heatmaps
    im1 = ax1.imshow(weak_diff.T, aspect='auto', cmap='coolwarm', 
                    interpolation='none', vmin=vmin, vmax=vmax)
    ax1.set_title('Strong vs Weak Action Differences')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Action Dimension')
    ax1.set_yticks(range(9))
    ax1.set_yticklabels([
        "Base Rotation", "Shoulder Joint", "Elbow Joint", 
        "Forearm Rotation", "Wrist Bend", "Wrist Rotation", 
        "End Effector Rotation", "Gripper Finger 1", "Gripper Finger 2"
    ])
    
    im2 = ax2.imshow(medium_diff.T, aspect='auto', cmap='coolwarm', 
                    interpolation='none', vmin=vmin, vmax=vmax)
    ax2.set_title('Strong vs Medium Action Differences')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Action Dimension')
    ax2.set_yticks(range(9))
    ax2.set_yticklabels([
        "Base Rotation", "Shoulder Joint", "Elbow Joint", 
        "Forearm Rotation", "Wrist Bend", "Wrist Rotation", 
        "End Effector Rotation", "Gripper Finger 1", "Gripper Finger 2"
    ])
    
    # Add colorbars
    fig.colorbar(im1, ax=ax1, label='Action Difference')
    fig.colorbar(im2, ax=ax2, label='Action Difference')
    
    plt.tight_layout()
    plt.suptitle(title, fontsize=16)
    plt.subplots_adjust(top=0.92)
    plt.show()

def plot_action_trajectories(
        mu_list,
        obs_list=None,
        action_dims=None, 
        actor_idx=None, 
        labels=None, 
        colors=None
    ):
    """
    Plot action trajectories for selected dimensions across multiple models.
    
    Args:
        mu_list: List of model actions, each with shape (time, batch, action_dim)
        obs_list: List of model observations, each with shape (time, batch, obs_dim)
                  If provided, will plot position and velocity data alongside actions.
        action_dims: Which action dimensions to plot. Default is all.
        actor_idx: Which actor to visualize. If None, averages across all.
        labels: List of labels for each model. If None, uses default labels.
        colors: List of colors for each model. If None, uses default colors.
    """
    num_actions = len(joint_names)
    if action_dims is None:
        action_dims = list(range(num_actions))
    
    # Default labels and colors if not provided
    if labels is None:
        labels = [f'Model {i+1}' for i in range(len(mu_list))]
    if colors is None:
        default_colors = ['g', 'b', 'r', 'c', 'm', 'y', 'k']  # Default color cycle
    
        # Create color cycle if we have more models than colors
        color_cycle = cycle(default_colors)
        colors = [next(color_cycle) for _ in range(len(mu_list))]
    
    # Extract position and velocity from observations if provided
    positions_list = None
    velocities_list = None
    if obs_list is not None:
        positions_list, velocities_list = extract_pos_vel_from_obs_list(obs_list, num_actions)
        
    # Process actions based on batch_idx
    if actor_idx is not None:
        # Extract specific batch
        actions_list = [mu[:, actor_idx, :] for mu in mu_list]
        
        # Process observations if provided
        if positions_list is not None:
            positions_list = [pos[:, actor_idx, :] for pos in positions_list]
            velocities_list = [vel[:, actor_idx, :] for vel in velocities_list]
    else:
        # Average across batch dimension
        actions_list = [np.mean(mu, axis=1) for mu in mu_list]
        
        # Process observations if provided
        if positions_list is not None:
            positions_list = [np.mean(pos, axis=1) for pos in positions_list]
            velocities_list = [np.mean(vel, axis=1) for vel in velocities_list]
    
    # Calculate timesteps for each model
    time_steps_list = [np.arange(len(actions)) for actions in actions_list]
    max_timesteps = max(len(time_steps) for time_steps in time_steps_list)
    
    # Create figure with subplots for each action dimension
    fig, axes = plt.subplots(len(action_dims), 1, figsize=(12, 4*len(action_dims)), sharex=True)
    
    # If only one dimension, wrap axes in a list
    if len(action_dims) == 1:
        axes = [axes]
    
    # Plot each selected action dimension
    for i, dim in enumerate(action_dims):
        ax = axes[i]
        
        # Plot each model's trajectory and observations
        for idx, (actions, time_steps, label, color) in enumerate(zip(actions_list, time_steps_list, labels, colors)):
            # Plot action as solid line
            ax.plot(time_steps, actions[:, dim], f'{color}-', label=f"{label} (Action)")
            
            # Plot position and velocity if available
            if positions_list is not None:
                # Plot position as dashed line
                ax.plot(time_steps, positions_list[idx][:, dim], f'{color}--', label=f"{label} (Position)")
                
                # Plot velocity as dotted line
                ax.plot(time_steps, velocities_list[idx][:, dim], f'{color}:', label=f"{label} (Velocity)")
        
        # Add horizontal line at y=0
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # Set y-axis limits to be centered around 0
        y_data = []
        for actions in actions_list:
            y_data.append(actions[:, dim])
        
        if positions_list is not None:
            for pos, vel in zip(positions_list, velocities_list):
                y_data.append(pos[:, dim])
                y_data.append(vel[:, dim])
        
        y_data = np.concatenate(y_data)
        y_max = np.max(np.abs(y_data))
        ax.set_ylim(-y_max*1.1, y_max*1.1)
        
        # Set x-axis limits to match the longest trajectory
        ax.set_xlim(-0.1, max_timesteps-0.9)  # Add small padding
        
        ax.set_title(f'Action Dimension {dim}: {joint_names[dim]}')
        ax.set_ylabel('Action Value')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Show x-axis labels on all subplots
        ax.tick_params(labelbottom=True)
    
    axes[-1].set_xlabel('Time Step')
    plt.tight_layout()
    plt.show()

def plot_action_magnitudes(mu_strong, mu_weak, mu_medium, batch_idx=0):
    """
    Plot L2 norm of action vectors over time.
    
    Args:
        mu_strong: Strong model's actions (time, batch, action_dim)
        mu_weak: Weak model's actions (time, batch, action_dim)
        mu_medium: Medium model's actions (time, batch, action_dim)
        batch_idx: Which batch to visualize
    """
    # Extract a single batch
    strong_actions = mu_strong[:, batch_idx, :]
    weak_actions = mu_weak[:, batch_idx, :]
    medium_actions = mu_medium[:, batch_idx, :]
    
    # Calculate L2 norms
    strong_magnitudes = np.linalg.norm(strong_actions, axis=1)
    weak_magnitudes = np.linalg.norm(weak_actions, axis=1)
    medium_magnitudes = np.linalg.norm(medium_actions, axis=1)
    
    # Plot
    plt.figure(figsize=(12, 6))
    time_steps = np.arange(len(strong_actions))
    
    plt.plot(time_steps, strong_magnitudes, 'g-', label='Strong Model')
    plt.plot(time_steps, medium_magnitudes, 'b-', label='Medium Model')
    plt.plot(time_steps, weak_magnitudes, 'r-', label='Weak Model')
    
    plt.title('Action Magnitude Comparison')
    plt.xlabel('Time Step')
    plt.ylabel('Action Magnitude (L2 Norm)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
