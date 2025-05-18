"""
Plotting utils for observations
TODO: needs refactoring to use BaseFeatureExtractor
"""
import numpy as np
import matplotlib.pyplot as plt

from utils.base_feature_extractor import segment_observation

def plot_to_target_distance(obs, actor_idx, timesteps=500):
    """
    Plot the distance to target, drawer position, and drawer velocity over time.
    
    Parameters:
    -----------
    obs : ndarray
        pre-normalized Observation data with shape (time, batch, obs_dim)
    actor_idx : int
        Index of the actor to plot data for
    timesteps : int, optional
        Number of timesteps to plot, defaults to 500
    """
    # Extract a single actor's observations and reshape for segment_observation
    actor_obs = obs[:, actor_idx:actor_idx+1, :]
    
    # Use segment_observation to extract components (num_actions=9 for Franka robot)
    _, _, rel_pos_to_handle, drawer_pos, drawer_vel = segment_observation(actor_obs, num_actions=9)
    
    # Calculate Euclidean distance to target using numpy's norm function
    to_target_distance = np.linalg.norm(rel_pos_to_handle, axis=2)[:, 0]
    
    # Squeeze out the batch dimension for drawer position and velocity
    drawer_pos = drawer_pos[:, 0]
    drawer_vel = drawer_vel[:, 0]
    
    # Create time steps array
    time_steps = np.arange(min(timesteps, obs.shape[0]))

    # Create a figure with three separate subplots
    fig, axs = plt.subplots(3, 1, figsize=(12, 12))

    # Plot 1: Distance to Target
    axs[0].plot(time_steps, to_target_distance[:timesteps], 'bo', linewidth=2)
    axs[0].set_title('Distance to Target Over Time')
    axs[0].set_ylabel('Distance')
    axs[0].grid(True)

    # Plot 2: Drawer Position
    axs[1].plot(time_steps, drawer_pos[:timesteps], 'ro', linewidth=2)
    axs[1].set_title('Drawer Position Over Time')
    axs[1].set_ylabel('Position')
    axs[1].grid(True)

    # Plot 3: Drawer Velocity
    axs[2].plot(time_steps, drawer_vel[:timesteps], 'go', linewidth=2)
    axs[2].set_title('Drawer Velocity Over Time')
    axs[2].set_xlabel('Time Step')
    axs[2].set_ylabel('Velocity')
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()
