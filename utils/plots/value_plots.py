"""
Tools to plot critic values
"""
import numpy as np
import matplotlib.pyplot as plt


def plot_value_comparison(value_strong, value_weak, value_medium, batch_idx=0):
    """
    Plot value estimates over time for all models.
    
    Args:
        value_strong: Strong model's value estimates (time, batch, 1)
        value_weak: Weak model's value estimates (time, batch, 1)
        value_medium: Medium model's value estimates (time, batch, 1)
        batch_idx: Which batch to visualize
    """
    # Extract a single batch and flatten
    strong_values = value_strong[:, batch_idx, 0]
    weak_values = value_weak[:, batch_idx, 0]
    medium_values = value_medium[:, batch_idx, 0]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    time_steps = np.arange(len(strong_values))
    
    # Plot absolute values
    ax1.plot(time_steps, strong_values, 'g-', label='Strong Model')
    ax1.plot(time_steps, medium_values, 'b-', label='Medium Model')
    ax1.plot(time_steps, weak_values, 'r-', label='Weak Model')
    ax1.set_title('Value Estimates Comparison')
    ax1.set_ylabel('Value Estimate')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot differences
    ax2.plot(time_steps, strong_values - weak_values, 'r-', label='Strong - Weak')
    ax2.plot(time_steps, strong_values - medium_values, 'b-', label='Strong - Medium')
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.set_title('Value Estimate Differences')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Value Difference')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def plot_value_vs_action(mu_strong, value_strong, mu_weak, value_weak, 
                         mu_medium, value_medium, batch_idx=0):
    """
    Create a scatter plot of value estimates vs action magnitudes.
    
    Args:
        mu_strong, value_strong: Strong model's actions and values
        mu_weak, value_weak: Weak model's actions and values
        mu_medium, value_medium: Medium model's actions and values
        batch_idx: Which batch to visualize
    """
    # Extract a single batch
    strong_actions = mu_strong[:, batch_idx, :]
    weak_actions = mu_weak[:, batch_idx, :]
    medium_actions = mu_medium[:, batch_idx, :]
    
    strong_values = value_strong[:, batch_idx, 0]
    weak_values = value_weak[:, batch_idx, 0]
    medium_values = value_medium[:, batch_idx, 0]
    
    # Calculate action magnitudes
    strong_magnitudes = np.linalg.norm(strong_actions, axis=1)
    weak_magnitudes = np.linalg.norm(weak_actions, axis=1)
    medium_magnitudes = np.linalg.norm(medium_actions, axis=1)
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    
    plt.scatter(strong_values, strong_magnitudes, c='g', alpha=0.7, label='Strong Model')
    plt.scatter(medium_values, medium_magnitudes, c='b', alpha=0.7, label='Medium Model')
    plt.scatter(weak_values, weak_magnitudes, c='r', alpha=0.7, label='Weak Model')
    
    plt.title('Value Estimates vs Action Magnitudes')
    plt.xlabel('Value Estimate')
    plt.ylabel('Action Magnitude (L2 Norm)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
