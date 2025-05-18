"""
Extract phases of trajectories (basically a mask)

The phase types are based on human observation of the data.

There are roughly 3 phases:
- Approach phase: Arm is approaching the handle
- Opening: Drawer is being pulled open
- Decelerating: Drawer is decelerating

The functions below classify into these phases via rules defined by human observation.

Fuctions also counts data points in each phase in case it doesnt apply (e.g. weak model doesnt manage to pull the drawer)

# TODO: visualize open drawer phase by, after filter, scatterplot of joint velocities wrt drawer_velocity. each joint is its own plot

# TODO: phase 3: deceleration. characterized by positive drawer_velocity, and negative acceleration. visualize this same as open drawer phase
"""
import numpy as np
from typing import Dict, Any


def _prepare_mask_array(time_steps, batch_size, max_timesteps=100):
    """
    Helper function to create a mask array of the correct shape with the first and last
    timesteps set to False and timesteps beyond max_timesteps set to False.
    
    Parameters:
    -----------
    time_steps : int
        Total number of timesteps
    batch_size : int
        Batch size
    max_timesteps : int
        Maximum number of timesteps to consider
        
    Returns:
    --------
    mask : ndarray
        Boolean mask of shape (time_steps, batch_size) with False for filtered timesteps
    """
    mask = np.ones((time_steps, batch_size), dtype=bool)
    
    # Set first and last timesteps to False to remove endpoint artifacts
    if time_steps > 0:
        mask[0, :] = False
    if time_steps > 1:
        mask[-1, :] = False
    
    # Set timesteps beyond max_timesteps to False
    # since this is visually confirmed (anything beyond that is the arm fidgeting aimlessly)
    mask[max_timesteps:, :] = False
    
    return mask

def get_approaching_handle_mask(drawer_position, drawer_velocity, base_mask, epsilon=1e-5):
    """
    Internal function to create a binary mask for the phase of approaching the handle.
    
    Parameters:
    -----------
    drawer_position : ndarray
        Drawer position array of shape (time, batch)
    drawer_velocity : ndarray
        Drawer velocity array of shape (time, batch)
    base_mask : ndarray
        Base mask with time filtering already applied
    epsilon : float, optional
        Threshold for considering drawer position and velocity as zero
        
    Returns:
    --------
    tuple
        (mask, count) where:
        - mask: Boolean mask for the approaching phase
        - count: Number of timesteps in this phase
    """
    # Phase 1: Both drawer position and velocity are approximately zero
    phase_mask = base_mask & (np.abs(drawer_position) < epsilon) & (np.abs(drawer_velocity) < epsilon)
    
    # Count timesteps in this phase
    phase_count = np.sum(phase_mask)
    
    return phase_mask, phase_count

def apply_mask_to_joint_velocities(joint_velocities, phase_mask):
    """
    Applies a phase mask to joint velocities and flattens the time and batch dimensions.
    
    Parameters:
    -----------
    joint_velocities : ndarray
        Joint velocities array of shape (time, batch, num_actions)
    phase_mask : ndarray
        Boolean mask of shape (time, batch) for a specific phase
        
    Returns:
    --------
    filtered_velocities : ndarray
        Filtered joint velocities of shape (num_filtered, num_actions)
    """
    # Create flattened arrays for indexing
    flat_velocities = joint_velocities.reshape(-1, joint_velocities.shape[2])  # shape: (time*batch, num_actions)
    
    # Create flattened mask for indexing
    flat_mask = phase_mask.reshape(-1)  # shape: (time*batch,)
    
    # Apply mask to get filtered data
    filtered_velocities = flat_velocities[flat_mask]  # shape: (num_filtered, num_actions)
    
    return filtered_velocities

# TODO: deprecate in favor of activation_dataset.filter_by_phase?
def apply_approaching_mask(joint_velocities, rel_pos_to_handle, phase_mask):
    """
    Applies the approaching handle phase mask to joint velocities and rel_pos_to_handle, 
    and prepares data for plotting.
    
    Parameters:
    -----------
    joint_velocities : ndarray
        Joint velocities array of shape (time, batch, num_actions)
    rel_pos_to_handle : ndarray
        Relative coordinates to handle of shape (time, batch, 3)
    phase_mask : ndarray
        Boolean mask of shape (time, batch) for the approaching phase
        
    Returns:
    --------
    tuple
        (filtered_velocities, filtered_distances) where:
        - filtered_velocities: Filtered joint velocities of shape (num_filtered, num_actions)
        - filtered_distances: Filtered distances to handle of shape (num_filtered,)
    """
    # Calculate distance to handle using norm
    distances = np.linalg.norm(rel_pos_to_handle, axis=2)  # shape: (time, batch)
    
    # Apply mask to joint velocities using the generic function
    filtered_velocities = apply_mask_to_joint_velocities(joint_velocities, phase_mask)
    
    # Apply mask to distances
    flat_distances = distances.reshape(-1)  # shape: (time*batch,)
    flat_mask = phase_mask.reshape(-1)  # shape: (time*batch,)
    filtered_distances = flat_distances[flat_mask]  # shape: (num_filtered,)
    
    return filtered_velocities, filtered_distances

def get_acceleration(drawer_velocity: np.ndarray) -> np.ndarray:
    # Calculate acceleration (change in velocity)
    acceleration = np.zeros_like(drawer_velocity)
    if drawer_velocity.shape[0] > 1:
        acceleration[1:] = drawer_velocity[1:] - drawer_velocity[:-1]
    return acceleration

def get_opening_drawer_mask(drawer_velocity, base_mask, epsilon=1e-5):
    """
    Internal function to create a binary mask for the phase of exerting opening force on the handle.
    
    Parameters:
    -----------
    drawer_velocity : ndarray
        Drawer velocity array of shape (time, batch)
    base_mask : ndarray
        Base mask with time filtering already applied
    epsilon : float, optional
        Threshold for considering drawer velocity as positive
        
    Returns:
    --------
    tuple
        (mask, count) where:
        - mask: Boolean mask for the opening phase
        - count: Number of timesteps in this phase
    """
    acceleration = get_acceleration(drawer_velocity)
    
    # Phase 2: Positive drawer velocity and positive acceleration
    phase_mask = base_mask & (drawer_velocity > epsilon) & (acceleration > 0)
    
    # Count timesteps in this phase
    phase_count = np.sum(phase_mask)
    
    return phase_mask, phase_count

def get_deceleration_mask(drawer_velocity, base_mask, epsilon=1e-5):
    """
    Internal function to create a binary mask for the deceleration phase.
    
    Parameters:
    -----------
    drawer_velocity : ndarray
        Drawer velocity array of shape (time, batch)
    base_mask : ndarray
        Base mask with time filtering already applied
    epsilon : float, optional
        Threshold for considering drawer velocity as positive
        
    Returns:
    --------
    tuple
        (mask, count) where:
        - mask: Boolean mask for the deceleration phase
        - count: Number of timesteps in this phase
    """
    acceleration = get_acceleration(drawer_velocity)
    
    # Phase 3: Positive drawer velocity but negative acceleration
    phase_mask = base_mask & (drawer_velocity > epsilon) & (acceleration < 0)
    
    # Count timesteps in this phase
    phase_count = np.sum(phase_mask)
    
    return phase_mask, phase_count

def get_all_phase_masks(drawer_position: np.ndarray, drawer_velocity: np.ndarray, epsilon: float = 1e-5, max_timesteps: int = 100):
    """
    Creates masks for all phases of the drawer opening task.
    
    Parameters:
    -----------
    drawer_position : ndarray
        Drawer position array of shape (time, batch)
    drawer_velocity : ndarray
        Drawer velocity array of shape (time, batch)
    epsilon : float, optional
        Threshold for considering drawer position/velocity as zero/positive
    max_timesteps : int, optional
        Maximum number of timesteps to consider, defaults to 100
    
    Returns:
    --------
    tuple
        (approaching_mask, approaching_count, opening_mask, opening_count, 
         deceleration_mask, deceleration_count)
    """
    # Create base mask for time filtering - only do this once for all masks
    time_steps, batch_size = drawer_position.shape
    base_mask = _prepare_mask_array(time_steps, batch_size, max_timesteps)
    
    # Get masks for each phase using the same base mask
    approaching_mask, approaching_count = get_approaching_handle_mask(
        drawer_position, drawer_velocity, base_mask, epsilon
    )
    
    opening_mask, opening_count = get_opening_drawer_mask(
        drawer_velocity, base_mask, epsilon
    )
    
    deceleration_mask, deceleration_count = get_deceleration_mask(
        drawer_velocity, base_mask, epsilon
    )
    
    return (
        approaching_mask, approaching_count, 
        opening_mask, opening_count, 
        deceleration_mask, deceleration_count
    )


class PhaseDetector:
    def __init__(self, epsilon: float = 1e-5, max_timesteps: int = 100):
        """
        Parameters:
        -----------
        epsilon : float
            Threshold value for considering position/velocity as zero or positive
        max_timesteps : int
            Maximum number of timesteps to consider in the trajectory
        """
        self.epsilon = epsilon
        self.max_timesteps = max_timesteps
        
    def detect_phases(self, features: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Parameters:
        -----------
        features : Dict[str, np.ndarray]
            Dictionary containing features extracted from observations.
            Must contain 'drawer_position' and 'drawer_velocity' keys.
        """
        # Extract required features
        drawer_position = features['drawer_position']
        drawer_velocity = features['drawer_velocity']
        
        # Create base mask
        time_steps, batch_size = drawer_position.shape
        base_mask = _prepare_mask_array(time_steps, batch_size, self.max_timesteps)
        
        # Get phase masks
        approaching_mask, approaching_count = get_approaching_handle_mask(
            drawer_position, drawer_velocity, base_mask, self.epsilon
        )
        
        opening_mask, opening_count = get_opening_drawer_mask(
            drawer_velocity, base_mask, self.epsilon
        )
        
        deceleration_mask, deceleration_count = get_deceleration_mask(
            drawer_velocity, base_mask, self.epsilon
        )
        
        # Create result dictionary
        phase_masks = {
            'approaching': approaching_mask,
            'opening': opening_mask,
            'deceleration': deceleration_mask
        }
        # Print counts for each phase
        print("\nPhase counts:")
        print(f"Approaching phase: {approaching_count} timesteps")
        print(f"Opening phase: {opening_count} timesteps")
        print(f"Deceleration phase: {deceleration_count} timesteps")
        
        return phase_masks
    
    # def apply_mask_to_data(self, data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    #     """
    #     Parameters:
    #     -----------
    #     data : np.ndarray
    #         Data array with shape (..., time_steps, batch_size, ...)
    #     mask : np.ndarray
    #         Boolean mask with shape (time_steps, batch_size)
    #     """
    #     # Handle different data dimensions
    #     if data.ndim == 2:  # (time_steps, batch_size)
    #         flat_data = data.reshape(-1)
    #         flat_mask = mask.reshape(-1)
    #         return flat_data[flat_mask]
            
    #     elif data.ndim == 3:  # (time_steps, batch_size, features)
    #         flat_data = data.reshape(-1, data.shape[2])
    #         flat_mask = mask.reshape(-1)
    #         return flat_data[flat_mask]
            
    #     else:
    #         raise ValueError(f"Unsupported data shape with {data.ndim} dimensions")
