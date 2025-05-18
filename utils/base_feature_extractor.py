"""
Basic features from observation
"""
import numpy as np
from typing import Dict


class BaseFeatureExtractor:
    """Extracts base features from observations for Franka cabinet"""
    
    def __init__(self, num_joints: int = 9):
        """Initialize the feature extractor.
        
        Args:
            num_joints: Number of robot joints (default: 9 for Franka arm with 7 arm joints + 2 fingers)
        """
        self.num_joints = num_joints
        
        # Define indices for different observation components
        self.joint_pos_indices = slice(0, num_joints)
        self.joint_vel_indices = slice(num_joints, 2 * num_joints)
        # Relative coordinates to handle
        self.distance_indices = slice(2 * num_joints, 2 * num_joints + 3)
        self.drawer_pos_index = 2 * num_joints + 3
        self.drawer_vel_index = 2 * num_joints + 4
    
    def extract_distance_features(self, observation: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract distance-related features from observation.
        
        Args:
            observation: Raw observation array of shape [..., obs_dim]
            
        Returns:
            Dictionary with distance features:
                - 'distance_vec': Vector components [x, y, z]
                - 'distance': Magnitude of distance
                - 'distance_x', 'distance_y', 'distance_z': Individual components
        """
        # Extract distance vector components (x, y, z)
        distance_vec = observation[..., self.distance_indices]
        
        # Calculate distance magnitude (Euclidean norm)
        distance = np.linalg.norm(distance_vec, axis=-1)
        
        # Create feature dictionary
        features = {
            'distance_vec': distance_vec,
            'distance': distance,
            'distance_x': distance_vec[..., 0],
            'distance_y': distance_vec[..., 1],
            'distance_z': distance_vec[..., 2]
        }
        
        return features
    
    def extract_joint_features(self, observation: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract joint position and velocity features.
        
        Args:
            observation: Raw observation array of shape [..., obs_dim]
            
        Returns:
            Dictionary with joint features:
                - 'joint_positions': Joint positions
                - 'joint_velocities': Joint velocities
                - Individual joint positions and velocities
        """
        # Extract joint positions and velocities
        joint_positions = observation[..., self.joint_pos_indices]
        joint_velocities = observation[..., self.joint_vel_indices]
        
        # Create feature dictionary
        features = {
            'joint_positions': joint_positions,
            'joint_velocities': joint_velocities
        }
        
        # Add individual joint positions and velocities
        for i in range(self.num_joints):
            features[f'joint{i}_pos'] = joint_positions[..., i]
            features[f'joint{i}_vel'] = joint_velocities[..., i]
        
        return features
    
    def extract_drawer_features(self, observation: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract drawer position and velocity features.
        
        Args:
            observation: Raw observation array of shape [..., obs_dim]
            
        Returns:
            Dictionary with drawer features:
                - 'drawer_position': Drawer position
                - 'drawer_velocity': Drawer velocity
        """
        # Extract drawer position and velocity
        drawer_position = observation[..., self.drawer_pos_index]
        drawer_velocity = observation[..., self.drawer_vel_index]
        
        # Create feature dictionary
        features = {
            'drawer_position': drawer_position,
            'drawer_velocity': drawer_velocity
        }
        
        return features
    
    def extract_all_features(self, observation: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract all relevant features from observation.
        
        Args:
            observation: Raw observation array of shape [..., obs_dim]
            
        Returns:
            Dictionary with all extracted features
        """
        # Convert input to numpy if it's a torch tensor
        if hasattr(observation, 'detach'):
            observation = observation.detach().cpu().numpy()
        
        # Extract features using individual methods
        distance_features = self.extract_distance_features(observation)
        joint_features = self.extract_joint_features(observation)
        drawer_features = self.extract_drawer_features(observation)
        
        # Combine all features
        all_features = {}
        all_features.update(distance_features)
        all_features.update(joint_features)
        all_features.update(drawer_features)
        
        return all_features

# TODO: deprecate in favor of BaseFeatureExtractor
def segment_observation(obs, num_actions=9):
    """
    Segments the observation of shape (time, batch, obs_dim) into meaningful components.

    Parameters:
    -----------
    obs : tensor
        Observation tensor of shape (time, batch, obs_dim)
    num_actions : int
        Number of action dimensions (typically 9 for Franka robot with 7 arm joints + 2 finger joints)

    Returns:
    --------
    - 'positions': Joint positions (first num_actions dimensions)
    - 'velocities': Joint velocities (next num_actions dimensions)
    - 'rel_pos_to_handle': Relative coordinates to the drawer handle (next 3 dimensions)
    - 'drawer_position': Drawer position (next 1 dimension)
    - 'drawer_velocity': Drawer velocity (last dimension)
    """
    # Segment the observation into meaningful components
    positions = obs[:, :, :num_actions]  # Joint positions
    velocities = obs[:, :, num_actions:2*num_actions]  # Joint velocities
    rel_pos_to_handle = obs[:, :, 2*num_actions:2*num_actions+3]  # Relative coordinates to handle
    drawer_position = obs[:, :, 2*num_actions+3]  # Drawer position
    drawer_velocity = obs[:, :, 2*num_actions+4]  # Drawer velocity

    return positions, velocities, rel_pos_to_handle, drawer_position, drawer_velocity

# TODO: consider merging this into BaseFeatureExtractor
def extract_pos_vel_from_obs_list(obs_list, num_actions=9):
    """
    Extracts position and velocity data from a list of observation tensors.

    Parameters:
    -----------
    obs_list : list
        List of observation tensors, each with shape (time, batch, obs_dim)
    num_actions : int
        Number of action dimensions (typically 9 for Franka robot with 7 arm joints + 2 finger joints)

    Returns:
    --------
    tuple
        (positions_list, velocities_list) containing extracted data from all observations
    """
    positions_list = []
    velocities_list = []

    for obs in obs_list:
        # Use the segment_observation function to extract components
        positions, velocities, _, _, _ = segment_observation(obs, num_actions)
        positions_list.append(positions)
        velocities_list.append(velocities)

    return positions_list, velocities_list


def get_all_feature_keys():
    # Extract all 1D features from the dataset's metadata
    feature_keys = [
        # Distance features
        'distance', 'distance_x', 'distance_y', 'distance_z',

        # Drawer features
        'drawer_position', 'drawer_velocity'
    ]

    # Add joint features - assuming 9 joints as in the BaseFeatureExtractor
    num_joints = 9
    for i in range(num_joints):
        feature_keys.append(f'joint{i}_pos')
        feature_keys.append(f'joint{i}_vel')

    return feature_keys
