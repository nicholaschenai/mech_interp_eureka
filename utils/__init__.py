import re
import numpy as np
from typing import List

# Joint names for franka robot
joint_names = [
    "Base Rotation",
    "Shoulder Joint",
    "Arm Rotation",
    "Elbow Joint",
    "Forearm Rotation",
    "Wrist Bend",
    "End Effector Rotation",
    "Gripper Finger 1",
    "Gripper Finger 2"
]

def get_observation_labels() -> List[str]:
    """
    Generate labels for all observations in the environment.
    
    Returns:
        List of observation labels in order:
        - Joint positions (9)
        - Joint velocities (9)
        - Target vector (3)
        - Drawer position (1)
        - Drawer velocity (1)
    """
    labels = []
    
    # Joint positions
    labels.extend([f"{joint} position" for joint in joint_names])
    
    # Joint velocities
    labels.extend([f"{joint} velocity" for joint in joint_names])
    
    # Target vector
    labels.extend(["distance_x", "distance_y", "distance_z"])
    
    # Drawer state
    labels.extend(["drawer_position", "drawer_velocity"])
    
    return labels

def get_layer_order(layer_names: List[str]) -> List[str]:
    """
    Args:
        layer_names: List of layer names

    Returns:
        Sorted list of layer names in forward pass order
    """
    # For our specific model, we know the layer order from the architecture
    # actor_mlp_0, actor_mlp_1, ..., actor_mlp_5, mu

    # Extract numeric part from actor_mlp_X layers
    mlp_layers = [layer for layer in layer_names if layer.startswith("actor_mlp_")]
    mlp_indices = [(layer, int(layer.split("_")[-1])) for layer in mlp_layers]

    # Sort by index
    sorted_mlp_layers = [layer for layer, _ in sorted(mlp_indices, key=lambda x: x[1])]

    return sorted_mlp_layers

def assert_outputs_match(computed_mu, computed_value, original_mu, original_value, tolerance=1e-5):
    mu_diff = np.abs(computed_mu - original_mu)
    value_diff = np.abs(computed_value - original_value)
    
    max_mu_diff = mu_diff.max()
    max_value_diff = value_diff.max()
    
    if max_mu_diff >= tolerance or max_value_diff >= tolerance:
        print("Warning: difference from the original outputs")

        print(f"  Max absolute difference in mu: {max_mu_diff:.6e}")
        print(f"  Max absolute difference in value: {max_value_diff:.6e}")
        print(f"  Percentage difference in mu: {100 * np.mean(mu_diff / (np.abs(original_mu) + 1e-9)):.4f}%")
    
        max_idx = np.unravel_index(np.argmax(mu_diff), mu_diff.shape)
        print(f"  Largest difference at index {max_idx}:")
        print(f"    Original mu: {original_mu[max_idx]:.6f}")
        print(f"    Reconstructed mu: {computed_mu[max_idx]:.6f}")
        raise ValueError("accuracy issue detected")

def parse_joint_name(feature_name: str) -> str:
    # Find all joint{idx} patterns
    joint_pattern = r'joint(\d+)'
    matches = re.finditer(joint_pattern, feature_name)
    
    # Replace each match with the corresponding joint name
    result = feature_name
    for match in matches:
        joint_idx = int(match.group(1))
        if joint_idx < len(joint_names):
            result = result.replace(match.group(0), joint_names[joint_idx])
    
    return result
