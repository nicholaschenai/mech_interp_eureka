"""
Utility functions for loading and analyzing RL models.
"""

import os
import dill
import numpy as np
import torch

def load_model(checkpoint_file, checkpoint_dir):
    """Load a model from dill file
    
    Args:
        checkpoint_file: Name of the checkpoint file
        checkpoint_dir: Directory containing the checkpoint
        
    Returns:
        The loaded model
    """
    checkpoint_name = os.path.splitext(checkpoint_file)[0]
    dill_filename = f"{checkpoint_name}_dill.pkl"
    dill_path = os.path.join(checkpoint_dir, dill_filename)
    
    if not os.path.exists(dill_path):
        raise FileNotFoundError(f"No dill model file found at {dill_path}")
    
    print(f"Loading model from: {dill_path}")
    with open(dill_path, 'rb') as f:
        model = dill.load(f)
    print(f"Model loaded successfully! Type: {type(model).__name__}")
    
    return model

def load_data(checkpoint_file, checkpoint_dir):
    """Load saved data from npz file
    
    Args:
        checkpoint_file: Name of the checkpoint file
        checkpoint_dir: Directory containing the checkpoint
        
    Returns:
        Loaded numpy data
    """
    checkpoint_name = os.path.splitext(checkpoint_file)[0]
    npz_filename = f"{checkpoint_name}.npz"
    npz_path = os.path.join(checkpoint_dir, npz_filename)
    
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"No npz file found at {npz_path}")
    
    print(f"Loading data from: {npz_path}")
    data = np.load(npz_path)
    
    return data

def compute_model_outputs(model, observations, expected_mu, expected_value):
    """Run observations through model and return outputs
    
    Args:
        model: The RL model to evaluate
        observations: Pre-normalized observations to process
        expected_mu: Expected mu shape from loaded data
        expected_value: Expected value shape from loaded data
        
    Returns:
        Tuple of (mu_values, value_estimates, post_norm_obs)
    """
    device = next(model.parameters()).device
    obs_tensor = torch.tensor(observations, dtype=torch.float32, device=device)
    
    n_batches = observations.shape[0]
    
    # Prepare arrays for outputs using np.zeros_like
    post_norm_obs = np.zeros_like(observations)
    mu_values = np.zeros_like(expected_mu)
    value_estimates = np.zeros_like(expected_value)
    
    # Process all batches
    model.eval()
    with torch.no_grad():
        for i in range(n_batches):
            # Get batch
            obs_batch = obs_tensor[i]
            
            normalized_obs = model.norm_obs(obs_batch)

            post_norm_obs[i] = normalized_obs.cpu().numpy()

            input_dict = {"obs": normalized_obs}
            
            # Forward pass through the model
            mu, logstd, value, states = model.a2c_network(input_dict)

            # Extract mu and value from model output
            mu_values[i] = mu.cpu().numpy()
            value_estimates[i] = value.cpu().numpy()
            
    return mu_values, value_estimates, post_norm_obs


def process_model_outputs(mu, value, model):
    """
    Process model outputs by clipping mu and denormalizing value

    Args:
        mu: The mu values to clip between -1 and 1
        value: The value estimates to denormalize
        model: The model to use for denormalization

    Returns:
        Tuple of (clipped_mu, denormalized_value) as numpy arrays
    """
    # Clip mu values between -1 and 1
    clipped_mu = np.clip(mu, -1, 1)

    # Denormalize value using torch, then convert back to numpy
    value_tensor = torch.tensor(value).to(next(model.parameters()).device)
    denormalized_value = model.denorm_value(value_tensor).cpu().numpy()

    return clipped_mu, denormalized_value

def unwrap_data(data):
    return data['pre_norm_obs'], data['mu'], data['value']
