"""
PCA Projection Analyzer for evaluating how PCA bottlenecks affect model outputs.

This module analyzes the effect of reducing the dimensionality of neural activations
through PCA on model performance, focusing on action fidelity metrics.

Responsibilities:
- Load and process PCA bottleneck experiment data
- Compute action fidelity metrics (MSE, cosine similarity, etc.)
- Analyze per-dimension errors in actions
- Identify critical component thresholds
"""
import numpy as np

from typing import Dict
from utils.model_utils import load_data as load_model_data, unwrap_data


class PCAProjectionAnalyzer:
    def __init__(self):
        self.data = None
        self.metrics = {}
        self.dimension_errors = {}
        self.shape_info = {}
    
    def load_data(self, ablation_data_path: str, checkpoint_dir: str, model_file: str) -> None:
        """
        Load both original data and PCA bottlenecked data.
        
        Args:
            ablation_data_path: Path to the npz file with PCA bottleneck results
            checkpoint_dir: Directory containing model checkpoints
            model_file: Model file name for loading original data
        """
        # Load bottlenecked data from NPZ
        raw_data = np.load(ablation_data_path)
        
        component_counts = raw_data['component_counts']
        
        # Load original data directly from model outputs
        original_data = load_model_data(model_file, checkpoint_dir)
        
        # Extract original mu and value
        _, original_mu, original_value = unwrap_data(original_data)
        
        # Store shape information
        self.shape_info = {
            'original_shape': original_mu.shape,
            'is_3d': len(original_mu.shape) == 3,  # Check if shape is (time, batch, dim)
            'time_steps': original_mu.shape[0] if len(original_mu.shape) == 3 else 1,
            'action_dims': original_mu.shape[-1]
        }
        
        print(f"Data shape: {original_mu.shape}")
        
        # Structure the data
        structured_data = {
            'component_counts': component_counts,
            'target_layers': raw_data['target_layers'],
            'bottlenecked': {},
            'original': {
                'mu': original_mu,
                'value': original_value
            }
        }
        
        # Extract bottlenecked data for each component count
        for n_components in component_counts:
            structured_data['bottlenecked'][n_components] = {
                'mu': raw_data[f'mu_{n_components}'],
                'value': raw_data[f'value_{n_components}']
            }
        
        self.data = structured_data
    
    def compute_action_fidelity_metrics(self, 
                                      original_outputs: Dict[str, np.ndarray], 
                                      bottlenecked_outputs: Dict[str, np.ndarray]) -> Dict:
        """
        Compute metrics comparing original model outputs to PCA-bottlenecked outputs.
        
        Args:
            original_outputs: Dict with 'mu' and 'value' from original model
            bottlenecked_outputs: Dict with 'mu' and 'value' from PCA-bottlenecked model
            
        Returns:
            Dict of metrics including MSE, max deviation, and cosine similarity
        """
        metrics = {}
        
        # Ensure we're working with numpy arrays
        orig_mu = original_outputs['mu']
        orig_value = original_outputs['value']
        pca_mu = bottlenecked_outputs['mu']
        pca_value = bottlenecked_outputs['value']
        
        # Calculate MSE for mu and value
        mu_mse = np.mean((orig_mu - pca_mu) ** 2)
        value_mse = np.mean((orig_value - pca_value) ** 2)
        
        # Calculate max deviation (max absolute difference)
        mu_max_dev = np.max(np.abs(orig_mu - pca_mu))
        value_max_dev = np.max(np.abs(orig_value - pca_value))
        
        # Handle clipped mu values (-1 to 1)
        clipped_orig_mu = np.clip(orig_mu, -1, 1)
        clipped_pca_mu = np.clip(pca_mu, -1, 1)
        
        clipped_mu_mse = np.mean((clipped_orig_mu - clipped_pca_mu) ** 2)
        clipped_mu_max_dev = np.max(np.abs(clipped_orig_mu - clipped_pca_mu))
        
        # Calculate normalized MSE (as percentage of original variance)
        mu_var = np.var(orig_mu.reshape(-1))  # Flatten for overall variance
        value_var = np.var(orig_value.reshape(-1))
        
        mu_nmse = (mu_mse / mu_var) if mu_var > 0 else float('inf')
        value_nmse = (value_mse / value_var) if value_var > 0 else float('inf')
        
        # Calculate clipped normalized MSE
        clipped_mu_var = np.var(clipped_orig_mu.reshape(-1))
        clipped_mu_nmse = (clipped_mu_mse / clipped_mu_var) if clipped_mu_var > 0 else float('inf')
        
        # Calculate cosine similarity for mu (vectorized version for efficiency)
        cosine_sim = self._compute_cosine_similarity(orig_mu, pca_mu)
        clipped_cosine_sim = self._compute_cosine_similarity(clipped_orig_mu, clipped_pca_mu)
        
        # Combine metrics
        metrics['mu_mse'] = mu_mse
        metrics['value_mse'] = value_mse
        metrics['mu_max_dev'] = mu_max_dev
        metrics['value_max_dev'] = value_max_dev
        metrics['mu_nmse'] = mu_nmse
        metrics['value_nmse'] = value_nmse
        metrics['mu_cosine'] = cosine_sim
        
        # Add clipped metrics
        metrics['clipped_mu_mse'] = clipped_mu_mse
        metrics['clipped_mu_max_dev'] = clipped_mu_max_dev
        metrics['clipped_mu_nmse'] = clipped_mu_nmse
        metrics['clipped_mu_cosine'] = clipped_cosine_sim
        
        # Add time-specific metrics if we have time dimension
        if self.shape_info.get('is_3d', False):
            # Calculate MSE over time (average across batch and dimensions)
            metrics['time_mu_mse'] = np.mean((orig_mu - pca_mu) ** 2, axis=(1, 2))  # Shape: (time,)
            metrics['time_value_mse'] = np.mean((orig_value - pca_value) ** 2, axis=(1, 2))
            
            # Store the average error progression over time
            metrics['avg_error_over_time'] = metrics['time_mu_mse']
        
        return metrics
    
    def _compute_cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        # If the arrays are multi-dimensional, they are flattened first.
        a_flat = a.reshape(-1)
        b_flat = b.reshape(-1)
        
        # Handle zero vectors
        a_norm = np.linalg.norm(a_flat)
        b_norm = np.linalg.norm(b_flat)
        
        if a_norm > 0 and b_norm > 0:
            return np.dot(a_flat, b_flat) / (a_norm * b_norm)
        else:
            return 0.0
    
    def analyze_per_dimension_errors(self, component_count: int) -> Dict:
        """
        Analyze which action dimensions are most affected by the PCA bottleneck.
        
        Args:
            component_count: Number of PCA components to analyze
            
        Returns:
            Dictionary with per-dimension error metrics for both unclipped and clipped actions
        """
        if self.data is None:
            raise ValueError("Data must be loaded first using load_data()")
        
        original_mu = self.data['original']['mu']
        bottlenecked_mu = self.data['bottlenecked'][component_count]['mu']
        
        # Determine if we're working with (time, batch, dim) or another shape
        is_3d = self.shape_info.get('is_3d', False)
        
        # Create clipped versions (typically actions are clipped to [-1, 1])
        clip_range = (-1, 1)
        clipped_original_mu = np.clip(original_mu, clip_range[0], clip_range[1])
        clipped_bottlenecked_mu = np.clip(bottlenecked_mu, clip_range[0], clip_range[1])
        
        # Calculate errors for both unclipped and clipped data
        results = {}
        
        # Process unclipped data
        results.update(self._calculate_dimension_errors(
            original_mu, bottlenecked_mu, is_3d, prefix=""))
        
        # Process clipped data
        results.update(self._calculate_dimension_errors(
            clipped_original_mu, clipped_bottlenecked_mu, is_3d, prefix="clipped_"))
        
        # Store in instance for later use
        self.dimension_errors[component_count] = results
        
        return results
    
    def _calculate_dimension_errors(self, original: np.ndarray, bottlenecked: np.ndarray, 
                                   is_3d: bool, prefix: str = "") -> Dict:
        """
        Helper function to calculate dimension errors for a given pair of original and bottlenecked data.
        
        Args:
            original: Original data
            bottlenecked: Bottlenecked data
            is_3d: Whether data is 3D (time, batch, dim) or 2D (batch, dim)
            prefix: Prefix to add to metric names (e.g., "clipped_" for clipped metrics)
            
        Returns:
            Dictionary with per-dimension error metrics
        """
        results = {}
        
        # For 3D data (time, batch, dim), average across time and batch
        if is_3d:
            # Calculate per-dimension MSE (average across time and batch)
            squared_errors = (original - bottlenecked) ** 2
            results[f'{prefix}per_dim_mse'] = np.mean(squared_errors, axis=(0, 1))  # Average across time and batch
            
            # Calculate per-dimension max error
            results[f'{prefix}per_dim_max_error'] = np.max(np.abs(original - bottlenecked), axis=(0, 1))
            
            # Calculate per-dimension normalized MSE
            # First reshape to (time*batch, dim) for correct variance calculation
            reshaped_orig = original.reshape(-1, original.shape[-1])
            per_dim_var = np.var(reshaped_orig, axis=0)
            
            # Calculate per-dimension relative error (preserving time and batch)
            with np.errstate(divide='ignore', invalid='ignore'):
                # Calculate relative errors across all samples
                rel_errors = np.abs(original - bottlenecked) / np.abs(original)
                rel_errors = np.nan_to_num(rel_errors)  # Replace NaNs with 0s
                results[f'{prefix}per_dim_relative_error'] = np.mean(rel_errors, axis=(0, 1))
                
            # Calculate time-specific errors (average across batch)
            results[f'{prefix}time_dim_mse'] = np.mean(squared_errors, axis=1)  # Shape: (time, dim)
            
        else:
            # For 2D data (samples, dim)
            squared_errors = (original - bottlenecked) ** 2
            results[f'{prefix}per_dim_mse'] = np.mean(squared_errors, axis=0)
            results[f'{prefix}per_dim_max_error'] = np.max(np.abs(original - bottlenecked), axis=0)
            per_dim_var = np.var(original, axis=0)
            
            with np.errstate(divide='ignore', invalid='ignore'):
                rel_errors = np.abs(original - bottlenecked) / np.abs(original)
                rel_errors = np.nan_to_num(rel_errors)
                results[f'{prefix}per_dim_relative_error'] = np.mean(rel_errors, axis=0)
        
        # Calculate normalized MSE
        results[f'{prefix}per_dim_nmse'] = np.divide(
            results[f'{prefix}per_dim_mse'], 
            per_dim_var, 
            out=np.full_like(per_dim_var, fill_value=float('inf')), 
            where=per_dim_var > 0
        )
        
        # Add sorted indices
        results[f'{prefix}most_affected_dims'] = np.argsort(results[f'{prefix}per_dim_mse'])[::-1]
        results[f'{prefix}least_affected_dims'] = np.argsort(results[f'{prefix}per_dim_mse'])
        
        # Add time-specific info if available
        if is_3d and f'{prefix}time_dim_mse' in results:
            if 'shape_info' not in results:
                results['shape_info'] = {
                    'time_steps': self.shape_info['time_steps'],
                    'action_dims': self.shape_info['action_dims']
                }
        
        return results
    
    def compute_metrics_across_components(self) -> Dict:
        """
        Compute metrics for each component count.
        
        Returns:
            Dict mapping component counts to metric dictionaries
        """
        if self.data is None:
            raise ValueError("Data must be loaded first using load_data()")
        
        results = {}
        for n_components in self.data['component_counts']:
            results[n_components] = self.compute_action_fidelity_metrics(
                self.data['original'],
                self.data['bottlenecked'][n_components]
            )
            
            # Also compute per-dimension metrics
            self.analyze_per_dimension_errors(n_components)
        
        # Store in instance for later use
        self.metrics = results
        
        return results
