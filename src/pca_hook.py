import torch
import numpy as np
from typing import Optional


class PCAHook:
    """
    Forward hook for PyTorch modules that applies a PCA bottleneck transformation.
    
    This hook intercepts the output of a module, projects it to a lower-dimensional
    PCA space, and then projects it back to the original space.
    """
    def __init__(self, 
                components_file: str, 
                layer_name: str,
                n_components: Optional[int] = None,
                use_cuda: bool = True):
        """
        Args:
            components_file: Path to the combined .npz file containing PCA components
            layer_name: Layer name to load components for
            n_components: Number of principal components to use. If None, uses all available.
            use_cuda: Whether to move tensors to CUDA if model is on GPU
        """
        all_data = np.load(components_file)
        component_key = f"{layer_name}_components"
        mean_key = f"{layer_name}_mean"
        
        if component_key not in all_data:
            raise KeyError(f"Could not find components for layer '{layer_name}' in {components_file}. "
                           f"Available keys: {list(all_data.keys())}")
            
        self.components = all_data[component_key]
        self.n_components_available = self.components.shape[0]
        
        # Load the mean vector (crucial for proper PCA transformation)
        if mean_key in all_data:
            self.mean = all_data[mean_key]
        else:
            print(f"Warning: Could not find mean for layer '{layer_name}'. Using zero mean, "
                  f"which may result in reconstruction errors.")
            # If mean is missing, assume zero mean (this is a fallback, but will likely cause errors)
            self.mean = np.zeros(self.components.shape[1])
        
        self.set_n_components(n_components)
        
        self.use_cuda = use_cuda
        self.device = None
        self._encoder = None
        self._decoder = None
        self._mean_tensor = None
    
    def set_n_components(self, n_components: Optional[int] = None):
        if n_components is None:
            print(f"No n_components specified, using all available components")
            n_components = self.n_components_available
        
        if n_components > self.n_components_available:
            print(f"Warning: Requested {n_components} components but only "
                  f"{self.n_components_available} are available. Using the maximum available.")

        self.n_components = min(n_components, self.n_components_available)
        
        # Force recomputation of encoder/decoder
        self._encoder = None
        self._decoder = None
        self._mean_tensor = None
    
    @property
    def encoder(self):
        """Get the encoder matrix (original_dim x n_components)"""
        if self._encoder is None or self._encoder.shape[1] != self.n_components:
            components = self.components[:self.n_components]
            
            self._encoder = torch.from_numpy(components.T).float().detach().requires_grad_(False)
            
            if self.device is not None:
                self._encoder = self._encoder.to(self.device)
        
        return self._encoder
    
    @property
    def decoder(self):
        """Get the decoder matrix (n_components x original_dim)"""
        if self._decoder is None or self._decoder.shape[0] != self.n_components:
            components = self.components[:self.n_components]
            
            self._decoder = torch.from_numpy(components).float().detach().requires_grad_(False)
            
            if self.device is not None:
                self._decoder = self._decoder.to(self.device)
        
        return self._decoder
    
    @property
    def mean_tensor(self):
        """Get the mean tensor with proper device placement"""
        if self._mean_tensor is None:
            self._mean_tensor = torch.from_numpy(self.mean).float().detach().requires_grad_(False)
            
            if self.device is not None:
                self._mean_tensor = self._mean_tensor.to(self.device)
        
        return self._mean_tensor
    
    def __call__(self, module, input_tensor, output_tensor):
        """
        Apply the PCA bottleneck transformation.
        
        Args:
            module: The module this hook is registered to
            input_tensor: Input to the module (not used)
            output_tensor: Output from the module
            
        Returns:
            Transformed tensor
        """
        if self.use_cuda and self.device != output_tensor.device:
            self.device = output_tensor.device
            
            # Force recomputation of tensors on new device
            self._encoder = None
            self._decoder = None
            self._mean_tensor = None
        
        # Proper PCA transformation: Center, project, and reconstruct
        centered = output_tensor - self.mean_tensor
        projected = torch.matmul(centered, self.encoder)
        reconstructed = torch.matmul(projected, self.decoder) + self.mean_tensor
        
        return reconstructed
