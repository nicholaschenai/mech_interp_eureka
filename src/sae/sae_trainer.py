import torch
import os
import pickle

from typing import Dict, Any, Optional, Callable, Tuple, List
from tqdm import tqdm

from .sae_model import SAEModel


def constant_lr(step: int, total_steps: int) -> float:
    """Constant learning rate schedule."""
    return 1.0


class SAETrainer:
    """
    Trainer for Sparse Autoencoder models.
    
    This class implements the training process for SAE models, including
    optimization, logging, and weight normalization. It separates the training
    logic from the model itself for better modularity.
    """
    
    def __init__(self, hidden_dim: int, sparsity: float = 0.05, l1_coef: float = 1e-3, 
                 learning_rate: float = 1e-3, batch_size: int = 256, 
                 tied_weights: bool = True, weight_normalize_eps: float = 1e-8):
        """
        Initialize trainer with model hyperparameters.
        
        Args:
            hidden_dim: Dimension of sparse feature space (typically 2-3x input_dim)
            sparsity: Target activation sparsity (fraction of active features)
            l1_coef: L1 regularization coefficient to encourage sparsity
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            tied_weights: Whether to use tied weights (decoder = encoder.T)
            weight_normalize_eps: Small constant for numerical stability in weight normalization
        """
        self.hidden_dim = hidden_dim
        self.sparsity = sparsity
        self.l1_coef = l1_coef
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.tied_weights = tied_weights
        self.weight_normalize_eps = weight_normalize_eps
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def train(self, activations: torch.Tensor, epochs: int = 100, 
              log_freq: int = 10, lr_scale: Callable[[int, int], float] = constant_lr,
              ) -> Tuple[SAEModel, Dict[str, Any]]:
        """
        Train SAE on activation data.
        
        Args:
            activations: Tensor of activation data (samples x features)
            epochs: Number of training epochs
            log_freq: Frequency of logging metrics
            lr_scale: Learning rate scaling function (default: constant learning rate)
            
        Returns:
            Tuple of (trained SAEModel instance, training metrics dictionary)
        """
        # Get input dimension from activations
        input_dim = activations.shape[1]
        
        # Create model
        model = SAEModel(
            input_dim=input_dim, 
            hidden_dim=self.hidden_dim,
            l1_coef=self.l1_coef,
            tied_weights=self.tied_weights,
            weight_normalize_eps=self.weight_normalize_eps
        )
        
        # Create data loader
        data_loader = self._create_data_loader(activations)
        
        # Optimize the model
        data_log = self._optimize(model, data_loader, epochs, log_freq, lr_scale)
        
        # Print final metrics
        # final_metrics = self.compute_metrics_batched(model, data_loader)
        final_metrics = self.compute_metrics(activations, model)
        print(f"Training complete. Final loss: {final_metrics['loss']:.6f}, "
              f"Sparsity: {final_metrics['sparsity']:.6f}")
        
        metrics_data = {'data_log': data_log, 'final_metrics': final_metrics}

        return model, metrics_data
    
    def compute_metrics(self, activations: torch.Tensor, model: SAEModel) -> Dict[str, Any]:
        """
        Compute metrics like sparsity, reconstruction error, etc.
        
        Args:
            activations: Activation data
            model: Trained SAE model
            
        Returns:
            Dictionary of metrics
        """
        activations = activations.to(model.device)
        
        with torch.no_grad():
            loss_dict, loss, acts_post, x_reconstructed = model.forward(activations)
            # Compute per-sample reconstruction error (the delta between original and reconstructed)
            # This will be used for activation intervention later
            reconstruction_error = activations - x_reconstructed

        mean_loss = loss.mean().item()
        mean_reconstruction_loss = loss_dict["L_reconstruction"].mean().item()
        mean_sparsity_loss = loss_dict["L_sparsity"].mean().item()
        
        # Calculate overall sparsity (fraction of hidden units that are active)
        sparsity = (acts_post.abs() > 1e-8).float().mean().item()
        
        # Calculate per-feature sparsity (fraction of samples where each feature is active)
        per_feature_sparsity = (acts_post.abs() > 1e-8).float().mean(dim=0)
        
        # Count dead features (never activate)
        dead_features = (per_feature_sparsity < 1e-6).sum().item()
        
        # Calculate l0 sparsity (average number of active features per sample)
        l0_sparsity = (acts_post.abs() > 1e-8).float().sum(dim=1).mean().item()
        
        feature_means = acts_post.mean(dim=0)
        feature_stds = acts_post.std(dim=0)
        feature_max_acts = acts_post.max(dim=0)[0]
        
        return {
            'loss': mean_loss,
            'reconstruction_loss': mean_reconstruction_loss,
            'sparsity_loss': mean_sparsity_loss,
            'sparsity': sparsity,
            'per_feature_sparsity': per_feature_sparsity,
            'dead_features': dead_features,
            'dead_features_percent': dead_features / model.hidden_dim * 100,
            'l0_sparsity': l0_sparsity,
            'l0_sparsity_percent': l0_sparsity / model.hidden_dim * 100,
            'feature_means': feature_means,
            'feature_stds': feature_stds,
            'feature_max_acts': feature_max_acts,
            'reconstruction_error': reconstruction_error,
        }
        
    def _create_data_loader(self, activations: torch.Tensor) -> torch.utils.data.DataLoader:
        dataset = torch.utils.data.TensorDataset(activations)
        
        data_loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.batch_size,
            shuffle=True
        )
        
        return data_loader
    
    def _optimize(self, model: SAEModel, data_loader: torch.utils.data.DataLoader, 
                 epochs: int, log_freq: int = 10,
                 lr_scale: Callable[[int, int], float] = constant_lr) -> List[Dict[str, Any]]:
        """
        Optimize the SAE model using the provided data.
        
        Args:
            model: SAE model to train
            data_loader: DataLoader providing batches of activations
            epochs: Number of epochs to train
            log_freq: How often to log metrics (in batches)
            lr_scale: Learning rate scaling function (default: constant learning rate)
            
        Returns:
            List of dictionaries containing training metrics and logs at each logged step
        """
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        
        # Metrics tracking
        frac_active_list = []
        
        # Create list of dictionaries for logging - matches example format
        data_log = []
        
        # Sample size for saving activation examples
        hidden_sample_size = min(self.batch_size, 256)
        
        total_steps = len(data_loader) * epochs
        
        # Training loop
        step = 0
        progress_bar = tqdm(total=total_steps, desc="Training SAE")
        
        for epoch in range(epochs):
            for batch_idx, (x,) in enumerate(data_loader):
                x = x.to(model.device)
                
                # Update learning rate according to schedule
                current_lr = self.learning_rate * lr_scale(step, total_steps)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
                
                optimizer.zero_grad()
                
                loss_dict, loss, acts_post, x_reconstructed = model.forward(x)
                
                loss.mean().backward()
                
                optimizer.step()
                
                # If using non-tied weights, normalize the decoder weights
                if not model.tied_weights:
                    with torch.no_grad():
                        model.normalize_decoder_weights()
                
                # Calculate the mean sparsities over batch dim for each feature
                frac_active = (acts_post.abs() > 1e-8).float().mean(dim=0)
                frac_active_list.append(frac_active)
                
                progress_bar.update(1)
                
                if step % log_freq == 0 or step == total_steps - 1:
                    progress_bar.set_postfix({
                        'loss': loss.mean().item(),
                        'sparsity': frac_active.mean().item(),
                        'lr': current_lr,
                        **{k: v.mean(0).sum().item() for k, v in loss_dict.items()},
                    })
                    # Get a sample batch for visualization - only when logging
                    with torch.no_grad():
                        sample_x = x[:hidden_sample_size]
                        loss_dict, loss, acts, h_r = model.forward(sample_x)
                        
                    log_entry = {
                        "step": step,
                        "epoch": epoch,
                        "frac_active": (acts.abs() > 1e-8).float().mean(dim=0).detach().cpu(),
                        "loss": loss.detach().cpu(),
                        "h": x.detach().cpu(),
                        "h_r": h_r.detach().cpu(),
                        **{name: param.detach().cpu() for name, param in model.named_parameters()},
                        **{name: loss_term.detach().cpu() for name, loss_term in loss_dict.items()},
                        "mean_loss": loss.mean().item(),
                        "mean_sparsity": frac_active.mean().item(),
                        "lr": current_lr
                    }
                    
                    data_log.append(log_entry)
                    
                    if epoch % max(1, epochs // 10) == 0 and batch_idx == 0:
                        print(f"Epoch {epoch+1}/{epochs}, "
                              f"Loss: {loss.mean().item():.6f}, "
                              f"Sparsity: {frac_active.mean().item():.6f}, "
                              f"LR: {current_lr:.6f}")
                
                step += 1
        
        progress_bar.close()
        
        return data_log 

    # Only do when needed
    # def compute_metrics_batched(self, model: SAEModel, 
    #                            data_loader: torch.utils.data.DataLoader) -> Dict[str, Any]:
    #     """
    #     Compute metrics in batches to avoid memory issues with large datasets.
        
    #     Args:
    #         model: Trained SAE model
    #         data_loader: DataLoader for batch processing
            
    #     Returns:
    #         Dictionary of metrics
    #     """
    #     # Initialize accumulators
    #     total_samples = 0
    #     weighted_metrics = {
    #         'loss': 0,
    #         'reconstruction_loss': 0,
    #         'sparsity_loss': 0,
    #         'sparsity': 0,
    #         'mse': 0,
    #         'l0_sparsity': 0,
    #     }
        
    #     # For per-feature statistics
    #     feature_acts = torch.zeros(model.hidden_dim, device=model.device)
    #     feature_sum = torch.zeros(model.hidden_dim, device=model.device)
    #     feature_sum_sq = torch.zeros(model.hidden_dim, device=model.device)
    #     feature_max_acts = torch.zeros(model.hidden_dim, device=model.device)
        
    #     # Process in batches
    #     for batch_idx, (x,) in enumerate(data_loader):
    #         x = x.to(model.device)
    #         batch_size = x.shape[0]
    #         total_samples += batch_size
            
    #         # Get metrics for this batch
    #         batch_metrics = self.compute_metrics(x, model)
            
    #         # Weight and accumulate scalar metrics by batch size
    #         for metric in weighted_metrics:
    #             weighted_metrics[metric] += batch_metrics[metric] * batch_size
            
    #         # Accumulate tensor metrics manually
    #         feature_acts += (batch_metrics['per_feature_sparsity'] * batch_size)
    #         feature_sum += batch_metrics['feature_means'] * batch_size
    #         feature_sum_sq += (batch_metrics['feature_stds']**2 + batch_metrics['feature_means']**2) * batch_size
    #         feature_max_acts = torch.maximum(feature_max_acts, batch_metrics['feature_max_acts'])
        
    #     # Normalize accumulated metrics
    #     result = {}
    #     for metric in weighted_metrics:
    #         result[metric] = weighted_metrics[metric] / total_samples
        
    #     # Calculate per-feature sparsity
    #     per_feature_sparsity = feature_acts / total_samples
        
    #     # Count dead features
    #     dead_features = (per_feature_sparsity < 1e-6).sum().item()
        
    #     # Feature statistics
    #     feature_means = feature_sum / total_samples
    #     feature_vars = (feature_sum_sq / total_samples) - (feature_means ** 2)
    #     feature_stds = torch.sqrt(torch.clamp(feature_vars, min=1e-8))
        
    #     # Add derived metrics and tensor metrics
    #     result.update({
    #         'per_feature_sparsity': per_feature_sparsity,
    #         'dead_features': dead_features,
    #         'dead_features_percent': dead_features / model.hidden_dim * 100,
    #         'l0_sparsity_percent': result['l0_sparsity'] / model.hidden_dim * 100,
    #         'feature_means': feature_means,
    #         'feature_stds': feature_stds,
    #         'feature_max_acts': feature_max_acts
    #     })
        
    #     return result
