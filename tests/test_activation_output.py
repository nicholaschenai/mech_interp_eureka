"""
Checks that the activation collected via hooks (in ActivationDataset) matches the output of the model.

This script verifies that the 'mu' and 'value' outputs stored in ActivationDataset
match the outputs when running the same observations through the model.
"""
import numpy as np
from config import DEFAULT_CHECKPOINT_DIR, MODEL_CONFIGS

from utils.model_utils import load_model, load_data, unwrap_data
from utils.activation_dataset_utils import load_model_dataset

model_name = 'strong'
tolerance = 1e-5


if __name__ == "__main__":
    model_file = MODEL_CONFIGS[model_name]['file']
    activation_dataset = load_model_dataset(DEFAULT_CHECKPOINT_DIR, model_name)
    
    mu_activations = activation_dataset.get_activation_tensor('mu')
    value_activations = activation_dataset.get_activation_tensor('value')
    
    print("\nActivation shapes:")
    print(f"mu: {mu_activations.shape}")
    print(f"value: {value_activations.shape}")
    
    data = load_data(model_file, DEFAULT_CHECKPOINT_DIR)
    full_obs, mu, value = unwrap_data(data)

    print("\nData shapes:")
    print(f"mu: {mu.shape}")
    print(f"value: {value.shape}")

    mu_diff = np.abs(mu_activations-mu).max()
    value_diff = np.abs(value_activations-value).max()
    
    if not (mu_diff < tolerance and value_diff < tolerance):
        print(f"  Mu difference: {mu_diff:.6f}, Value difference: {value_diff:.6f}")
        raise ValueError("Model outputs differ from expected values")
    print("Model outputs match expected values")