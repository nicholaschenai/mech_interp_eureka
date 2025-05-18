"""
Get activations from models and save them
"""
import sys
import os

import torch
import numpy as np

sys.path.append(os.path.abspath('..'))

from config import MODEL_CONFIGS, DEFAULT_CHECKPOINT_DIR as CHECKPOINT_DIR
from src.model_hook import ModelHook
from src.activation_collector import ActivationCollector

from utils.base_feature_extractor import BaseFeatureExtractor
from utils.base_phase_detection import PhaseDetector
from utils.model_utils import load_model, load_data, unwrap_data

PERFORM_ASSERTION = True
MAX_TIMESTEPS = 100  # Truncate at 100 timesteps as robot behavior becomes erratic beyond this point
tolerance = 1e-5


def verify_model_outputs(model_outputs, mu, value, t):
    mu_diff = np.abs(model_outputs['mu'].cpu().numpy() - mu[t]).max()
    value_diff = np.abs(model_outputs['value'].cpu().numpy() - value[t]).max()
    
    if not (mu_diff < tolerance and value_diff < tolerance):
        print(f"  Mu difference: {mu_diff:.6f}, Value difference: {value_diff:.6f}")
        raise ValueError("Model outputs differ from expected values")

def process_model(model_type, model_file):
    """Process a single model and save its activation dataset.
    
    Args:
        model_type: Type of model (strong, medium, weak)
        model_file: Filename of the model checkpoint
    """
    output_file = os.path.join(CHECKPOINT_DIR, f'{model_type}_activation_dataset.pkl')
    
    # Check if the output file already exists
    if os.path.exists(output_file):
        print(f"Output file for {model_type} model already exists, skipping...")
        return
    
    print(f"Processing {model_type} model...")
    
    # Load model and data
    model = load_model(model_file, CHECKPOINT_DIR)
    data = load_data(model_file, CHECKPOINT_DIR)

    full_obs, mu, value = unwrap_data(data)
    
    # Truncate at MAX_TIMESTEPS as behavior becomes erratic beyond this point
    full_obs = full_obs[:MAX_TIMESTEPS]
    mu = mu[:MAX_TIMESTEPS]
    value = value[:MAX_TIMESTEPS]

    feature_extractor = BaseFeatureExtractor()
    all_features = feature_extractor.extract_all_features(full_obs)
    phase_detector = PhaseDetector()
    all_phase_masks = phase_detector.detect_phases(all_features)

    model_hook = ModelHook(model)
    model_hook.register_hooks()

    dataset = ActivationCollector()
    dataset.prepare_for_collection()
    
    if not isinstance(full_obs, torch.Tensor):
        full_obs = torch.tensor(full_obs)
    
    # Process activations one timestep at a time (for memory efficiency)
    total_timesteps = len(full_obs)
    for t, batch_obs in enumerate(full_obs):
        if t % 10 == 0:
            print(f"  Processing timestep {t}/{total_timesteps} for {model_type} model...")
        
        batch_activations, model_outputs = model_hook.get_activations_and_outputs(batch_obs)

        if PERFORM_ASSERTION:
            verify_model_outputs(model_outputs, mu, value, t)

        dataset.add_batch_timestep(
            observations=batch_obs,
            activations=batch_activations,
        )
        
        # Optional: clear GPU memory
        torch.cuda.empty_cache()
    
    dataset.add_metadata({
        'features': all_features,
        'phase_masks': all_phase_masks
    })
    
    # Finalize the dataset (convert lists to tensors)
    print(f"Finalizing dataset for {model_type} model...")
    dataset.finalize_collection()
    
    print(f"Dataset for {model_type} model finalized")
    
    model_hook.clear_hooks()

    print(f"Saving dataset for {model_type} model...")
    dataset.save(output_file)
    print(f"Dataset for {model_type} model saved successfully!")


if __name__ == "__main__":
    for model_type, config in MODEL_CONFIGS.items():
        model_file = config["file"]
        process_model(model_type, model_file)
        print(f"Completed processing {model_type} model\n")

    print("All models processed successfully!")
