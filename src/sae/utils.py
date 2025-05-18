import glob
import os
import re

from typing import Dict

import torch

from .sae_model import SAEModel


def load_sae_models(sae_dir: str, model_name: str = "strong") -> Dict[str, SAEModel]:
    """
    Returns:
        Dictionary mapping layer names to their SAE models
    """
    sae_models = {}

    # Find all model files matching the pattern <model_name>_sae_<layer_name>.pkl
    # but excluding files ending with _log.pkl or _final_metrics.pkl
    pattern = os.path.join(sae_dir, f"{model_name}_sae_*.pkl")
    model_files = glob.glob(pattern)

    # Define regex pattern to extract layer name and filter unwanted files
    regex_pattern = re.compile(f"{model_name}_sae_(.+)\\.pkl$")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading SAE models on {device}")

    for model_file in model_files:
        # Skip files containing "_log" or "_final_metrics" before .pkl
        if "_log.pkl" in model_file or "_final_metrics.pkl" in model_file:
            continue

        # Extract layer name from filename
        match = regex_pattern.search(os.path.basename(model_file))
        if match:
            layer_name = match.group(1)

            print(f"Loading SAE model for layer {layer_name} from {model_file}")
            try:
                sae_models[layer_name] = SAEModel.load(model_file).to(device)
            except Exception as e:
                print(f"Error loading SAE model from {model_file}: {e}")

    return sae_models

# LAYERS_DEFAULT = ["actor_mlp_3"]
SAE_LAYERS_DEFAULT = ["actor_mlp_1", "actor_mlp_3", "actor_mlp_5"]
