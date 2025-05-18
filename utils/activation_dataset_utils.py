import os

from typing import Optional, Dict

from src.activation_dataset import ActivationDataset


def load_model_dataset(ckpt_dir: str, model_type: str) -> Optional[ActivationDataset]:
    """Load activation dataset for a single model type.
    
    Args:
        ckpt_dir: Directory containing the checkpoint files
        model_type: Type of model ('strong', 'medium', or 'weak')
        
    Returns:
        ActivationDataset if found, None otherwise
    """
    dataset_path = os.path.join(ckpt_dir, f'{model_type}_activation_dataset.pkl')
    if not os.path.exists(dataset_path):
        print(f"Warning: Dataset for {model_type} model not found at {dataset_path}")
        return None

    print(f"Loading dataset for {model_type} model...")
    dataset = ActivationDataset()
    dataset.load(dataset_path)
    return dataset

def load_model_datasets(ckpt_dir: str) -> Dict[str, ActivationDataset]:
    """Load activation datasets for strong, medium, and weak models."""
    model_datasets = {}

    for model_type in ['strong', 'medium', 'weak']:
        dataset = load_model_dataset(ckpt_dir, model_type)
        if dataset is not None:
            model_datasets[model_type] = dataset

    return model_datasets
