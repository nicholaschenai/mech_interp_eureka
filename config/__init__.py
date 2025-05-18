# Model configuration dictionary mapping model types to their filenames
MODEL_CONFIGS = {
    "strong": {"file": "FrankaCabinetGPT_epoch__eval.pth"},
    "medium": {"file": "FrankaCabinetGPT_epoch__iter1.pth"},
    "weak": {"file": "FrankaCabinetGPT_epoch__iter0.pth"}
}

# Default directories
DEFAULT_CHECKPOINT_DIR = './ckpts/2025-02-13_09-26-08'
DEFAULT_OUTPUT_DIR = './results'
