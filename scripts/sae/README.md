# Sparse Autoencoder Analysis for Robot Control Model

This directory contains scripts for analyzing neural network activations using Sparse Autoencoders (SAEs) for mechanistic interpretability of the robot arm control model.

## Overview

The goal is to understand how the neural network represents and processes information during the drawer-opening task. We use SAEs to extract interpretable features from activation layers.

## Current Workflow

The SAE-based mechanistic interpretability workflow implemented in this codebase includes:

1. **Train SAE models** (train_sae.py)
   - Train SAEs on activation data from specific network layers
   - Output: Trained SAE model files (*.pkl)

2. **Analyze feature correlations** (input_sae_feature_analysis.py)
   - Compute correlations between input features and SAE features
   - Identify relationships between interpretable inputs and learned features
   - Visualize correlation patterns across different layers

3. **Collect causal intervention data** (collect_sae_intervention_data.py)
   - Load trained model and SAE models for all layers
   - Systematically ablate features in each source layer
   - Collect effects on all downstream layers and model outputs (mu, value)
   - Save raw data for later analysis

4. **Analyze intervention results** (analyze_sae_interventions.py)
   - Load pre-collected causal intervention data
   - Analyze how feature ablations affect downstream layers and outputs
   - Produce visualizations (causal matrices, feature importance rankings)
   - Identify important features and their effects
   - Optionally cluster features with similar causal patterns

## Available Scripts

- **train_sae.py**: Trains SAE models on neural network activations
  ```bash
  python scripts/sae/train_sae.py --dataset_path path/to/activations.pkl --output_dir results/sae_models
  ```

- **input_sae_feature_analysis.py**: Analyzes correlations between input features and SAE features
  ```bash
  python scripts/sae/input_sae_feature_analysis.py --input_data path/to/activation_dataset.pkl --sae_dir path/to/sae_dir --output_dir path/to/output
  ```

- **collect_sae_intervention_data.py**: Collects causal intervention data
  ```bash
  python scripts/sae/collect_sae_intervention_data.py \
    --model_path "FrankaCabinetGPT_epoch__eval.pth" \
    --checkpoint_dir "./ckpts/2025-02-13_09-26-08" \
    --output_dir "./results/causal_interventions/data" \
    --num_samples 100 \
    --max_features_per_layer 200
  ```

- **analyze_sae_interventions.py**: Analyzes and visualizes causal intervention data
  ```bash
  python scripts/sae/analyze_sae_interventions.py \
    --data_dir "./results/causal_interventions/data" \
    --output_dir "./results/causal_interventions/analysis" \
    --top_n_features 30 \
    --cluster_features
  ```

## Data Format

The collected causal intervention data follows this structure:

```
data_dir/
├── model_name_sae_features.pkl (baseline features with no ablation)
├── source_layer_1/
│   ├── feature_0.pkl
│   ├── feature_1.pkl
│   └── ...
├── source_layer_2/
│   ├── feature_0.pkl
│   └── ...
└── ...
```

Each feature_X.pkl file contains a dictionary with:
- features: Dictionary mapping layer names to feature activations
- mu: Action outputs after ablation
- value: Value outputs after ablation

## Training Stats

See `TRAINING_STATS.md` for detailed training statistics across different layers. Key metrics tracked include:
- Reconstruction loss
- Sparsity
- Dead features percentage
- Mean active features per sample


## Code Architecture

- `src/sae/sae_intervention.py` - Core intervention infrastructure
- `src/sae/sae_wrapper.py` - Model wrapping utilities for interventions
- `src/sae/sae_intervention_utils.py` - Visualization and analysis utilities
