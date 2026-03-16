# Usage

## Basic Usage

```python
import MDTerp

result = MDTerp.run(
    np_data=training_data,                          # (N, D) numpy array
    model_function_loc='model_script.txt',          # Path to model functions
    numeric_dict={'dist_1': [0], 'dist_2': [1]},   # Numeric features
    angle_dict={'phi': [2], 'psi': [3]},            # Angular features
    sin_cos_dict={'omega': [4, 5]},                 # Sin/cos feature pairs
    save_dir='./results/',
)
```

## Parameters

### Transition Detection

- **`point_max`** (default: 50): Target number of points to analyze per transition. MDTerp adaptively tunes the probability threshold per transition to achieve this target.
- **`prob_threshold_min`** (default: 0.475): Absolute floor for the probability threshold. If a transition has fewer than `point_max` samples even at this threshold, all available samples are used and a warning is issued. Should be close to but less than 0.50 to ensure linearity in the generated neighborhood.

### Parallelization

- **`n_workers`** (default: all CPUs): Number of parallel worker processes. Each worker loads the black-box model once and processes multiple points. Set to 1 for serial execution.

### Crash Recovery

- **`keep_checkpoints`** (default: True): Keep per-point `.npz` result files. When True, interrupted runs can be resumed by calling `MDTerp.run()` with the same `save_dir` — previously completed points are automatically detected and skipped.

### Analysis

- **`num_samples`** (default: 10000): Size of the perturbed neighborhood generated around each transition-state sample.
- **`cutoff`** (default: 15): Maximum features retained after the initial selection round.
- **`unfaithfulness_threshold`** (default: 0.01): Stopping criterion for forward feature selection.
- **`alpha`** (default: 1.0): Ridge regression L2 regularization strength.
- **`seed`** (default: 0): Random seed for reproducibility.

## Model Script Format

MDTerp requires a text file defining two functions:

```python
def load_model():
    # Load and return the trained black-box model
    import torch
    model = torch.load('my_model.pt')
    return model

def run_model(model, data):
    # Run model on data, return state probabilities (N, K) array
    # Rows sum to 1, K = number of metastable states
    return model.predict_proba(data)
```

## Analyzing Results

```python
# Transition-level summary
summary = MDTerp.transition_summary(
    'results/MDTerp_results_all.pkl',
    importance_coverage=0.8,
)

# Per-sample dominant feature
dominant = MDTerp.dominant_feature(
    'results/MDTerp_results_all.pkl',
    n=0,  # 0 = most important
)
```

## Visualizations

```python
# Feature importance bar plot
MDTerp.plot_feature_importance(
    'results/MDTerp_results_all.pkl',
    'results/MDTerp_feature_names.npy',
    transition='0_1',
    show_std=False,  # Default: no error bars
    top_n=10,
)

# Cross-transition heatmap
MDTerp.plot_importance_heatmap(
    'results/MDTerp_results_all.pkl',
    'results/MDTerp_feature_names.npy',
)

# Unfaithfulness curve (requires keep_checkpoints=True)
MDTerp.plot_unfaithfulness_curve(
    'results/', transition='0_1', point_index=0,
)

# Per-point variability
MDTerp.plot_point_variability(
    'results/MDTerp_results_all.pkl',
    'results/MDTerp_feature_names.npy',
    transition='0_1',
    top_n=5,
)
```
