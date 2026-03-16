# MDTerp v2.0 Design Document

**Date:** 2026-03-16
**Author:** Shams Mehdi + Claude

## Overview

Major refactor and feature addition to MDTerp, the Python implementation of TERP
(Thermodynamics-inspired Explanations using Ridge regression with Perturbation)
from the Nature Communications paper by Mehdi & Tiwary.

## 1. Refactoring

### Module restructuring
- Extract shared `similarity_kernel` and `ridge_regression` (renamed from `SGDreg`) into `MDTerp/models.py`
- New `MDTerp/parallel.py` — worker functions and pool management
- New `MDTerp/visualization.py` — post-analysis plotting
- New `MDTerp/checkpoint.py` — per-point result saving and crash recovery

### Variable naming cleanup
| Old | New |
|-----|-----|
| `TERP_dat` | `perturbation_data` |
| `TERP_SGD_*` | `ridge_*` |
| `SGDreg` | `ridge_regression` |
| `make_prediction` | `prediction_input` |
| `importance_master` | `importance_results` |
| `picker_fn` | `select_transition_points` |
| `np_dat`/`np_data` | `training_data` |
| `interp` | `interpretation_entropy` |

### Class refactor
- `run` class: constructor stores config, `.run()` method executes pipeline
- Makes the class testable and configurable

## 2. Multi-CPU Parallelization

- Flatten all `(transition, point_index, sample_index)` tuples into a single work queue
- Use `multiprocessing.Pool` with initializer that loads model once per worker process
- New parameter `n_workers` (default: `os.cpu_count()`)
- Each worker runs both analysis rounds for one point, saves per-point result file
- No file contention — unique filenames per worker: `{transition}_point{idx}_result.npz`

### Worker design
```python
def _worker_init(model_function_loc):
    """Load model once per worker process into global."""
    global _worker_model, _worker_run_model
    ...

def _analyze_point(args):
    """Process one (transition, point) pair using pre-loaded model."""
    # 1. Generate neighborhood (round 1)
    # 2. Run model on neighborhood
    # 3. Initial feature selection
    # 4. Generate neighborhood (round 2, selected features only)
    # 5. Run model on neighborhood
    # 6. Forward feature selection
    # 7. Save per-point result file
    # 8. Return status
```

## 3. Crash Recovery

### On startup
1. Scan `save_dir` for existing `{transition}_point{idx}_result.npz` files
2. Build set of completed `(transition, point_index)` pairs
3. Filter work queue to exclude completed pairs
4. Log: "Resuming: found X/Y completed, Z remaining"

### Per-point result file (`{transition}_point{idx}_result.npz`)
- `sample_index`: original data row index
- `transition`: transition string
- `importance`: final importance vector
- `importance_all`: importance for all k values
- `unfaithfulness_all`: unfaithfulness curve
- `selected_features`: features kept after round 1

### Config validation
- Save `run_config.json` at start with all parameters and selected points
- On resume, validate config matches — warn if not

### Final aggregation
- Assemble `MDTerp_results_all.pkl` from individual files
- Parameter `keep_checkpoints` (default: `True`)

## 4. Adaptive prob_threshold

### Per-transition algorithm
1. For each sample, identify top-2 classes and probabilities
2. Group by transition key
3. Per transition:
   - Sort candidates by `min(top2_probs)` descending (closest to 0.50 first)
   - Take top `point_max` if enough candidates exist
   - If fewer, lower threshold until `point_max` reached or `prob_threshold_min` hit
   - If at floor and still < `point_max`: use all available, log warning
   - If zero at floor: skip transition with warning

### Parameters
- `point_max` (default: 50)
- `prob_threshold_min` (default: 0.475) — absolute floor, user-configurable, same for all transitions
- Old `prob_threshold` parameter removed

## 5. Visualizations

All in `MDTerp/visualization.py`, returning matplotlib figures.

1. **`plot_feature_importance()`** — horizontal bar plot, mean importance per feature for a transition. `show_std=False` by default.
2. **`plot_importance_heatmap()`** — features vs transitions, colored by importance, filtered by `importance_coverage`.
3. **`plot_unfaithfulness_curve()`** — elbow plot from per-point `unfaithfulness_all.npy`.
4. **`plot_point_variability()`** — strip/swarm plot showing per-point importance spread for top N features within a transition.

## 6. Documentation & Citation

- Full citation of "Thermodynamics-inspired explanations of artificial intelligence" (Mehdi & Tiwary, Nature Communications)
- `CITATION.cff` for GitHub citation
- Rewritten `README.md`, `docs/usage.md`, `docs/index.md`
- New `docs/visualization.md`
- Updated `docs/changelog.md` with v2.0.0 entry
- Fix docstring inaccuracies (e.g., "SGD" → "Ridge regression")
- Copyright update to 2024-2026
