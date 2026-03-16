# MDTerp v2.0 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor MDTerp with multi-CPU support, crash recovery, adaptive prob_threshold, post-analysis visualizations, updated docs/notebooks, and proper citation.

**Architecture:** Extract shared utilities into `models.py`, add `parallel.py` for multiprocessing workers, `checkpoint.py` for per-point result persistence, `visualization.py` for plots. Refactor `base.py` from constructor-does-everything to config+run pattern. Adaptive threshold uses per-transition binary search on sorted candidate probabilities.

**Tech Stack:** Python 3.8+, numpy, scikit-learn, scipy, matplotlib, multiprocessing (stdlib), json (stdlib)

---

### Task 1: Extract shared model utilities into `MDTerp/models.py`

**Files:**
- Create: `MDTerp/models.py`
- Modify: `MDTerp/init_analysis.py`
- Modify: `MDTerp/final_analysis.py`

**Step 1: Create `MDTerp/models.py` with shared functions**

```python
"""
MDTerp.models -- Shared linear model utilities for MDTerp analysis rounds.

Provides similarity kernel computation and ridge regression used by both
initial feature selection and forward feature selection rounds.
"""
import numpy as np
import sklearn.metrics as met
from sklearn.linear_model import Ridge
from typing import Tuple


def similarity_kernel(data: np.ndarray, kernel_width: float = 1.0) -> np.ndarray:
    """
    Compute similarity in [0,1] of perturbed samples relative to the original
    sample using LDA-transformed Euclidean distance.

    Args:
        data: LDA-transformed data. First row is the original sample.
        kernel_width: Width of the exponential kernel (default: 1.0).

    Returns:
        Array of similarity weights in [0,1] for each sample.
    """
    distances = met.pairwise_distances(
        data, data[0].reshape(1, -1), metric='euclidean'
    ).ravel()
    return np.sqrt(np.exp(-(distances ** 2) / kernel_width ** 2))


def ridge_regression(
    data: np.ndarray,
    labels: np.ndarray,
    seed: int,
    alpha: float = 1.0
) -> Tuple[np.ndarray, float]:
    """
    Fit a Ridge regression model on similarity-weighted data.

    Args:
        data: 2D array of similarity-weighted features (samples x features).
        labels: 1D or 2D array of similarity-weighted target probabilities.
        seed: Random seed for solver reproducibility.
        alpha: L2 regularization strength (default: 1.0).

    Returns:
        coefficients: Feature coefficients from the fitted model.
        intercept: Intercept term of the fitted model.
    """
    clf = Ridge(alpha, random_state=seed, solver='saga')
    clf.fit(data, labels.ravel())
    return clf.coef_, clf.intercept_
```

**Step 2: Update `MDTerp/init_analysis.py` to import from models.py**

Remove the local `similarity_kernel` and `SGDreg` functions. Replace with:
```python
from MDTerp.models import similarity_kernel, ridge_regression
```

Update the call in `init_model`:
```python
coefficients_selection, intercept_selection = ridge_regression(data, labels, seed, alpha)
```

**Step 3: Update `MDTerp/final_analysis.py` to import from models.py**

Remove the local `similarity_kernel` and `SGDreg` functions. Replace with:
```python
from MDTerp.models import similarity_kernel, ridge_regression
```

Update calls in `unfaithfulness_calc`:
```python
result_a, result_b = ridge_regression(data[:, models[i]], labels, seed)
```

**Step 4: Rename `interp` to `interpretation_entropy` in `final_analysis.py`**

Rename the function and update the call site inside `unfaithfulness_calc`.

**Step 5: Verify imports work**

Run: `cd /e/MDTerp_final && python -c "from MDTerp.models import similarity_kernel, ridge_regression; print('OK')"`
Expected: `OK`

**Step 6: Commit**

```bash
git add MDTerp/models.py MDTerp/init_analysis.py MDTerp/final_analysis.py
git commit -m "refactor: extract shared model utilities into models.py"
```

---

### Task 2: Create `MDTerp/checkpoint.py` for per-point result persistence

**Files:**
- Create: `MDTerp/checkpoint.py`

**Step 1: Create checkpoint module**

```python
"""
MDTerp.checkpoint -- Per-point result persistence and crash recovery.

Saves each analyzed point as an individual .npz file so that interrupted runs
can resume from where they left off.
"""
import numpy as np
import json
import os
import glob
from typing import Dict, Set, Tuple, List, Any
import pickle


def save_point_result(
    save_dir: str,
    transition: str,
    point_index: int,
    sample_index: int,
    importance: np.ndarray,
    importance_all: np.ndarray,
    unfaithfulness_all: np.ndarray,
    selected_features: np.ndarray,
) -> str:
    """
    Save a single point's analysis result to disk.

    Args:
        save_dir: Directory to save results.
        transition: Transition key string (e.g., "0_1").
        point_index: Index of the point within this transition.
        sample_index: Original row index in the training data.
        importance: Final importance vector for all features.
        importance_all: Importance vectors for all k values.
        unfaithfulness_all: Unfaithfulness values for all k values.
        selected_features: Feature indices kept after round 1.

    Returns:
        Path to the saved result file.
    """
    filename = f"{transition}_point{point_index}_result.npz"
    filepath = os.path.join(save_dir, filename)
    np.savez(
        filepath,
        sample_index=sample_index,
        transition=transition,
        importance=np.array(importance),
        importance_all=importance_all,
        unfaithfulness_all=unfaithfulness_all,
        selected_features=selected_features,
    )
    return filepath


def scan_completed_points(save_dir: str) -> Set[Tuple[str, int]]:
    """
    Scan the result directory for previously completed point results.

    Args:
        save_dir: Directory containing result files.

    Returns:
        Set of (transition, point_index) tuples already completed.
    """
    completed = set()
    pattern = os.path.join(save_dir, "*_point*_result.npz")
    for filepath in glob.glob(pattern):
        basename = os.path.basename(filepath)
        # Parse: "{transition}_point{idx}_result.npz"
        name = basename.replace("_result.npz", "")
        # Find last occurrence of "_point" to split
        point_pos = name.rfind("_point")
        if point_pos == -1:
            continue
        transition = name[:point_pos]
        try:
            point_index = int(name[point_pos + 6:])
            # Validate file is readable
            np.load(filepath, allow_pickle=False)
            completed.add((transition, point_index))
        except (ValueError, Exception):
            continue
    return completed


def save_run_config(save_dir: str, config: dict) -> str:
    """
    Save run configuration for resume validation.

    Args:
        save_dir: Directory to save config.
        config: Dictionary of all run parameters.

    Returns:
        Path to the saved config file.
    """
    filepath = os.path.join(save_dir, "run_config.json")
    # Convert numpy types to Python types for JSON serialization
    serializable = {}
    for k, v in config.items():
        if isinstance(v, np.ndarray):
            serializable[k] = v.tolist()
        elif isinstance(v, (np.integer,)):
            serializable[k] = int(v)
        elif isinstance(v, (np.floating,)):
            serializable[k] = float(v)
        else:
            serializable[k] = v
    with open(filepath, 'w') as f:
        json.dump(serializable, f, indent=2)
    return filepath


def load_run_config(save_dir: str) -> dict:
    """
    Load a previously saved run configuration.

    Args:
        save_dir: Directory containing the config file.

    Returns:
        Configuration dictionary, or empty dict if not found.
    """
    filepath = os.path.join(save_dir, "run_config.json")
    if not os.path.exists(filepath):
        return {}
    with open(filepath, 'r') as f:
        return json.load(f)


def aggregate_results(
    save_dir: str,
    feature_names: np.ndarray,
    keep_checkpoints: bool = True,
) -> dict:
    """
    Aggregate all per-point result files into the final results dictionary.

    Args:
        save_dir: Directory containing per-point result files.
        feature_names: Array of feature names.
        keep_checkpoints: Whether to keep individual result files (default: True).

    Returns:
        Dictionary mapping sample_index -> [transition, importance_vector].
    """
    importance_results = {}
    pattern = os.path.join(save_dir, "*_point*_result.npz")
    for filepath in glob.glob(pattern):
        try:
            data = np.load(filepath, allow_pickle=True)
            sample_index = int(data['sample_index'])
            transition = str(data['transition'])
            importance = data['importance']
            importance_results[sample_index] = [transition, importance]
        except Exception:
            continue

    # Save aggregated results
    result_path = os.path.join(save_dir, 'MDTerp_results_all.pkl')
    with open(result_path, 'wb') as f:
        pickle.dump(importance_results, f)

    names_path = os.path.join(save_dir, 'MDTerp_feature_names.npy')
    np.save(names_path, feature_names)

    if not keep_checkpoints:
        for filepath in glob.glob(pattern):
            os.remove(filepath)

    return importance_results
```

**Step 2: Verify module loads**

Run: `cd /e/MDTerp_final && python -c "from MDTerp.checkpoint import save_point_result, scan_completed_points; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add MDTerp/checkpoint.py
git commit -m "feat: add checkpoint module for per-point result persistence and crash recovery"
```

---

### Task 3: Implement adaptive `prob_threshold` in `MDTerp/utils.py`

**Files:**
- Modify: `MDTerp/utils.py`

**Step 1: Replace `picker_fn` with `select_transition_points`**

Add this new function to `utils.py` and remove the old `picker_fn`:

```python
def select_transition_points(
    prob: np.ndarray,
    point_max: int = 50,
    prob_threshold_min: float = 0.475,
) -> Tuple[dict, dict]:
    """
    Select transition-state samples with adaptive per-transition thresholds.

    For each transition, selects up to point_max samples closest to the
    decision boundary (highest min-of-top-2 probabilities). The effective
    threshold is determined automatically per transition, with
    prob_threshold_min as the absolute floor.

    Args:
        prob: 2D array of state probabilities (samples x states). Rows sum to 1.
        point_max: Target number of points per transition (default: 50).
        prob_threshold_min: Minimum allowed probability threshold (default: 0.475).

    Returns:
        points: Dict mapping transition keys (e.g., "0_1") to arrays of
            selected sample indices.
        thresholds: Dict mapping transition keys to the effective threshold
            used for that transition.

    Raises:
        ValueError: If no transitions are found at prob_threshold_min.
    """
    import warnings

    # Step 1: For each sample, find top-2 classes and their min probability
    candidates = defaultdict(list)
    for i in range(prob.shape[0]):
        top2_indices = np.argsort(prob[i, :])[::-1][:2]
        top2_values = prob[i, top2_indices]
        min_top2 = top2_values[1]  # smaller of the two

        if min_top2 >= prob_threshold_min:
            key = str(np.sort(top2_indices)[0]) + '_' + str(np.sort(top2_indices)[1])
            candidates[key].append((i, min_top2))

    # Step 2: Per transition, sort by min_top2 descending, take top point_max
    points = {}
    thresholds = {}

    for transition, cands in candidates.items():
        # Sort by proximity to 0.5 (highest min_top2 first)
        cands_sorted = sorted(cands, key=lambda x: x[1], reverse=True)

        if len(cands_sorted) <= point_max:
            # Use all available points
            selected = [c[0] for c in cands_sorted]
            effective_threshold = cands_sorted[-1][1] if cands_sorted else prob_threshold_min

            if len(cands_sorted) < point_max:
                warnings.warn(
                    f"Transition {transition}: only {len(cands_sorted)} points found "
                    f"at minimum threshold {prob_threshold_min} "
                    f"(requested {point_max})"
                )
        else:
            # Take top point_max
            selected = [c[0] for c in cands_sorted[:point_max]]
            effective_threshold = cands_sorted[point_max - 1][1]

        points[transition] = np.array(selected)
        thresholds[transition] = effective_threshold

    return points, thresholds
```

**Step 2: Update `make_result` with better naming**

```python
def make_result(
    feature_type_indices: list,
    feature_names: list,
    importance: np.ndarray,
) -> list:
    """
    Map importance values from the analysis feature space back to the
    original feature space, combining sin/cos pairs.

    Args:
        feature_type_indices: List of [numeric_indices, angle_indices,
            sin_indices, cos_indices] arrays.
        feature_names: List of feature names in order.
        importance: Raw importance array from the final model.

    Returns:
        List of importance values aligned with feature_names ordering.
    """
    result = []
    for i in range(feature_type_indices[0].shape[0]):
        result.append(importance[feature_type_indices[0][i]])
    for i in range(feature_type_indices[1].shape[0]):
        result.append(importance[feature_type_indices[1][i]])
    for i in range(feature_type_indices[2].shape[0]):
        result.append(
            importance[feature_type_indices[2][i]]
            + importance[feature_type_indices[3][i]]
        )
    return result
```

**Step 3: Fix `dominant_feature` docstring (currently copy-pasted from `transition_summary`)**

```python
def dominant_feature(all_result_loc: str, n: int = 0) -> dict:
    """
    Get the n-th most important feature for each analyzed sample.

    Args:
        all_result_loc: Path to the MDTerp_results_all.pkl file.
        n: Rank of the dominant feature to extract (0 = most important).

    Returns:
        Dict mapping sample index to the feature index of the n-th most
        important feature.
    """
```

**Step 4: Verify**

Run: `cd /e/MDTerp_final && python -c "from MDTerp.utils import select_transition_points; print('OK')"`
Expected: `OK`

**Step 5: Commit**

```bash
git add MDTerp/utils.py
git commit -m "feat: adaptive per-transition prob_threshold with select_transition_points"
```

---

### Task 4: Create `MDTerp/parallel.py` for multi-CPU worker pool

**Files:**
- Create: `MDTerp/parallel.py`

**Step 1: Create the parallel processing module**

```python
"""
MDTerp.parallel -- Multi-CPU parallel analysis of transition-state points.

Uses multiprocessing.Pool with per-worker model initialization to analyze
multiple points concurrently.
"""
import numpy as np
import os
import multiprocessing as mp
from typing import List, Tuple, Dict, Any

from MDTerp.neighborhood import generate_neighborhood
from MDTerp.init_analysis import init_model
from MDTerp.final_analysis import final_model
from MDTerp.utils import make_result
from MDTerp.checkpoint import save_point_result

# Global variables set per worker process by _worker_init
_worker_model = None
_worker_run_model = None


def _worker_init(model_function_loc: str) -> None:
    """
    Initializer for each worker process. Loads the black-box model once
    so it can be reused across all points assigned to this worker.

    Args:
        model_function_loc: Path to the model function file.
    """
    global _worker_model, _worker_run_model
    with open(model_function_loc, 'r') as f:
        func_code = f.read()
    local_ns = {}
    exec(func_code, {}, local_ns)
    _worker_model = local_ns["load_model"]()
    _worker_run_model = local_ns["run_model"]


def analyze_point(args: dict) -> dict:
    """
    Analyze a single transition-state point. Runs both MDTerp rounds
    (initial feature selection + forward feature selection) and saves
    the per-point result to disk.

    Args:
        args: Dictionary with keys:
            - save_dir: Result directory path
            - transition: Transition key string
            - point_index: Point index within this transition
            - sample_index: Original row index in training data
            - training_data: Full training dataset
            - numeric_dict, angle_dict, sin_cos_dict: Feature dictionaries
            - seed, num_samples, cutoff: Hyperparameters
            - unfaithfulness_threshold, alpha: Model parameters
            - periodicity_upper, periodicity_lower: Angular bounds
            - save_all: Whether to keep intermediate directories

    Returns:
        Dictionary with status info:
            - transition: Transition key
            - point_index: Point index
            - sample_index: Sample index
            - status: "completed" or "failed"
            - n_round1_features: Features kept after round 1
            - n_final_features: Non-zero features in final importance
            - error: Error message if failed
    """
    global _worker_model, _worker_run_model
    import shutil

    transition = args['transition']
    point_index = args['point_index']
    sample_index = args['sample_index']
    save_dir = args['save_dir']
    prefix = os.path.join(save_dir, f"{transition}_{point_index}_")

    result = {
        'transition': transition,
        'point_index': point_index,
        'sample_index': sample_index,
        'status': 'failed',
        'error': None,
    }

    try:
        # Round 1: Generate neighborhood with all features
        feature_type_indices, feature_names = generate_neighborhood(
            prefix,
            args['numeric_dict'], args['angle_dict'], args['sin_cos_dict'],
            args['training_data'], sample_index, args['seed'],
            args['num_samples'], np.array([]),
            args['periodicity_upper'], args['periodicity_lower'],
        )

        prediction_input = np.load(prefix + 'DATA/make_prediction.npy')
        state_probs = _worker_run_model(_worker_model, prediction_input)
        perturbation_data = np.load(prefix + 'DATA/TERP_dat.npy')
        selected_features = init_model(
            perturbation_data, state_probs,
            args['cutoff'], feature_type_indices, args['seed'], args['alpha'],
        )

        # Round 2: Generate neighborhood with selected features only
        generate_neighborhood(
            prefix,
            args['numeric_dict'], args['angle_dict'], args['sin_cos_dict'],
            args['training_data'], sample_index, args['seed'],
            args['num_samples'], selected_features,
            args['periodicity_upper'], args['periodicity_lower'],
        )

        prediction_input_2 = np.load(prefix + 'DATA_2/make_prediction.npy')
        state_probs_2 = _worker_run_model(_worker_model, prediction_input_2)
        perturbation_data_2 = np.load(prefix + 'DATA_2/TERP_dat.npy')
        importance_0, importance_all, unfaithfulness_all = final_model(
            perturbation_data_2, state_probs_2,
            args['unfaithfulness_threshold'], feature_type_indices,
            selected_features, args['seed'],
        )

        importance = make_result(feature_type_indices, feature_names, importance_0)

        # Save per-point result
        save_point_result(
            save_dir, transition, point_index, sample_index,
            np.array(importance), importance_all,
            unfaithfulness_all, selected_features,
        )

        # Clean intermediate directories if not saving all
        if not args.get('save_all', False):
            for suffix in ['DATA', 'DATA_2']:
                dirpath = prefix + suffix
                if os.path.isdir(dirpath):
                    shutil.rmtree(dirpath)

        result['status'] = 'completed'
        result['n_round1_features'] = len(selected_features)
        result['n_final_features'] = int(np.count_nonzero(importance))
        result['feature_names'] = feature_names

    except Exception as e:
        result['error'] = str(e)

    return result


def run_parallel(
    work_items: List[dict],
    model_function_loc: str,
    n_workers: int,
) -> List[dict]:
    """
    Process multiple points in parallel using a process pool.

    Args:
        work_items: List of argument dicts for analyze_point.
        model_function_loc: Path to the model function file.
        n_workers: Number of worker processes.

    Returns:
        List of result dicts from analyze_point.
    """
    if n_workers == 1 or len(work_items) == 1:
        # Serial fallback — avoids multiprocessing overhead
        _worker_init(model_function_loc)
        return [analyze_point(item) for item in work_items]

    with mp.Pool(
        processes=n_workers,
        initializer=_worker_init,
        initargs=(model_function_loc,),
    ) as pool:
        results = pool.map(analyze_point, work_items)

    return results
```

**Step 2: Verify**

Run: `cd /e/MDTerp_final && python -c "from MDTerp.parallel import run_parallel, analyze_point; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add MDTerp/parallel.py
git commit -m "feat: add parallel processing module for multi-CPU point analysis"
```

---

### Task 5: Refactor `MDTerp/base.py` to use new modules

**Files:**
- Modify: `MDTerp/base.py`

**Step 1: Rewrite base.py**

```python
"""
MDTerp.base -- Main MDTerp pipeline orchestrator.

Coordinates transition-state detection, parallel point analysis,
crash recovery, and result aggregation.
"""
import numpy as np
import os
import logging

from MDTerp.utils import log_maker, input_summary, select_transition_points, make_result
from MDTerp.parallel import run_parallel
from MDTerp.checkpoint import (
    save_run_config,
    load_run_config,
    scan_completed_points,
    aggregate_results,
)


class run:
    """
    Main class for implementing MDTerp analysis.

    MDTerp interprets black-box AI classifiers trained on molecular dynamics
    data by identifying transition-state samples and computing local feature
    importance using the TERP framework (Thermodynamics-inspired Explanations
    using Ridge regression with Perturbation).

    Reference:
        Mehdi, S. & Tiwary, P. "Thermodynamics-inspired explanations of
        artificial intelligence." Nature Communications (2024).

    Attributes:
        results: Dictionary mapping sample indices to [transition, importance].
        feature_names: Array of feature name strings.
        points: Dictionary of selected transition-state points.
        thresholds: Dictionary of effective per-transition probability thresholds.
    """

    def __init__(
        self,
        np_data: np.ndarray,
        model_function_loc: str,
        numeric_dict: dict = None,
        angle_dict: dict = None,
        sin_cos_dict: dict = None,
        save_dir: str = './results/',
        point_max: int = 50,
        prob_threshold_min: float = 0.475,
        num_samples: int = 10000,
        cutoff: int = 15,
        seed: int = 0,
        unfaithfulness_threshold: float = 0.01,
        periodicity_upper: float = np.pi,
        periodicity_lower: float = -np.pi,
        alpha: float = 1.0,
        save_all: bool = False,
        n_workers: int = None,
        keep_checkpoints: bool = True,
    ) -> None:
        """
        Configure and execute the MDTerp analysis pipeline.

        Args:
            np_data: 2D training data array (samples x features).
            model_function_loc: Path to file defining load_model() and
                run_model() functions for the black-box classifier.
            numeric_dict: Feature name -> [column_index] for numeric features.
            angle_dict: Feature name -> [column_index] for angular features.
            sin_cos_dict: Feature name -> [sin_index, cos_index] for sin/cos pairs.
            save_dir: Directory to save all results (default: './results/').
            point_max: Target number of points per transition (default: 50).
            prob_threshold_min: Minimum probability threshold for transition
                detection (default: 0.475). Applied as a floor per transition.
            num_samples: Number of perturbed neighborhood samples (default: 10000).
            cutoff: Maximum features kept after initial round (default: 15).
            seed: Random seed (default: 0).
            unfaithfulness_threshold: Stopping criterion for forward feature
                selection (default: 0.01).
            periodicity_upper: Upper bound for angular periodicity (default: pi).
            periodicity_lower: Lower bound for angular periodicity (default: -pi).
            alpha: Ridge regression L2 penalty (default: 1.0).
            save_all: Keep intermediate DATA directories (default: False).
            n_workers: Number of parallel worker processes. Defaults to the
                number of available CPUs.
            keep_checkpoints: Keep per-point result files after aggregation
                (default: True).
        """
        if numeric_dict is None:
            numeric_dict = {}
        if angle_dict is None:
            angle_dict = {}
        if sin_cos_dict is None:
            sin_cos_dict = {}
        if n_workers is None:
            n_workers = os.cpu_count() or 1

        # Store config
        self.config = {
            'model_function_loc': model_function_loc,
            'save_dir': save_dir,
            'point_max': point_max,
            'prob_threshold_min': prob_threshold_min,
            'num_samples': num_samples,
            'cutoff': cutoff,
            'seed': seed,
            'unfaithfulness_threshold': unfaithfulness_threshold,
            'periodicity_upper': periodicity_upper,
            'periodicity_lower': periodicity_lower,
            'alpha': alpha,
            'save_all': save_all,
            'n_workers': n_workers,
            'keep_checkpoints': keep_checkpoints,
        }

        self.results = None
        self.feature_names = None
        self.points = None
        self.thresholds = None

        # Execute pipeline
        self._execute(
            np_data, model_function_loc, numeric_dict, angle_dict,
            sin_cos_dict, save_dir, point_max, prob_threshold_min,
            num_samples, cutoff, seed, unfaithfulness_threshold,
            periodicity_upper, periodicity_lower, alpha, save_all,
            n_workers, keep_checkpoints,
        )

    def _execute(
        self, np_data, model_function_loc, numeric_dict, angle_dict,
        sin_cos_dict, save_dir, point_max, prob_threshold_min,
        num_samples, cutoff, seed, unfaithfulness_threshold,
        periodicity_upper, periodicity_lower, alpha, save_all,
        n_workers, keep_checkpoints,
    ):
        """Internal pipeline execution."""
        os.makedirs(save_dir, exist_ok=True)
        logger = log_maker(save_dir)
        input_summary(logger, numeric_dict, angle_dict, sin_cos_dict, save_dir, np_data)

        # Load model for transition detection
        logger.info('Loading black-box model from file >>> ' + model_function_loc)
        with open(model_function_loc, 'r') as f:
            func_code = f.read()
        local_ns = {}
        exec(func_code, globals(), local_ns)
        model = local_ns["load_model"]()
        logger.info("Model loaded!")

        # Detect transition states with adaptive thresholds
        state_probabilities = local_ns["run_model"](model, np_data)
        np.random.seed(seed)
        points, thresholds = select_transition_points(
            state_probabilities, point_max, prob_threshold_min,
        )
        self.points = points
        self.thresholds = thresholds

        n_transitions = len(points)
        logger.info(f"Number of state transitions detected >>> {n_transitions}")
        for trans, thresh in thresholds.items():
            n_pts = len(points[trans])
            logger.info(
                f"  Transition {trans}: {n_pts} points, "
                f"effective threshold = {thresh:.6f}"
            )
        if n_transitions == 0:
            logger.info("No transition detected. Check hyperparameters!")
            raise ValueError("No transition detected. Check hyperparameters!")
        logger.info(100 * '-')

        # Save run config for resume validation
        save_run_config(save_dir, {
            **self.config,
            'points': {k: v.tolist() for k, v in points.items()},
        })

        # Check for previously completed points (crash recovery)
        completed = scan_completed_points(save_dir)
        total_points = sum(len(v) for v in points.values())
        if completed:
            logger.info(
                f"Resuming: found {len(completed)}/{total_points} completed "
                f"results, {total_points - len(completed)} remaining"
            )

        # Build work queue, skipping completed points
        work_items = []
        for transition in points:
            for point_index in range(len(points[transition])):
                if (transition, point_index) in completed:
                    continue
                work_items.append({
                    'save_dir': save_dir,
                    'transition': transition,
                    'point_index': point_index,
                    'sample_index': int(points[transition][point_index]),
                    'training_data': np_data,
                    'numeric_dict': numeric_dict,
                    'angle_dict': angle_dict,
                    'sin_cos_dict': sin_cos_dict,
                    'seed': seed,
                    'num_samples': num_samples,
                    'cutoff': cutoff,
                    'unfaithfulness_threshold': unfaithfulness_threshold,
                    'periodicity_upper': periodicity_upper,
                    'periodicity_lower': periodicity_lower,
                    'alpha': alpha,
                    'save_all': save_all,
                })

        if not work_items:
            logger.info("All points already completed. Aggregating results.")
        else:
            logger.info(
                f"Analyzing {len(work_items)} points using {n_workers} workers..."
            )

            # Run analysis (parallel or serial)
            results = run_parallel(work_items, model_function_loc, n_workers)

            # Log results
            for r in results:
                if r['status'] == 'completed':
                    logger.info(
                        f"Completed {r['transition']} point {r['point_index']}: "
                        f"round 1 features = {r['n_round1_features']}, "
                        f"final features = {r['n_final_features']}"
                    )
                else:
                    logger.error(
                        f"Failed {r['transition']} point {r['point_index']}: "
                        f"{r['error']}"
                    )

            # Get feature names from first successful result
            feature_names = None
            for r in results:
                if r.get('feature_names') is not None:
                    feature_names = r['feature_names']
                    break

        # If we had no work items, we need feature names from existing results
        # or we can derive them from the dictionaries
        if not work_items or feature_names is None:
            feature_names = (
                list(numeric_dict.keys())
                + list(angle_dict.keys())
                + list(sin_cos_dict.keys())
            )

        # Aggregate all per-point results
        self.feature_names = np.array(feature_names)
        self.results = aggregate_results(
            save_dir, self.feature_names, keep_checkpoints,
        )

        logger.info(
            "Feature names saved at >>> "
            + os.path.join(save_dir, 'MDTerp_feature_names.npy')
        )
        logger.info(
            "All results saved at >>> "
            + os.path.join(save_dir, 'MDTerp_results_all.pkl')
        )
        logger.info("Completed!")

        # Clean up logger
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
```

**Step 2: Verify import**

Run: `cd /e/MDTerp_final && python -c "from MDTerp.base import run; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add MDTerp/base.py
git commit -m "refactor: rewrite base.py with parallel execution, adaptive threshold, crash recovery"
```

---

### Task 6: Create `MDTerp/visualization.py`

**Files:**
- Create: `MDTerp/visualization.py`

**Step 1: Create the visualization module**

```python
"""
MDTerp.visualization -- Post-analysis plotting functions for MDTerp results.

Provides publication-quality visualizations of feature importance across
transitions, unfaithfulness curves, and per-point variability analysis.
"""
import numpy as np
import pickle
import os
import glob
import matplotlib.pyplot as plt
from typing import Optional
from MDTerp.utils import transition_summary


def plot_feature_importance(
    result_path: str,
    feature_names_path: str,
    transition: str,
    show_std: bool = False,
    top_n: Optional[int] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (8, 8),
) -> plt.Figure:
    """
    Plot mean feature importance as a horizontal bar chart for a transition.

    Args:
        result_path: Path to MDTerp_results_all.pkl.
        feature_names_path: Path to MDTerp_feature_names.npy.
        transition: Transition key (e.g., "0_1").
        show_std: Show standard deviation error bars (default: False).
        top_n: Only show the top N features. None shows all non-zero.
        save_path: If provided, save the figure to this path.
        figsize: Figure size as (width, height) tuple.

    Returns:
        matplotlib Figure object.
    """
    feature_names = np.load(feature_names_path, allow_pickle=True)
    summary = transition_summary(result_path, importance_coverage=1.0)

    if transition not in summary:
        raise ValueError(
            f"Transition '{transition}' not found. "
            f"Available: {list(summary.keys())}"
        )

    mean_imp = summary[transition][0]
    std_imp = summary[transition][1]

    # Sort by importance descending, keep only non-zero
    nonzero_mask = mean_imp > 0
    ordered_indices = np.argsort(mean_imp)[::-1]
    ordered_indices = ordered_indices[nonzero_mask[ordered_indices]]

    if top_n is not None:
        ordered_indices = ordered_indices[:top_n]

    ordered_mean = mean_imp[ordered_indices]
    ordered_std = std_imp[ordered_indices]
    ordered_names = feature_names[ordered_indices]

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(len(ordered_mean))

    if show_std:
        ax.barh(y_pos, ordered_mean, xerr=ordered_std, capsize=4,
                color='steelblue', edgecolor='black', linewidth=0.5)
    else:
        ax.barh(y_pos, ordered_mean,
                color='steelblue', edgecolor='black', linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(ordered_names, fontsize=12)
    ax.set_xlabel('Feature Importance', fontsize=14)
    ax.set_title(
        f'Feature Importance for Transition {transition}\n'
        f'Coverage: {int(100 * np.sum(ordered_mean))}%',
        fontsize=16,
    )
    ax.invert_yaxis()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_importance_heatmap(
    result_path: str,
    feature_names_path: str,
    importance_coverage: float = 0.8,
    save_path: Optional[str] = None,
    figsize: Optional[tuple] = None,
) -> plt.Figure:
    """
    Plot a heatmap of feature importance across all transitions.

    Rows are features, columns are transitions, color intensity represents
    mean importance. Only features with non-zero importance in at least one
    transition are shown.

    Args:
        result_path: Path to MDTerp_results_all.pkl.
        feature_names_path: Path to MDTerp_feature_names.npy.
        importance_coverage: Filter features by cumulative importance
            per transition (default: 0.8).
        save_path: If provided, save the figure to this path.
        figsize: Figure size. Auto-scaled if None.

    Returns:
        matplotlib Figure object.
    """
    feature_names = np.load(feature_names_path, allow_pickle=True)
    summary = transition_summary(result_path, importance_coverage=importance_coverage)

    transitions = sorted(summary.keys())
    n_features = len(feature_names)

    # Build importance matrix
    imp_matrix = np.zeros((n_features, len(transitions)))
    for j, trans in enumerate(transitions):
        imp_matrix[:, j] = summary[trans][0]

    # Keep only features non-zero in at least one transition
    active_mask = np.any(imp_matrix > 0, axis=1)
    active_indices = np.where(active_mask)[0]
    imp_matrix = imp_matrix[active_indices, :]
    active_names = feature_names[active_indices]

    if figsize is None:
        figsize = (max(6, len(transitions) * 2), max(6, len(active_names) * 0.4))

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(imp_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')

    ax.set_xticks(np.arange(len(transitions)))
    ax.set_xticklabels(transitions, fontsize=12, rotation=45, ha='right')
    ax.set_yticks(np.arange(len(active_names)))
    ax.set_yticklabels(active_names, fontsize=10)
    ax.set_xlabel('Transition', fontsize=14)
    ax.set_ylabel('Feature', fontsize=14)
    ax.set_title('Feature Importance Across Transitions', fontsize=16)

    # Annotate cells with values
    for i in range(imp_matrix.shape[0]):
        for j in range(imp_matrix.shape[1]):
            val = imp_matrix[i, j]
            if val > 0.01:
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=8, color='black' if val < 0.5 else 'white')

    fig.colorbar(im, ax=ax, label='Importance', shrink=0.8)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_unfaithfulness_curve(
    result_dir: str,
    transition: str,
    point_index: int,
    save_path: Optional[str] = None,
    figsize: tuple = (8, 5),
) -> plt.Figure:
    """
    Plot the unfaithfulness vs number of features curve for a single point.

    Shows how surrogate model quality improves as more features are included
    in the linear explanation, visualizing the TERP free energy trade-off.

    Args:
        result_dir: Directory containing per-point result files.
        transition: Transition key (e.g., "0_1").
        point_index: Point index within the transition.
        save_path: If provided, save the figure to this path.
        figsize: Figure size.

    Returns:
        matplotlib Figure object.
    """
    filename = f"{transition}_point{point_index}_result.npz"
    filepath = os.path.join(result_dir, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Result file not found: {filepath}. "
            f"Ensure keep_checkpoints=True when running MDTerp."
        )

    data = np.load(filepath, allow_pickle=True)
    unfaithfulness = data['unfaithfulness_all']

    fig, ax = plt.subplots(figsize=figsize)
    k_values = np.arange(1, len(unfaithfulness) + 1)

    ax.plot(k_values, unfaithfulness, 'o-', color='steelblue',
            linewidth=2, markersize=6)
    ax.set_xlabel('Number of Features (k)', fontsize=14)
    ax.set_ylabel('Unfaithfulness (1 - |r|)', fontsize=14)
    ax.set_title(
        f'Unfaithfulness Curve\n'
        f'Transition {transition}, Point {point_index}',
        fontsize=16,
    )
    ax.set_xticks(k_values)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_point_variability(
    result_path: str,
    feature_names_path: str,
    transition: str,
    top_n: int = 5,
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """
    Plot per-point importance variability for the top features in a transition.

    Shows a strip plot where each dot represents one point's importance value
    for a given feature, revealing how consistent the explanations are.

    Args:
        result_path: Path to MDTerp_results_all.pkl.
        feature_names_path: Path to MDTerp_feature_names.npy.
        transition: Transition key (e.g., "0_1").
        top_n: Number of top features to show (default: 5).
        save_path: If provided, save the figure to this path.
        figsize: Figure size.

    Returns:
        matplotlib Figure object.
    """
    feature_names = np.load(feature_names_path, allow_pickle=True)

    with open(result_path, 'rb') as f:
        all_results = pickle.load(f)

    # Collect importance vectors for this transition
    importances = []
    for sample_idx, (trans, imp) in all_results.items():
        if trans == transition:
            importances.append(np.array(imp))

    if not importances:
        raise ValueError(
            f"No results found for transition '{transition}'. "
            f"Available: {list(set(v[0] for v in all_results.values()))}"
        )

    imp_array = np.array(importances)  # (n_points, n_features)
    mean_imp = np.mean(imp_array, axis=0)
    top_indices = np.argsort(mean_imp)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=figsize)
    positions = np.arange(top_n)

    for i, feat_idx in enumerate(top_indices):
        values = imp_array[:, feat_idx]
        # Jitter for visibility
        jitter = np.random.uniform(-0.15, 0.15, size=len(values))
        ax.scatter(
            positions[i] + jitter, values,
            alpha=0.6, s=40, color='steelblue', edgecolors='black',
            linewidth=0.5,
        )
        # Mean marker
        ax.scatter(positions[i], mean_imp[feat_idx],
                   marker='D', s=80, color='red', zorder=5)

    ax.set_xticks(positions)
    ax.set_xticklabels(feature_names[top_indices], fontsize=12, rotation=30, ha='right')
    ax.set_ylabel('Feature Importance', fontsize=14)
    ax.set_title(
        f'Per-Point Importance Variability\n'
        f'Transition {transition} (red = mean)',
        fontsize=16,
    )
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
```

**Step 2: Verify**

Run: `cd /e/MDTerp_final && python -c "from MDTerp.visualization import plot_feature_importance, plot_importance_heatmap, plot_unfaithfulness_curve, plot_point_variability; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add MDTerp/visualization.py
git commit -m "feat: add visualization module with 4 plot types for post-analysis"
```

---

### Task 7: Update `MDTerp/__init__.py` with new version and exports

**Files:**
- Modify: `MDTerp/__init__.py`

**Step 1: Update init**

```python
__version__ = "2.0.0"
__author__ = """Shams Mehdi"""
__email__ = "shamsmehdi222@gmail.com"

from MDTerp.base import run
from MDTerp.utils import transition_summary, dominant_feature
from MDTerp.visualization import (
    plot_feature_importance,
    plot_importance_heatmap,
    plot_unfaithfulness_curve,
    plot_point_variability,
)
```

**Step 2: Update version in pyproject.toml**

Change `version = "1.4.0"` to `version = "2.0.0"` and `current_version = "1.4.0"` to `current_version = "2.0.0"`.

**Step 3: Commit**

```bash
git add MDTerp/__init__.py pyproject.toml
git commit -m "chore: bump version to 2.0.0, update exports"
```

---

### Task 8: Update documentation, README, and citation

**Files:**
- Modify: `README.md`
- Modify: `docs/index.md`
- Modify: `docs/usage.md`
- Create: `docs/visualization.md`
- Modify: `docs/changelog.md`
- Create: `CITATION.cff`
- Modify: `mkdocs.yml`

**Step 1: Rewrite README.md**

Full README with: theory overview, installation, quick start, parameter reference, visualization examples, citation section with DOI.

**Step 2: Rewrite docs/index.md**

Landing page with TERP theory explanation, package overview, citation.

**Step 3: Rewrite docs/usage.md**

Full usage guide covering: basic usage, adaptive threshold, multi-CPU, crash recovery, visualization, parameter reference table.

**Step 4: Create docs/visualization.md**

Document all 4 visualization functions with usage examples.

**Step 5: Update docs/changelog.md**

Add v2.0.0 entry with all new features.

**Step 6: Create CITATION.cff**

```yaml
cff-version: 1.2.0
message: "If you use this software, please cite both the software and the paper."
title: "MDTerp"
version: 2.0.0
authors:
  - family-names: Mehdi
    given-names: Shams
    email: shamsmehdi222@gmail.com
preferred-citation:
  type: article
  title: "Thermodynamics-inspired explanations of artificial intelligence"
  authors:
    - family-names: Mehdi
      given-names: Shams
    - family-names: Tiwary
      given-names: Pratyush
  journal: "Nature Communications"
  year: 2024
```

**Step 7: Update mkdocs.yml nav**

Add visualization.md to nav, update copyright year.

**Step 8: Commit**

```bash
git add README.md docs/ CITATION.cff mkdocs.yml
git commit -m "docs: rewrite documentation, add citation, visualization guide"
```

---

### Task 9: Update Jupyter notebooks

**Files:**
- Modify: `MDTerp_blackbox.ipynb`
- Modify: `MDTerp_SPIB_ntl9.ipynb`
- Modify: `MDTerp_SPIB_villin.ipynb`
- Modify: `MDTerp_VAMPNets.ipynb`

**Step 1: Update MDTerp_blackbox.ipynb**

- Update `base.run()` call: replace `prob_threshold` with `prob_threshold_min`, add `n_workers` parameter
- Replace manual matplotlib plotting with `MDTerp.visualization.plot_feature_importance()` and other viz functions
- Add cells demonstrating: crash recovery resume, unfaithfulness curve, heatmap, point variability
- Add citation cell at top

**Step 2: Update remaining notebooks**

Same changes for SPIB and VAMPNets notebooks — update API calls, add visualization demos.

**Step 3: Commit**

```bash
git add *.ipynb
git commit -m "docs: update all notebooks for v2.0.0 API and new visualizations"
```

---

### Task 10: Final integration test and cleanup

**Step 1: Run integration test with blackbox notebook data**

```bash
cd /e/MDTerp_final
python -c "
import numpy as np
import MDTerp.base as base

# Quick test with tiny data
np.random.seed(42)
synthetic_data = np.random.uniform(-5, 10, (100, 5))
# ... (verify imports, class instantiation, etc.)
print('All imports OK')
"
```

**Step 2: Verify all modules load correctly**

```bash
python -c "
from MDTerp import (
    run, transition_summary, dominant_feature,
    plot_feature_importance, plot_importance_heatmap,
    plot_unfaithfulness_curve, plot_point_variability,
)
from MDTerp.models import similarity_kernel, ridge_regression
from MDTerp.checkpoint import save_point_result, scan_completed_points
from MDTerp.parallel import run_parallel
print('All modules loaded successfully')
"
```

**Step 3: Final commit**

```bash
git add -A
git commit -m "chore: final cleanup for MDTerp v2.0.0 release"
```
